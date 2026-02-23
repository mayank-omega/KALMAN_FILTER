"""
FastAPI Tracking Engine
========================
Real-time tracking endpoint that processes streaming measurements
and maintains filter state between requests.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import numpy as np
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.kalman_filter import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from core.models import ConstantVelocityModel, ConstantAccelerationModel, CoordinatedTurnModel

app = FastAPI(
    title="Kalman Tracking Engine",
    description="""
    Production-grade real-time object tracking via Kalman Filter variants.
    
    Supports:
    - Standard KF (linear systems)
    - EKF (mildly nonlinear)
    - UKF (strongly nonlinear, derivative-free)
    
    Motion models: CV, CA, Coordinated Turn
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Request / Response Models ────────────────────────────────────────────────

class TrackConfig(BaseModel):
    filter_type: Literal["KF", "EKF", "UKF"] = Field("UKF", description="Filter algorithm")
    motion_model: Literal["CV", "CA", "CT"] = Field("CV", description="Motion model")
    dt: float = Field(0.1, gt=0, description="Timestep [s]")
    sigma_process: float = Field(1.0, gt=0, description="Process noise std")
    sigma_measurement: float = Field(5.0, gt=0, description="Measurement noise std [m]")
    initial_position: List[float] = Field([0.0, 0.0], description="Initial [x, y]")
    initial_velocity: List[float] = Field([1.0, 0.0], description="Initial [vx, vy]")
    outlier_threshold: Optional[float] = Field(9.21, description="Mahalanobis gating (None=off)")
    adaptive_noise: bool = Field(False, description="Sage-Husa adaptive noise estimation")


class MeasurementRequest(BaseModel):
    track_id: str
    measurement: List[float] = Field(..., description="[x, y] measurement")
    missing: bool = Field(False, description="Mark measurement as missing")


class StateResponse(BaseModel):
    track_id: str
    state_estimate: List[float]
    covariance_diagonal: List[float]
    innovation: List[float]
    nis: float
    log_likelihood: float
    step: int
    filter_type: str
    motion_model: str


class BatchRequest(BaseModel):
    measurements: List[List[float]]
    missing_mask: Optional[List[bool]] = None
    config: TrackConfig


class BatchResponse(BaseModel):
    states_prior: List[List[float]]
    states_posterior: List[List[float]]
    covariances: List[List[float]]
    innovations: List[List[float]]
    nis_sequence: List[float]
    rmse_position: float
    mean_nees: Optional[float] = None


# ─── Track Registry ───────────────────────────────────────────────────────────

class TrackRegistry:
    """Thread-safe registry of active filter instances"""
    
    def __init__(self):
        self._tracks: Dict[str, Dict] = {}
    
    def create(self, track_id: str, config: TrackConfig):
        model, filt = self._build(config)
        self._tracks[track_id] = {
            'filter': filt,
            'model': model,
            'config': config,
            'step': 0,
            'history': []
        }
    
    def get(self, track_id: str):
        if track_id not in self._tracks:
            raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
        return self._tracks[track_id]
    
    def delete(self, track_id: str):
        self._tracks.pop(track_id, None)
    
    def list_tracks(self):
        return list(self._tracks.keys())
    
    def _build(self, config: TrackConfig):
        dt = config.dt
        sp = config.sigma_process
        sm = config.sigma_measurement
        
        # Build motion model
        if config.motion_model == "CV":
            model = ConstantVelocityModel(dt=dt, sigma_a=sp)
        elif config.motion_model == "CA":
            model = ConstantAccelerationModel(dt=dt, sigma_j=sp)
        else:
            model = CoordinatedTurnModel(dt=dt, sigma_a=sp)
        
        n = model.n_state
        m = model.n_meas
        R = sm**2 * np.eye(m)
        
        # Initial state
        x0 = np.zeros(n)
        x0[:2] = config.initial_position
        x0[2:4] = config.initial_velocity
        P0 = np.diag([10.0]*2 + [5.0]*2 + [1.0]*(n-4)) if n > 4 else np.diag([10.0]*2 + [5.0]*2)
        
        # Build filter
        if config.filter_type == "KF" and config.motion_model in ["CV", "CA"]:
            filt = KalmanFilter(
                F=model.F, H=model.H, Q=model.Q, R=R,
                x0=x0, P0=P0, adaptive=config.adaptive_noise
            )
        elif config.filter_type == "EKF":
            filt = ExtendedKalmanFilter(
                f=model.f, h=model.h,
                F_jac=model.F_jacobian, H_jac=model.H_jacobian,
                Q=model.Q, R=R, x0=x0, P0=P0
            )
        else:  # UKF (default for CT)
            filt = UnscentedKalmanFilter(
                f=model.f, h=model.h,
                Q=model.Q, R=R, x0=x0, P0=P0
            )
        
        return model, filt


registry = TrackRegistry()

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Kalman Tracking Engine v1.0"}


@app.post("/tracks/{track_id}", tags=["Tracks"])
def create_track(track_id: str, config: TrackConfig):
    """Initialize a new Kalman filter track."""
    registry.create(track_id, config)
    obs, rank = registry.get(track_id)['model'].is_observable() if hasattr(registry.get(track_id)['model'], 'is_observable') else (True, None)
    return {
        "track_id": track_id,
        "status": "created",
        "filter": config.filter_type,
        "model": config.motion_model,
        "observable": obs,
        "observability_rank": rank
    }


@app.post("/tracks/{track_id}/update", response_model=StateResponse, tags=["Tracking"])
def update_track(track_id: str, req: MeasurementRequest):
    """
    Process one measurement and return updated state estimate.
    
    Runs predict + update cycle.
    """
    track = registry.get(track_id)
    filt = track['filter']
    config = track['config']
    
    filt.predict()
    
    if req.missing:
        x_post = filt.x_prior.copy()
        P_post = filt.P_prior.copy()
        filt.x = x_post
        filt.P = P_post
        innov = [0.0, 0.0]
        nis = 0.0
        log_lik = -999.0
    else:
        z = np.array(req.measurement)
        result = filt.update(z, outlier_threshold=config.outlier_threshold)
        x_post = result.x_post
        P_post = result.P_post
        innov = result.innovation.tolist()
        nis = result.nis
        log_lik = result.log_likelihood
    
    track['step'] += 1
    track['history'].append({
        'step': track['step'],
        'x': x_post[:4].tolist(),
        'measurement': req.measurement
    })
    
    return StateResponse(
        track_id=track_id,
        state_estimate=x_post.tolist(),
        covariance_diagonal=np.diag(P_post).tolist(),
        innovation=innov,
        nis=nis,
        log_likelihood=log_lik,
        step=track['step'],
        filter_type=config.filter_type,
        motion_model=config.motion_model
    )


@app.post("/batch", response_model=BatchResponse, tags=["Tracking"])
def batch_filter(req: BatchRequest):
    """
    Run Kalman filter on a full sequence of measurements (offline mode).
    """
    config = req.config
    model, filt = registry._build(config)
    
    measurements = np.array(req.measurements)
    T = len(measurements)
    missing_mask = np.array(req.missing_mask) if req.missing_mask else np.zeros(T, dtype=bool)
    
    results = filt.run(measurements, missing_mask=missing_mask,
                       outlier_threshold=config.outlier_threshold)
    
    states_prior = [r.x_prior.tolist() for r in results]
    states_post = [r.x_post.tolist() for r in results]
    covs = [np.diag(r.P_post).tolist() for r in results]
    innovations = [r.innovation.tolist() for r in results]
    nis_seq = [r.nis for r in results]
    
    # RMSE on position (no ground truth available in API)
    pos_errors = np.array([np.linalg.norm(r.x_post[:2] - r.x_prior[:2]) for r in results])
    rmse = float(np.sqrt(np.mean(pos_errors**2)))
    
    return BatchResponse(
        states_prior=states_prior,
        states_posterior=states_post,
        covariances=covs,
        innovations=innovations,
        nis_sequence=nis_seq,
        rmse_position=rmse
    )


@app.get("/tracks", tags=["Tracks"])
def list_tracks():
    return {"tracks": registry.list_tracks()}


@app.get("/tracks/{track_id}/history", tags=["Tracks"])
def get_history(track_id: str, last_n: int = 100):
    track = registry.get(track_id)
    history = track['history'][-last_n:]
    return {"track_id": track_id, "history": history, "total_steps": track['step']}


@app.delete("/tracks/{track_id}", tags=["Tracks"])
def delete_track(track_id: str):
    registry.delete(track_id)
    return {"status": "deleted", "track_id": track_id}


@app.websocket("/ws/track/{track_id}")
async def ws_track(websocket: WebSocket, track_id: str):
    """
    WebSocket endpoint for real-time streaming.
    
    Client sends: {"measurement": [x, y], "missing": false}
    Server sends: StateResponse JSON
    """
    await websocket.accept()
    
    if track_id not in registry._tracks:
        await websocket.send_json({"error": f"Track {track_id} not found"})
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            req = MeasurementRequest(
                track_id=track_id,
                measurement=data.get("measurement", [0, 0]),
                missing=data.get("missing", False)
            )
            response = update_track(track_id, req)
            await websocket.send_json(response.dict())
    except WebSocketDisconnect:
        pass


@app.get("/analysis/observability/{track_id}", tags=["Analysis"])
def observability_analysis(track_id: str):
    """Compute observability matrix rank and eigenvalues of state transition."""
    track = registry.get(track_id)
    model = track['model']
    
    result = {}
    
    if hasattr(model, 'is_observable'):
        obs, rank = model.is_observable()
        result['observable'] = obs
        result['observability_rank'] = int(rank)
        result['n_state'] = model.n_state
    
    if hasattr(model, 'eigenvalue_stability'):
        eigs, stable = model.eigenvalue_stability()
        result['eigenvalues'] = [{'real': float(e.real), 'imag': float(e.imag), 
                                   'magnitude': float(abs(e))} for e in eigs]
        result['stable'] = bool(stable)
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)