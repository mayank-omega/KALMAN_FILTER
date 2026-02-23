"""
Advanced Kalman Filter Tracking Dashboard
==========================================
Industrial telemetry aesthetic · Canvas charts · Animated metrics
NEES/NIS panels · Covariance ellipses · Monte Carlo analysis
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (ConstantVelocityModel, ConstantAccelerationModel,
                          CoordinatedTurnModel, simulate_trajectory)
from filters.kalman_filter import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from smoothers.rts import RTSSmoother, UKFRTSSmoother
from analysis.monte_carlo import MonteCarloEvaluator

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KalmanOS · Tracking Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — Industrial Telemetry Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
  --bg-void:    #04060f;
  --bg-panel:   #080d1a;
  --bg-card:    #0d1424;
  --border:     #1a2d4a;
  --border-lit: #1e4d7a;
  --cyan:       #00d4ff;
  --cyan-dim:   #0a4a5e;
  --green:      #00ff88;
  --green-dim:  #0a3d26;
  --amber:      #ffaa00;
  --amber-dim:  #3d2a00;
  --rose:       #ff4466;
  --rose-dim:   #3d0f18;
  --violet:     #a855f7;
  --text-prim:  #c8d8f0;
  --text-sec:   #5a7a9a;
  --text-dim:   #2a4060;
  --font-mono:  'Share Tech Mono', monospace;
  --font-ui:    'Rajdhani', sans-serif;
  --font-head:  'Exo 2', sans-serif;
}

html, body, [class*="css"] {
  background: var(--bg-void) !important;
  color: var(--text-prim) !important;
  font-family: var(--font-ui) !important;
}
.stApp { background: var(--bg-void) !important; }

[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-prim) !important; }
[data-testid="stSidebar"] label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  color: var(--cyan) !important;
  text-transform: uppercase;
  letter-spacing: 1px;
}
[data-testid="stSidebar"] h2 {
  font-family: var(--font-head) !important;
  color: var(--cyan) !important;
  font-size: 12px !important;
  text-transform: uppercase;
  letter-spacing: 2px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
}

[data-baseweb="tag"] {
  background: var(--cyan-dim) !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
}

[data-baseweb="select"] > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-lit) !important;
  color: var(--text-prim) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}

.stButton > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 8px 12px !important;
  transition: all 0.2s !important;
  width: 100% !important;
  border-radius: 2px !important;
}
.stButton > button:hover {
  background: var(--cyan-dim) !important;
  box-shadow: 0 0 12px rgba(0,212,255,0.25) !important;
}
.stButton > button[kind="primary"] {
  border-color: var(--green) !important;
  color: var(--green) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--green-dim) !important;
  box-shadow: 0 0 12px rgba(0,255,136,0.25) !important;
}

[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  padding: 12px 16px !important;
}

h1, h2, h3 {
  font-family: var(--font-head) !important;
}
h1 { color: var(--text-prim) !important; font-weight: 800 !important; font-size: 22px !important; }
h2 { color: var(--cyan) !important; font-weight: 600 !important; font-size: 13px !important;
     text-transform: uppercase; letter-spacing: 2px; }
h3 { color: var(--text-sec) !important; font-size: 11px !important; text-transform: uppercase; }

.stSuccess { background: var(--green-dim) !important; border: 1px solid var(--green) !important;
             color: var(--green) !important; font-family: var(--font-mono) !important;
             font-size: 11px !important; border-radius: 2px !important; }
.stWarning { background: var(--amber-dim) !important; border: 1px solid var(--amber) !important;
             color: var(--amber) !important; font-family: var(--font-mono) !important;
             font-size: 11px !important; border-radius: 2px !important; }
.stError   { background: var(--rose-dim) !important; border: 1px solid var(--rose) !important;
             color: var(--rose) !important; font-family: var(--font-mono) !important;
             font-size: 11px !important; border-radius: 2px !important; }

hr { border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

[data-baseweb="tab-list"] {
  background: var(--bg-panel) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: 9px !important;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-sec) !important;
  padding: 10px 18px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--cyan) !important;
  border-bottom-color: var(--cyan) !important;
  background: var(--bg-card) !important;
}

.block-container { padding-top: 1.2rem !important; padding-bottom: 0.5rem !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 3px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def cov_ellipse(mean, cov2, n_sigma=2, n_pts=120):
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 1e-12)
    t = np.linspace(0, 2*np.pi, n_pts)
    circle = np.array([np.cos(t), np.sin(t)])
    e = vecs @ np.diag(n_sigma * np.sqrt(vals)) @ circle
    return (mean[0]+e[0]).tolist(), (mean[1]+e[1]).tolist()

def build_model(name, dt, sigma_a):
    if "CV" in name: return ConstantVelocityModel(dt=dt, sigma_a=sigma_a)
    elif "CA" in name: return ConstantAccelerationModel(dt=dt, sigma_j=sigma_a)
    return CoordinatedTurnModel(dt=dt, sigma_a=sigma_a)

def build_filter(ft, model, R, x0, P0):
    if ft == "KF" and hasattr(model,'F') and "CT" not in type(model).__name__:
        return KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0, P0=P0)
    elif ft == "EKF":
        return ExtendedKalmanFilter(f=model.f, h=model.h, F_jac=model.F_jacobian,
                                     H_jac=model.H_jacobian, Q=model.Q, R=R, x0=x0, P0=P0)
    return UnscentedKalmanFilter(f=model.f, h=model.h, Q=model.Q, R=R, x0=x0, P0=P0)

def get_x0(model, name):
    n = model.n_state
    if "CT" in name: x0 = np.array([0.,0.,15.,2.,0.08])
    elif "CA" in name: x0 = np.array([0.,0.,15.,2.,0.3,0.1])
    else: x0 = np.array([0.,0.,15.,2.])
    return x0, np.eye(n)*10.0

FC = {"KF":"#00d4ff","EKF":"#00ff88","UKF":"#ffaa00",
      "KF_smooth":"#4af0ff","EKF_smooth":"#66ffbb","UKF_smooth":"#ffd060"}

def _j(d): return json.dumps(d)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 18px 0;">
      <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:18px;
                  color:#00d4ff;letter-spacing:1px;">KALMAN OS</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                  color:#2a4060;letter-spacing:3px;margin-top:2px;">TRACKING ENGINE v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Motion Model")
    motion_model = st.selectbox("", ["Constant Velocity (CV)","Constant Acceleration (CA)",
                                      "Coordinated Turn (CT)"], label_visibility="collapsed")

    st.markdown("## Active Filters")
    filter_types = st.multiselect("", ["KF","EKF","UKF"], default=["KF","EKF","UKF"],
                                   label_visibility="collapsed")
    run_smoother = st.checkbox("RTS Backward Smoother", value=True)

    st.markdown("## Simulation")
    T          = st.slider("Timesteps T",        50,  500, 150, 10)
    dt         = st.slider("Sampling dt (s)",   0.05, 0.5, 0.1, 0.05)
    sigma_a    = st.slider("Process noise σ_a",  0.1, 5.0, 1.0, 0.1)
    sigma_meas = st.slider("Measurement σ_r (m)",0.5,20.0, 5.0, 0.5)

    st.markdown("## Outlier Gating")
    use_gating    = st.checkbox("Mahalanobis Gating", value=True)
    gating_thresh = st.slider("χ² threshold (dof=2)", 5.99, 20.0, 9.21) if use_gating else None

    st.markdown("## Monte Carlo")
    mc_runs = st.slider("Simulation runs", 100, 1000, 300, 50)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1: run_sim = st.button("▶ RUN",  type="secondary")
    with c2: run_mc  = st.button("⚡ MC",  type="primary")

    st.markdown("""
    <div style="margin-top:20px;padding:10px;border:1px solid #1a2d4a;border-radius:3px;
                background:#080d1a;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                  color:#5a7a9a;line-height:1.9;">
        NumPy-only · KF · EKF · UKF · PF<br>
        RTS Smoother · GPS+IMU Fusion<br>
        NEES/NIS · Mahalanobis Gating<br>
        FastAPI · Docker · Research-grade
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER BAR
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:12px 18px;background:#080d1a;border:1px solid #1a2d4a;
            border-radius:4px;margin-bottom:16px;">
  <div>
    <span style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:20px;
                 color:#c8d8f0;">Real-Time Object Tracking</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2a4060;
                 margin-left:12px;letter-spacing:2px;">KALMAN FILTER SUITE</span>
  </div>
  <div style="display:flex;gap:20px;align-items:center;">
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#2a4060;">BACKEND</div>
      <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:12px;color:#00ff88;">● ONLINE</div>
    </div>
    <div style="width:1px;height:28px;background:#1a2d4a;"></div>
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#2a4060;">SMOOTHER</div>
      <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:12px;color:#a855f7;">RTS ✓</div>
    </div>
    <div style="width:1px;height:28px;background:#1a2d4a;"></div>
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#2a4060;">GATING</div>
      <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:12px;
                  color:#00d4ff;">χ²={:.2f}</div>
    </div>
  </div>
</div>
""".format(gating_thresh or 0), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ══════════════════════════════════════════════════════════════════
if run_sim or 'sim_data' not in st.session_state:
    seed = int(np.random.randint(0,9999)) if run_sim else 42
    model  = build_model(motion_model, dt, sigma_a)
    x0, P0 = get_x0(model, motion_model)
    R      = np.eye(2) * sigma_meas**2
    x_true, measurements = simulate_trajectory(model, x0, T, R, seed=seed)

    filter_results = {}
    for ft in filter_types:
        filt    = build_filter(ft, model, R, x0.copy(), P0.copy())
        results = filt.run(measurements, outlier_threshold=gating_thresh)
        filter_results[ft] = (filt, results)

    smoother_results = {}
    if run_smoother:
        for ft, (filt, results) in filter_results.items():
            try:
                if ft == "KF" and hasattr(filt,'F'):
                    xs, Ps = RTSSmoother(F=filt.F).smooth(results)
                else:
                    xs, Ps = UKFRTSSmoother(f=model.f, n=model.n_state).smooth(results, Q=model.Q)
                smoother_results[ft] = (xs, Ps)
            except: pass

    st.session_state.update({
        'model':model, 'x_true':x_true, 'measurements':measurements,
        'filter_results':filter_results, 'smoother_results':smoother_results,
        'R':R, 'x0':x0, 'P0':P0, 'T':T, 'dt':dt, 'motion_model':motion_model
    })

model          = st.session_state['model']
x_true         = st.session_state['x_true']
measurements   = st.session_state['measurements']
filter_results = st.session_state['filter_results']
smoother_results = st.session_state['smoother_results']
T_cur          = st.session_state['T']

# ══════════════════════════════════════════════════════════════════
# METRICS ROW
# ══════════════════════════════════════════════════════════════════
st.markdown("## Performance Metrics")

fstats = {}
for ft, (filt, results) in filter_results.items():
    pe = [np.linalg.norm(r.x_post[:2]-x_true[t+1,:2]) for t,r in enumerate(results)]
    ve = [np.linalg.norm(r.x_post[2:4]-x_true[t+1,2:4]) for t,r in enumerate(results)]
    nis = [r.nis for r in results]
    ll  = [r.log_likelihood for r in results if r.log_likelihood > -999]
    sm_rmse = None
    if ft in smoother_results:
        xs,_ = smoother_results[ft]
        sm_rmse = float(np.sqrt(np.mean([np.linalg.norm(xs[t,:2]-x_true[t+1,:2])**2 for t in range(len(results))])))
    fstats[ft] = dict(rmse_pos=float(np.sqrt(np.mean(np.array(pe)**2))),
                      rmse_vel=float(np.sqrt(np.mean(np.array(ve)**2))),
                      mean_nis=float(np.mean(nis)), mean_ll=float(np.mean(ll)) if ll else 0,
                      sm_rmse=sm_rmse)

cards = []
col_map = {"KF":"#00d4ff","EKF":"#00ff88","UKF":"#ffaa00"}
for ft, s in fstats.items():
    c = col_map.get(ft,"#a855f7")
    nis_c = "#00ff88" if 1<=s['mean_nis']<=4.6 else "#ffaa00"
    sm_html = f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:8px;color:#5a7a9a;margin-top:3px;">SMOOTHED {s["sm_rmse"]:.3f}m</div>' if s['sm_rmse'] else ''
    cards.append(f"""
    <div style="flex:1;background:#0d1424;border:1px solid {c}22;border-top:2px solid {c};
                border-radius:3px;padding:14px 16px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;right:0;width:36px;height:36px;
                  background:linear-gradient(225deg,{c}0a,transparent);
                  border-left:1px solid {c}12;border-bottom:1px solid {c}12;"></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{c};
                  text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">{ft} Filter</div>
      <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:26px;
                  color:{c};line-height:1;">{s['rmse_pos']:.3f}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#5a7a9a;">RMSE POS (m)</div>
      {sm_html}
      <div style="margin-top:10px;display:flex;gap:14px;border-top:1px solid {c}15;padding-top:10px;">
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:#c8d8f0;">{s['rmse_vel']:.3f}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">VEL RMSE</div></div>
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:{nis_c};">{s['mean_nis']:.2f}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">AVG NIS</div></div>
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:#a855f7;">{s['mean_ll']:.1f}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">LOG-LIK</div></div>
      </div>
    </div>""")

# System card
obs_ok, obs_rank = (True, model.n_state)
if hasattr(model,'is_observable'): obs_ok, obs_rank = model.is_observable()
eig_max = 1.0
if hasattr(model,'eigenvalue_stability'):
    eigs, stab = model.eigenvalue_stability()
    eig_max = float(max(abs(e) for e in eigs))
oc = "#00ff88" if obs_ok else "#ff4466"
ec = "#ffaa00" if abs(eig_max-1)<0.001 else ("#00ff88" if eig_max<1 else "#ff4466")
cards.append(f"""
    <div style="flex:1;background:#0d1424;border:1px solid #1a2d4a;border-top:2px solid #a855f7;
                border-radius:3px;padding:14px 16px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#a855f7;
                  text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">System State</div>
      <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:18px;color:{oc};">
        {'OBSERVABLE' if obs_ok else 'NOT OBSERVABLE'}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#5a7a9a;">
        rank {obs_rank}/{model.n_state} · n={model.n_state}</div>
      <div style="margin-top:10px;display:flex;gap:14px;border-top:1px solid #1a2d4a;padding-top:10px;">
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:{ec};">{eig_max:.4f}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">MAX |λ|</div></div>
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:#00d4ff;">{T_cur}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">STEPS</div></div>
        <div><div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:13px;color:#c8d8f0;">{len(filter_results)}</div>
             <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">FILTERS</div></div>
      </div>
    </div>""")

st.markdown(f'<div style="display:flex;gap:10px;margin-bottom:18px;">{"".join(cards)}</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "  TRAJECTORY  ","  INNOVATION & NIS  ",
    "  COVARIANCE ELLIPSES  ","  SYSTEM ANALYSIS  ","  MONTE CARLO  "
])

# ─── TAB 1: TRAJECTORY ───────────────────────────────────────────
with tab1:
    st.markdown("### Estimated vs True Trajectory")

    traj = {
        'tx': x_true[1:,0].tolist(), 'ty': x_true[1:,1].tolist(),
        'mx': measurements[:,0].tolist(), 'my': measurements[:,1].tolist(),
        'filters': {}
    }
    for ft,(filt,res) in filter_results.items():
        traj['filters'][ft] = {'x':[r.x_post[0] for r in res],
                                'y':[r.x_post[1] for r in res], 'color':FC[ft]}
        if ft in smoother_results:
            xs,_ = smoother_results[ft]
            k = ft+'_smooth'
            traj['filters'][k] = {'x':xs[:,0].tolist(),'y':xs[:,1].tolist(),'color':FC.get(k,'#fff')}

    traj_html = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#04060f;font-family:'Courier New',monospace;}}
canvas{{display:block;}}
#leg{{position:absolute;top:12px;left:12px;font-size:10px;}}
.lr{{display:flex;align-items:center;gap:7px;margin-bottom:4px;color:#c8d8f0;}}
.ll{{width:20px;height:2px;}}
#tip{{position:absolute;bottom:10px;right:12px;font-size:9px;color:#2a4060;}}
</style></head><body>
<div style="position:relative;width:100%;height:420px;">
<canvas id="c" style="width:100%;height:420px;"></canvas>
<div id="leg"></div><div id="tip">HOVER FOR XY COORDS</div>
</div>
<script>
const d={_j(traj)};
const cv=document.getElementById('c');
cv.width=cv.offsetWidth||900; cv.height=cv.offsetHeight||420;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height;
const P={{l:48,r:18,t:22,b:34}};
let ax=[...d.tx,...d.mx],ay=[...d.ty,...d.my];
Object.values(d.filters).forEach(f=>{{ax.push(...f.x);ay.push(...f.y);}});
const xn=Math.min(...ax),xx=Math.max(...ax),yn=Math.min(...ay),yx=Math.max(...ay);
const xr=xx-xn||1,yr=yx-yn||1,ep=0.07;
function tc(x,y){{return[P.l+(x-xn+xr*ep)/(xr*(1+2*ep))*(W-P.l-P.r),
                          H-P.b-(y-yn+yr*ep)/(yr*(1+2*ep))*(H-P.t-P.b)];}}
// Grid
ctx.strokeStyle='#0d1828'; ctx.lineWidth=1;
for(let i=0;i<=8;i++){{
  const gx=P.l+i*(W-P.l-P.r)/8; const gy=P.t+i*(H-P.t-P.b)/8;
  ctx.beginPath();ctx.moveTo(gx,P.t);ctx.lineTo(gx,H-P.b);ctx.stroke();
  ctx.beginPath();ctx.moveTo(P.l,gy);ctx.lineTo(W-P.r,gy);ctx.stroke();
}}
// Axis labels
ctx.fillStyle='#2a4060';ctx.font='9px Courier New';ctx.textAlign='center';
for(let i=0;i<=4;i++){{const v=xn+i*xr/4;const[cx]=tc(v,yn);ctx.fillText(v.toFixed(0)+'m',cx,H-4);}}
ctx.textAlign='right';
for(let i=0;i<=4;i++){{const v=yn+i*yr/4;const[,cy]=tc(xn,v);ctx.fillText(v.toFixed(0),P.l-4,cy+3);}}
// Measurements
ctx.fillStyle='#1a2d4a';
d.mx.forEach((x,i)=>{{const[cx,cy]=tc(x,d.my[i]);ctx.beginPath();ctx.arc(cx,cy,2,0,2*Math.PI);ctx.fill();}});
function drawPath(xs,ys,col,lw,dash){{
  ctx.save();ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);
  xs.forEach((x,i)=>{{const[cx,cy]=tc(x,ys[i]);i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}});
  ctx.stroke();ctx.restore();
}}
// Glow
Object.values(d.filters).forEach(f=>{{
  ctx.save();ctx.shadowColor=f.color;ctx.shadowBlur=10;
  drawPath(f.x,f.y,f.color+'44',8);ctx.restore();
}});
// True trajectory
ctx.save();ctx.shadowColor='#ff4466';ctx.shadowBlur=6;
drawPath(d.tx,d.ty,'#ff4466',2.5,[6,4]);ctx.restore();
// Filter paths
Object.entries(d.filters).forEach(([n,f])=>{{
  drawPath(f.x,f.y,f.color,n.includes('smooth')?1.5:2,n.includes('smooth')?[5,3]:[]);
}});
// Start/end markers
function mark(x,y,col,lbl){{const[cx,cy]=tc(x,y);
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=8;
  ctx.fillStyle=col;ctx.beginPath();ctx.arc(cx,cy,5,0,2*Math.PI);ctx.fill();ctx.restore();
  ctx.fillStyle=col;ctx.font='9px Courier New';ctx.textAlign='left';ctx.fillText(lbl,cx+7,cy+3);
}}
mark(d.tx[0],d.ty[0],'#5a7a9a','START');
mark(d.tx[d.tx.length-1],d.ty[d.ty.length-1],'#ff4466','END');
// Legend
const leg=document.getElementById('leg');
const entries=[['True','#ff4466','dashed'],['Noise','#1a2d4a','dot'],
  ...Object.entries(d.filters).map(([n,f])=>[n,f.color,n.includes('smooth')?'dashed':'solid'])];
leg.innerHTML=entries.map(([l,c,s])=>s==='dot'
  ?`<div class="lr"><div style="width:8px;height:8px;border-radius:50%;background:${{c}}"></div><span>${{l}}</span></div>`
  :`<div class="lr"><div class="ll" style="background:${{s==='dashed'?`repeating-linear-gradient(90deg,${{c}} 0px,${{c}} 4px,transparent 4px,transparent 7px)`:c}}"></div><span>${{l}}</span></div>`
).join('');
// Hover
const tip=document.getElementById('tip');
cv.addEventListener('mousemove',e=>{{
  const r=cv.getBoundingClientRect();
  const mx=(e.clientX-r.left)*(W/r.width);
  const my=(e.clientY-r.top)*(H/r.height);
  const wx=xn+(mx-P.l)/(W-P.l-P.r)*xr;
  const wy=yn+(H-P.b-my)/(H-P.t-P.b)*yr;
  tip.textContent=`x: ${{wx.toFixed(1)}}m  y: ${{wy.toFixed(1)}}m`;
}});
</script></body></html>"""
    components.html(traj_html, height=440)

    st.markdown("### Position Error Evolution")
    err = {}
    for ft,(filt,res) in filter_results.items():
        err[ft] = {'e':[float(np.linalg.norm(r.x_post[:2]-x_true[t+1,:2])) for t,r in enumerate(res)],
                   'color':FC[ft]}
    for ft,(xs,Ps) in smoother_results.items():
        k = ft+'_smooth'
        err[k] = {'e':[float(np.linalg.norm(xs[t,:2]-x_true[t+1,:2])) for t in range(len(xs))],
                  'color':FC.get(k,'#fff')}

    err_html = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#04060f;}}canvas{{display:block;}}</style></head><body>
<canvas id="c" style="width:100%;height:180px;"></canvas>
<script>
const d={_j(err)};
const c=document.getElementById('c'); c.width=c.offsetWidth||900; c.height=c.offsetHeight||180;
const ctx=c.getContext('2d'); const W=c.width,H=c.height,P={{l:44,r:16,t:14,b:28}};
const allE=Object.values(d).flatMap(f=>f.e); const em=Math.max(...allE)*1.05||1;
const T=Object.values(d)[0].e.length;
ctx.strokeStyle='#0d1828'; ctx.lineWidth=1;
[0.25,0.5,0.75,1.0].forEach(f=>{{
  const y=P.t+(1-f)*(H-P.t-P.b);
  ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();
  ctx.fillStyle='#2a4060';ctx.font='8px Courier New';ctx.textAlign='right';
  ctx.fillText((f*em).toFixed(1),P.l-4,y+3);
}});
ctx.fillStyle='#2a4060';ctx.font='8px Courier New';ctx.textAlign='center';
[0,.25,.5,.75,1].forEach(f=>{{const x=P.l+f*(W-P.l-P.r);ctx.fillText(Math.round(f*T),x,H-4);}});
function line(vals,col,lw,dash){{
  ctx.save();ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);
  vals.forEach((v,i)=>{{const x=P.l+(i/(T-1))*(W-P.l-P.r),y=P.t+(1-v/em)*(H-P.t-P.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
  ctx.stroke();ctx.restore();
}}
Object.entries(d).forEach(([ft,f])=>{{
  if(ft.includes('smooth'))return;
  ctx.save();ctx.shadowColor=f.color;ctx.shadowBlur=8;
  line(f.e,f.color+'55',8);ctx.restore();
}});
Object.entries(d).forEach(([ft,f])=>line(f.e,f.color,ft.includes('smooth')?1.5:2,ft.includes('smooth')?[5,3]:[]));
ctx.fillStyle='#5a7a9a';ctx.font='9px Courier New';ctx.textAlign='center';
ctx.fillText('Timestep',W/2,H-1);
</script></body></html>"""
    components.html(err_html, height=195)

# ─── TAB 2: INNOVATION & NIS ──────────────────────────────────────
with tab2:
    lc, rc = st.columns(2)
    with lc:
        st.markdown("### Innovation Sequence")
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
        color:#5a7a9a;padding:6px 10px;background:#080d1a;border-left:2px solid #1a2d4a;
        margin-bottom:10px;line-height:1.7;">ỹₖ = zₖ − H·x̂ₖ  should be white N(0,S)<br>
        Blue band = ±1.96σ acceptance region</div>""", unsafe_allow_html=True)

        for ft,(filt,res) in filter_results.items():
            ix = [r.innovation[0] for r in res]
            iy = [r.innovation[1] for r in res]
            c = FC[ft]
            std = float(np.std(ix)) or 1
            ih = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#080d1a;}}canvas{{display:block;}}</style></head><body>
<canvas id="ic_{ft}" style="width:100%;height:115px;"></canvas>
<script>
const ix={_j(ix)},iy={_j(iy)},std={std};
const cv=document.getElementById('ic_{ft}'); cv.width=cv.offsetWidth||480; cv.height=cv.offsetHeight||115;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height,P={{l:6,r:6,t:16,b:6}};
const mx=Math.max(...ix.map(Math.abs),...iy.map(Math.abs))||1;
// ±2σ band
const bh=std/mx*(H/2-P.t);
ctx.fillStyle='#00d4ff06'; ctx.fillRect(P.l,H/2-bh,W-P.l-P.r,bh*2);
ctx.strokeStyle='#00d4ff33'; ctx.lineWidth=0.8; ctx.setLineDash([3,3]);
ctx.beginPath();ctx.moveTo(P.l,H/2-bh);ctx.lineTo(W-P.r,H/2-bh);ctx.stroke();
ctx.beginPath();ctx.moveTo(P.l,H/2+bh);ctx.lineTo(W-P.r,H/2+bh);ctx.stroke();
ctx.setLineDash([]);
ctx.strokeStyle='#1a2d4a'; ctx.lineWidth=0.8;
ctx.beginPath();ctx.moveTo(P.l,H/2);ctx.lineTo(W-P.r,H/2);ctx.stroke();
function ln(vs,col){{ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1.3;
  vs.forEach((v,i)=>{{const x=P.l+(i/(vs.length-1))*(W-P.l-P.r),y=H/2-v/mx*(H/2-P.t);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});ctx.stroke();}}
ctx.save();ctx.shadowColor='{c}';ctx.shadowBlur=4;ln(ix,'{c}');ctx.restore();
ln(iy,'{c}88');
ctx.fillStyle='{c}';ctx.font='9px Courier New';ctx.textAlign='left';ctx.fillText('{ft} innov',P.l+3,12);
ctx.fillStyle='{c}88';ctx.textAlign='right';ctx.fillText('y',W-P.r-3,12);
</script></body></html>"""
            components.html(ih, height=125)

    with rc:
        st.markdown("### NIS Consistency (χ²)")
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
        color:#5a7a9a;padding:6px 10px;background:#080d1a;border-left:2px solid #1a2d4a;
        margin-bottom:10px;line-height:1.7;">νₖ = ỹₖᵀSₖ⁻¹ỹₖ ~ χ²(m)  filter consistent if mean≈2<br>
        Dashed red = χ²₀.₉₅ = 5.99  · Green band = consistent zone</div>""",
        unsafe_allow_html=True)

        for ft,(filt,res) in filter_results.items():
            nis  = [r.nis for r in res]
            mn   = float(np.mean(nis))
            ok   = 0.5 <= mn <= 5.0
            c    = FC[ft]
            bc   = "#00ff88" if ok else "#ff4466"
            bt   = "CONSISTENT ✓" if ok else "INCONSISTENT ✗"
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
            align-items:center;padding:3px 0;margin-bottom:1px;">
            <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{c};">{ft}</span>
            <div style="display:flex;gap:8px;align-items:center;">
              <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#5a7a9a;">
                μ={mn:.3f}</span>
              <span style="font-family:'Share Tech Mono',monospace;font-size:8px;color:{bc};
                           border:1px solid {bc};padding:1px 6px;border-radius:2px;">{bt}</span>
            </div></div>""", unsafe_allow_html=True)

            nh = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#080d1a;}}canvas{{display:block;}}</style></head><body>
<canvas id="nc_{ft}" style="width:100%;height:98px;"></canvas>
<script>
const ns={_j(nis)};
const cv=document.getElementById('nc_{ft}'); cv.width=cv.offsetWidth||480; cv.height=cv.offsetHeight||98;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height,P={{l:6,r:6,t:6,b:6}};
const vm=Math.max(...ns,7)*1.1;
// Acceptance band
const y_lo=H-P.b-0.5/vm*(H-P.t-P.b), y_hi=H-P.b-5.99/vm*(H-P.t-P.b);
ctx.fillStyle='rgba(0,255,136,0.04)'; ctx.fillRect(P.l,y_hi,W-P.l-P.r,y_lo-y_hi);
ctx.strokeStyle='#ff446655'; ctx.lineWidth=0.8; ctx.setLineDash([4,2]);
ctx.beginPath();ctx.moveTo(P.l,y_hi);ctx.lineTo(W-P.r,y_hi);ctx.stroke();
ctx.strokeStyle='#00d4ff33'; ctx.lineWidth=0.6;
ctx.beginPath();ctx.moveTo(P.l,H-P.b-2/vm*(H-P.t-P.b));ctx.lineTo(W-P.r,H-P.b-2/vm*(H-P.t-P.b));ctx.stroke();
ctx.setLineDash([]);
// Area
ctx.beginPath(); ns.forEach((v,i)=>{{const x=P.l+(i/(ns.length-1))*(W-P.l-P.r),y=H-P.b-v/vm*(H-P.t-P.b);
  i===0?ctx.moveTo(x,H-P.b):ctx.lineTo(x,y);}});
ctx.lineTo(W-P.r,H-P.b);ctx.closePath();
const g=ctx.createLinearGradient(0,P.t,0,H-P.b);g.addColorStop(0,'{c}55');g.addColorStop(1,'{c}00');
ctx.fillStyle=g;ctx.fill();
// Line
ctx.beginPath();ctx.strokeStyle='{c}';ctx.lineWidth=1.5;
ns.forEach((v,i)=>{{const x=P.l+(i/(ns.length-1))*(W-P.l-P.r),y=H-P.b-v/vm*(H-P.t-P.b);
  i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});ctx.stroke();
</script></body></html>"""
            components.html(nh, height=108)

# ─── TAB 3: COVARIANCE ELLIPSES ──────────────────────────────────
with tab3:
    st.markdown("### 2σ Uncertainty Ellipses")
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
    color:#5a7a9a;padding:6px 10px;background:#080d1a;border-left:2px solid #1a2d4a;
    margin-bottom:14px;line-height:1.7;">P_{k|k} encodes 2D position uncertainty.
    True state should lie inside 2σ ellipse ≈95.4% of time.<br>
    1σ / 2σ / 3σ nested rings shown · Blue dot = estimate · Red dot = true position</div>""",
    unsafe_allow_html=True)

    ell_step = st.slider("Timestep", 0, T_cur-1, T_cur//2, key="ell")
    ell_cols = st.columns(len(filter_results))

    for i,(ft,(filt,res)) in enumerate(filter_results.items()):
        r = res[ell_step]
        mp = r.x_post[:2]; tp = x_true[ell_step+1,:2]
        cov2 = r.P_post[:2,:2]; c = FC[ft]
        e1x,e1y = cov_ellipse(mp,cov2,1); e2x,e2y = cov_ellipse(mp,cov2,2); e3x,e3y = cov_ellipse(mp,cov2,3)
        err_d = float(np.linalg.norm(mp-tp))
        sig1  = float(np.sqrt(np.trace(cov2)/2))
        inside = err_d <= 2*sig1

        ed = {'e1x':e1x,'e1y':e1y,'e2x':e2x,'e2y':e2y,'e3x':e3x,'e3y':e3y,
              'mx':float(mp[0]),'my':float(mp[1]),'tx':float(tp[0]),'ty':float(tp[1]),'c':c}

        eh = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#080d1a;}}canvas{{display:block;}}</style></head><body>
<canvas id="e_{ft}" style="width:100%;height:240px;"></canvas>
<script>
const d={_j(ed)};
const cv=document.getElementById('e_{ft}'); cv.width=cv.offsetWidth||300; cv.height=cv.offsetHeight||240;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height;
const ax=[...d.e3x,d.mx,d.tx],ay=[...d.e3y,d.my,d.ty];
const xn=Math.min(...ax),xx=Math.max(...ax),yn=Math.min(...ay),yx=Math.max(...ay);
const xr=xx-xn||1,yr=yx-yn||1,ep=0.18;
function tc(x,y){{return[10+(x-xn+xr*ep)/(xr*(1+2*ep))*(W-20),H-10-(y-yn+yr*ep)/(yr*(1+2*ep))*(H-20)];}}
function ell(ex,ey,stroke,fill,lw,dash){{
  ctx.beginPath();ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);
  if(fill)ctx.fillStyle=fill;
  ex.forEach((x,i)=>{{const[cx,cy]=tc(x,ey[i]);i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}});
  ctx.closePath();if(fill)ctx.fill();ctx.stroke();ctx.setLineDash([]);
}}
ell(d.e3x,d.e3y,d.c+'25',d.c+'04',0.8,[4,3]);
ell(d.e2x,d.e2y,d.c+'77',d.c+'0e',1.5);
ell(d.e1x,d.e1y,d.c+'cc',d.c+'20',2);
const[emx,emy]=tc(d.mx,d.my);
ctx.strokeStyle=d.c+'88';ctx.lineWidth=0.8;ctx.setLineDash([2,2]);
ctx.beginPath();ctx.moveTo(emx-10,emy);ctx.lineTo(emx+10,emy);ctx.stroke();
ctx.beginPath();ctx.moveTo(emx,emy-10);ctx.lineTo(emx,emy+10);ctx.stroke();
ctx.setLineDash([]);
ctx.save();ctx.shadowColor=d.c;ctx.shadowBlur=12;
ctx.fillStyle=d.c;ctx.beginPath();ctx.arc(emx,emy,4,0,2*Math.PI);ctx.fill();ctx.restore();
const[etx,ety]=tc(d.tx,d.ty);
ctx.save();ctx.shadowColor='#ff4466';ctx.shadowBlur=10;
ctx.fillStyle='#ff4466';ctx.beginPath();ctx.arc(etx,ety,5,0,2*Math.PI);ctx.fill();ctx.restore();
ctx.strokeStyle='#ffffff15';ctx.lineWidth=1;ctx.setLineDash([2,2]);
ctx.beginPath();ctx.moveTo(emx,emy);ctx.lineTo(etx,ety);ctx.stroke();ctx.setLineDash([]);
ctx.fillStyle=d.c;ctx.font='10px Courier New';ctx.textAlign='left';ctx.fillText('{ft}',7,15);
ctx.fillStyle='#5a7a9a';ctx.font='8px Courier New';ctx.fillText('1σ 2σ 3σ',7,27);
ctx.fillStyle='#ff4466';ctx.fillText('● TRUE',7,H-22);
ctx.fillStyle=d.c;ctx.fillText('● EST',7,H-10);
</script></body></html>"""

        with ell_cols[i]:
            components.html(eh, height=250)
            trc  = float(np.trace(r.P_post))
            cond = float(np.linalg.cond(r.P_post))
            ic   = "#00ff88" if inside else "#ff4466"
            st.markdown(f"""<div style="background:#080d1a;border:1px solid {c}25;
            border-radius:3px;padding:10px 12px;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
              <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">
                ERR DIST</div>
                <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:15px;
                color:{ic};">{err_d:.2f}m</div></div>
              <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">
                TR(P)</div>
                <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:15px;
                color:{c};">{trc:.3f}</div></div>
              <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">
                COND(P)</div>
                <div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:12px;
                color:#c8d8f0;">{cond:.1e}</div></div>
              <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">
                IN 2σ</div>
                <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:14px;
                color:{ic};">{'YES ✓' if inside else 'NO ✗'}</div></div>
            </div></div>""", unsafe_allow_html=True)

# ─── TAB 4: SYSTEM ANALYSIS ──────────────────────────────────────
with tab4:
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("### Eigenvalue Stability (Unit Circle)")
        if hasattr(model,'eigenvalue_stability'):
            eigs,stable = model.eigenvalue_stability()
            ed2 = [{'re':float(e.real),'im':float(e.imag),'mag':float(abs(e))} for e in eigs]
            ev_html = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#080d1a;}}canvas{{display:block;}}</style></head><body>
<canvas id="ev" style="width:100%;height:250px;"></canvas>
<script>
const eigs={_j(ed2)};
const cv=document.getElementById('ev'); cv.width=cv.offsetWidth||400; cv.height=cv.offsetHeight||250;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height,cx=W/2,cy=H/2;
const R=Math.min(W,H)*0.36;
ctx.strokeStyle='#0d1828'; ctx.lineWidth=1;
ctx.beginPath();ctx.arc(cx,cy,R/2,0,2*Math.PI);ctx.stroke();
ctx.beginPath();ctx.moveTo(cx-R*1.15,cy);ctx.lineTo(cx+R*1.15,cy);ctx.stroke();
ctx.beginPath();ctx.moveTo(cx,cy-R*1.15);ctx.lineTo(cx,cy+R*1.15);ctx.stroke();
ctx.save();ctx.shadowColor='#00d4ff';ctx.shadowBlur=8;
ctx.strokeStyle='#00d4ff44';ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(cx,cy,R,0,2*Math.PI);ctx.stroke();
ctx.restore();
ctx.fillStyle='#1a2d4a';ctx.font='8px Courier New';ctx.textAlign='center';
ctx.fillText('+1',cx+R+6,cy+3);ctx.fillText('-1',cx-R-6,cy+3);
ctx.fillText('+j',cx+4,cy-R-4);ctx.fillText('-j',cx+4,cy+R+12);
eigs.forEach((e,i)=>{{
  const px=cx+e.re*R,py=cy-e.im*R;
  const onU=Math.abs(e.mag-1)<0.005,col=onU?'#ffaa00':e.mag<1?'#00ff88':'#ff4466';
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=14;
  ctx.fillStyle=col;ctx.beginPath();ctx.arc(px,py,5,0,2*Math.PI);ctx.fill();ctx.restore();
  ctx.fillStyle=col;ctx.font='9px Courier New';ctx.textAlign='left';
  ctx.fillText('λ'+(i+1)+' |'+e.mag.toFixed(4)+'|',px+8,py-3);
}});
ctx.fillStyle='#5a7a9a';ctx.font='9px Courier New';ctx.textAlign='left';
ctx.fillText('EIGENVALUE STABILITY PLOT',8,14);
</script></body></html>"""
            components.html(ev_html, height=258)
            for j,e in enumerate(eigs):
                mag=abs(e); col="#ffaa00" if abs(mag-1)<0.001 else ("#00ff88" if mag<1 else "#ff4466")
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 10px;
                background:#0d1424;border:1px solid {col}22;border-radius:2px;margin-bottom:3px;">
                <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#5a7a9a;">λ_{j+1}</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#c8d8f0;">
                {e.real:.5f} {'+' if e.imag>=0 else ''}{e.imag:.4f}j</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{col};">|λ|={mag:.5f}</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:8px;color:{col};
                border:1px solid {col};padding:1px 5px;border-radius:2px;">
                {'STABLE' if mag<=1 else 'UNSTABLE'}</span></div>""", unsafe_allow_html=True)
            st.success("✓ All eigenvalues on/inside unit circle — Lyapunov stable") if stable else st.error("✗ UNSTABLE")

    with ac2:
        st.markdown("### Observability & Singular Values")
        if hasattr(model,'observability_matrix'):
            O = model.observability_matrix()
            rank = int(np.linalg.matrix_rank(O))
            svds = np.linalg.svd(O,compute_uv=False).tolist()
            obs_ok = rank==model.n_state

            oc2 = "#00ff88" if obs_ok else "#ff4466"
            st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;">
            <div style="background:#0d1424;border:1px solid {oc2}44;border-radius:3px;padding:10px;text-align:center;">
              <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">RANK</div>
              <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:20px;color:{oc2};">{rank}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#5a7a9a;">/{model.n_state}</div>
            </div>
            <div style="background:#0d1424;border:1px solid #1a2d4a;border-radius:3px;padding:10px;text-align:center;">
              <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">σ_MAX</div>
              <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:17px;color:#00d4ff;">{svds[0]:.2f}</div>
            </div>
            <div style="background:#0d1424;border:1px solid #1a2d4a;border-radius:3px;padding:10px;text-align:center;">
              <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">σ_MIN</div>
              <div style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:17px;
              color:{'#00ff88' if svds[-1]>0.01 else '#ff4466'};">{svds[-1]:.3f}</div>
            </div></div>""", unsafe_allow_html=True)

            svd_d = svds[:model.n_state]
            sm2   = max(svd_d) or 1
            sb_html = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#080d1a;}}canvas{{display:block;}}</style></head><body>
<canvas id="sv" style="width:100%;height:120px;"></canvas>
<script>
const sv={_j(svd_d)};
const cv=document.getElementById('sv'); cv.width=cv.offsetWidth||400; cv.height=cv.offsetHeight||120;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height;
const n=sv.length,bw=(W-20)/n-6,mx={sm2};
sv.forEach((v,i)=>{{
  const x=10+(i*(bw+6)), bh=v/mx*(H-28), p=v/mx;
  const col=p>0.5?'#00ff88':p>0.1?'#ffaa00':'#ff4466';
  ctx.fillStyle=col+'33'; ctx.fillRect(x,H-14-bh,bw,bh);
  ctx.fillStyle=col; ctx.fillRect(x,H-14-bh,bw,2);
  ctx.font='8px Courier New';ctx.fillStyle='#5a7a9a';ctx.textAlign='center';
  ctx.fillText('σ'+(i+1),x+bw/2,H-1);
  ctx.fillStyle=col; ctx.fillText(v.toFixed(2),x+bw/2,H-16-bh);
}});
ctx.fillStyle='#5a7a9a';ctx.font='9px Courier New';ctx.textAlign='left';
ctx.fillText('OBSERVABILITY MATRIX SINGULAR VALUES',8,12);
</script></body></html>"""
            components.html(sb_html, height=132)
            st.success("✓ Fully observable") if obs_ok else st.error("✗ NOT observable")

        st.markdown("### Final Covariance Conditioning")
        for ft,(filt,res) in filter_results.items():
            P = res[-1].P_post; c = FC[ft]
            cond = float(np.linalg.cond(P)); trc = float(np.trace(P))
            lev = ("🟢 GOOD" if cond<1e6 else "🟡 WARN" if cond<1e10 else "🔴 ILL")
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:7px 10px;background:#0d1424;border:1px solid {c}22;border-radius:2px;margin-bottom:4px;">
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{c};">{ft}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#5a7a9a;">tr={trc:.4f}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#c8d8f0;">κ={cond:.2e}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;">{lev}</span>
            </div>""", unsafe_allow_html=True)

# ─── TAB 5: MONTE CARLO ──────────────────────────────────────────
with tab5:
    if run_mc:
        st.markdown("### Monte Carlo Statistical Evaluation")
        mc_results = {}
        prog = st.progress(0, "Initializing…")

        for fi,ft in enumerate(filter_types):
            prog.progress(fi/len(filter_types), f"Running {ft} ({mc_runs} simulations)…")
            _m = build_model(st.session_state['motion_model'], dt, sigma_a)
            _x0, _P0 = get_x0(_m, st.session_state['motion_model'])
            _R = np.eye(2) * sigma_meas**2

            def sf(seed,m=_m,R=_R,x0=_x0.copy()):
                rng=np.random.default_rng(seed); x0c=x0.copy(); x0c[:2]+=rng.normal(0,2,2)
                return simulate_trajectory(m, x0c, T, R, seed=seed)

            def ff(seed,_ft=ft,m=_m,R=_R,x0=_x0.copy(),P0=_P0.copy()):
                return build_filter(_ft, m, R, x0.copy(), P0.copy())

            ev = MonteCarloEvaluator(n_runs=mc_runs, n_state=_m.n_state, n_meas=2)
            mc_results[ft] = ev.run(ff, sf, T=T, verbose=False)

        prog.progress(1.0, "Complete ✓")

        mc_cols = st.columns(len(mc_results))
        for i,(ft,s) in enumerate(mc_results.items()):
            c = FC[ft]; nlo,nhi = s.nees_bounds
            with mc_cols[i]:
                nc = "#00ff88" if s.nees_consistent else "#ff4466"
                ic = "#00ff88" if s.nis_consistent else "#ffaa00"
                st.markdown(f"""<div style="background:#0d1424;border:1px solid {c}25;
                border-top:3px solid {c};border-radius:3px;padding:16px;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{c};
                text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">{ft} · {mc_runs} runs</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                  <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">RMSE POS</div>
                    <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:22px;color:{c};">
                    {s.rmse_pos:.4f}</div></div>
                  <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">RMSE VEL</div>
                    <div style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:22px;color:#c8d8f0;">
                    {s.rmse_vel:.4f}</div></div>
                </div>
                <div style="background:#080d1a;border-radius:2px;padding:8px;margin-bottom:8px;">
                  <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;margin-bottom:4px;">
                  NEES CONSISTENCY</div>
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:15px;color:{nc};">
                    {s.mean_nees:.3f}</span>
                    <span style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#5a7a9a;">
                    ±{s.std_nees:.3f}</span>
                    <span style="flex:1;text-align:right;font-family:'Share Tech Mono',monospace;font-size:7px;
                    color:{nc};border:1px solid {nc};padding:1px 4px;border-radius:2px;">
                    {'PASS' if s.nees_consistent else 'FAIL'}</span>
                  </div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;margin-top:3px;">
                  95% CI: [{nlo:.3f}, {nhi:.3f}]</div>
                </div>
                <div style="background:#080d1a;border-radius:2px;padding:8px;margin-bottom:8px;">
                  <div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;margin-bottom:4px;">
                  NIS CONSISTENCY</div>
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:15px;color:{ic};">
                    {s.mean_nis:.3f}</span>
                    <span style="flex:1;text-align:right;font-family:'Share Tech Mono',monospace;font-size:7px;
                    color:{ic};border:1px solid {ic};padding:1px 4px;border-radius:2px;">
                    {'PASS' if s.nis_consistent else 'MARGINAL'}</span>
                  </div>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">COMPUTE</div>
                    <div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:12px;color:#a855f7;">
                    {s.computation_time*1000:.3f}ms</div></div>
                  <div><div style="font-family:'Share Tech Mono',monospace;font-size:7px;color:#2a4060;">LOG-LIK</div>
                    <div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:12px;color:#c8d8f0;">
                    {s.log_likelihood_total:.1f}</div></div>
                </div></div>""", unsafe_allow_html=True)

        # NEES trajectory
        st.markdown("### NEES Over Time")
        np_data = {ft: s.nees_trajectory.tolist() for ft,s in mc_results.items()}
        ns = _m.n_state

        nees_chart = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#04060f;}}canvas{{display:block;}}</style></head><body>
<canvas id="nc" style="width:100%;height:220px;"></canvas>
<script>
const d={_j(np_data)},n={ns},N={mc_runs};
const cv=document.getElementById('nc'); cv.width=cv.offsetWidth||900; cv.height=cv.offsetHeight||220;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height,P={{l:50,r:18,t:28,b:30}};
const allV=Object.values(d).flat(),vm=Math.max(...allV,n*2.5)*1.05;
const T=Object.values(d)[0].length;
function py(v){{return H-P.b-(v/vm)*(H-P.t-P.b);}}
function px(i){{return P.l+(i/(T-1))*(W-P.l-P.r);}}
ctx.strokeStyle='#0d1828';ctx.lineWidth=1;
[.25,.5,.75,1].forEach(f=>{{const y=P.t+(1-f)*(H-P.t-P.b);
  ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();
  ctx.fillStyle='#2a4060';ctx.font='8px Courier New';ctx.textAlign='right';
  ctx.fillText((f*vm).toFixed(1),P.l-4,y+3);}});
// Ideal band
const yi=py(n);
ctx.fillStyle='rgba(0,212,255,0.03)';ctx.fillRect(P.l,yi-4,W-P.l-P.r,8);
ctx.save();ctx.shadowColor='#00d4ff';ctx.shadowBlur=4;
ctx.strokeStyle='#00d4ff55';ctx.lineWidth=1.2;ctx.setLineDash([5,3]);
ctx.beginPath();ctx.moveTo(P.l,yi);ctx.lineTo(W-P.r,yi);ctx.stroke();
ctx.setLineDash([]);ctx.restore();
ctx.fillStyle='#00d4ff';ctx.font='9px Courier New';ctx.textAlign='left';
ctx.fillText('IDEAL n='+n,P.l+6,yi-6);
const cols={{'KF':'#00d4ff','EKF':'#00ff88','UKF':'#ffaa00'}};
Object.entries(d).forEach(([ft,vs])=>{{
  const col=cols[ft]||'#a855f7';
  ctx.beginPath();vs.forEach((v,i)=>{{i===0?ctx.moveTo(px(i),H-P.b):ctx.lineTo(px(i),py(v));}});
  ctx.lineTo(W-P.r,H-P.b);ctx.closePath();ctx.fillStyle=col+'0c';ctx.fill();
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=8;
  ctx.beginPath();ctx.strokeStyle=col+'66';ctx.lineWidth=6;
  vs.forEach((v,i)=>{{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));}});
  ctx.stroke();ctx.restore();
  ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=2;
  vs.forEach((v,i)=>{{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));}});
  ctx.stroke();
  const lv=vs[vs.length-1];
  ctx.fillStyle=col;ctx.font='9px Courier New';ctx.textAlign='left';
  ctx.fillText(ft+' '+lv.toFixed(1),px(T-1)-26,py(lv)-5);
}});
ctx.fillStyle='#5a7a9a';ctx.font='8px Courier New';ctx.textAlign='center';
ctx.fillText('Timestep',W/2,H);
ctx.fillStyle='#c8d8f0';ctx.font='10px Courier New';ctx.textAlign='left';
ctx.fillText('NEES EVOLUTION · N='+N+' MC RUNS',P.l,16);
</script></body></html>"""
        components.html(nees_chart, height=235)

        # RMSE trajectory
        st.markdown("### RMSE Position Over Time")
        rp = {ft: s.rmse_trajectory.tolist() for ft,s in mc_results.items()}
        rmse_chart = f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;background:#04060f;}}canvas{{display:block;}}</style></head><body>
<canvas id="rc" style="width:100%;height:180px;"></canvas>
<script>
const d={_j(rp)};
const cv=document.getElementById('rc'); cv.width=cv.offsetWidth||900; cv.height=cv.offsetHeight||180;
const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height,P={{l:46,r:16,t:18,b:28}};
const allV=Object.values(d).flat(),vm=Math.max(...allV)*1.05||1;
const T=Object.values(d)[0].length;
function py(v){{return H-P.b-(v/vm)*(H-P.t-P.b);}}
function px(i){{return P.l+(i/(T-1))*(W-P.l-P.r);}}
ctx.strokeStyle='#0d1828';ctx.lineWidth=1;
[.25,.5,.75,1].forEach(f=>{{const y=P.t+(1-f)*(H-P.t-P.b);
  ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();
  ctx.fillStyle='#2a4060';ctx.font='8px Courier New';ctx.textAlign='right';
  ctx.fillText((f*vm).toFixed(2)+'m',P.l-4,y+3);}});
const cols={{'KF':'#00d4ff','EKF':'#00ff88','UKF':'#ffaa00'}};
Object.entries(d).forEach(([ft,vs])=>{{
  const col=cols[ft]||'#a855f7';
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=8;
  ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=2;
  vs.forEach((v,i)=>{{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));}});
  ctx.stroke();ctx.restore();
  const lv=vs[vs.length-1];
  ctx.fillStyle=col;ctx.font='9px Courier New';ctx.textAlign='left';
  ctx.fillText(ft,px(T-1)-22,py(lv)-4);
}});
ctx.fillStyle='#5a7a9a';ctx.font='8px Courier New';ctx.textAlign='center';
ctx.fillText('Timestep',W/2,H);
ctx.fillStyle='#c8d8f0';ctx.font='10px Courier New';ctx.textAlign='left';
ctx.fillText('RMSE POSITION OVER TIME',P.l,13);
</script></body></html>"""
        components.html(rmse_chart, height=195)

    else:
        st.markdown("""<div style="display:flex;flex-direction:column;align-items:center;
        justify-content:center;height:300px;background:#080d1a;border:1px dashed #1a2d4a;
        border-radius:4px;gap:12px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:28px;color:#1a2d4a;">⚡</div>
        <div style="font-family:'Exo 2',sans-serif;font-weight:600;font-size:16px;color:#2a4060;">
          Monte Carlo not yet run</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#1a2d4a;
        text-align:center;max-width:280px;line-height:1.7;">
          Click <span style="color:#00ff88;">⚡ MC</span> in sidebar<br>
          NEES/NIS consistency tests · RMSE analysis · Filter comparison
        </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Filter Comparison Matrix")

import pandas as pd
rows = []
for ft,(filt,res) in filter_results.items():
    pe   = [np.linalg.norm(r.x_post[:2]-x_true[t+1,:2]) for t,r in enumerate(res)]
    nis  = [r.nis for r in res]
    ll   = [r.log_likelihood for r in res if r.log_likelihood>-999]
    sm   = fstats[ft]['sm_rmse']
    rows.append({
        'Filter':          ft,
        'RMSE_pos (m)':    round(fstats[ft]['rmse_pos'],4),
        'RMSE_vel (m/s)':  round(fstats[ft]['rmse_vel'],4),
        'Smooth_RMSE':     round(sm,4) if sm else "—",
        'Mean_NIS':        round(float(np.mean(nis)),3),
        'Max_NIS':         round(float(np.max(nis)),1),
        'Avg_LogLik':      round(float(np.mean(ll)),2) if ll else "—",
        'Jacobian?':       "Yes" if ft=="EKF" else "No",
        'Accuracy':        "1st" if ft=="EKF" else ("Exact" if ft=="KF" else "3rd"),
        'Complexity':      "O(n²m)" if ft in ["KF","EKF"] else "O((2n+1)n²)",
    })

df = pd.DataFrame(rows).set_index('Filter')
st.dataframe(df, use_container_width=True)

st.markdown("""
<div style="margin-top:20px;padding:12px 18px;background:#080d1a;
            border:1px solid #1a2d4a;border-radius:3px;
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2a4060;">
    KalmanOS · NumPy-only · KF · EKF · UKF · RTS Smoother · FastAPI · Docker
  </span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2a4060;">
    NEES/NIS validated · Mahalanobis gating · GPS+IMU fusion
  </span>
</div>
""", unsafe_allow_html=True)