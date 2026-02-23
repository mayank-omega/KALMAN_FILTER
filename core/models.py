"""
State-Space Models and Motion Dynamics
=======================================

Mathematical Foundation
-----------------------
Continuous-time stochastic differential equation (SDE):
    dx(t) = F_c * x(t) dt + G_c * dw(t)

where:
    x(t) ∈ R^n  - state vector
    F_c         - continuous-time state transition matrix
    G_c         - continuous process noise coupling matrix
    w(t)        - Wiener process (Brownian motion), dw ~ N(0, Q_c dt)

Discretization via matrix exponential (Van Loan method):
    x_{k+1} = F * x_k + w_k,    w_k ~ N(0, Q)
    z_k     = H * x_k + v_k,    v_k ~ N(0, R)

where:
    F = expm(F_c * dt)          - discrete state transition
    Q = ∫_0^dt expm(F_c*s) G_c Q_c G_c^T expm(F_c^T*s) ds  - discrete process noise

Motion Models:
    1. Constant Velocity (CV):    state = [x, y, vx, vy]
    2. Constant Acceleration (CA): state = [x, y, vx, vy, ax, ay]
    3. Coordinated Turn (CT):     state = [x, y, vx, vy, ω] - NONLINEAR
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional


class ConstantVelocityModel:
    """
    Constant Velocity (CV) Model - Linear
    
    State: x = [px, py, vx, vy]^T
    
    Continuous SDE:
        d[px]   [0 0 1 0] [px]       [0 0]
        d[py] = [0 0 0 1] [py] dt + [0 0] dw
        d[vx]   [0 0 0 0] [vx]       [1 0]
        d[vy]   [0 0 0 0] [vy]       [0 1]
    
    Discrete (exact zero-order hold):
        F = I + F_c*dt (exact for piecewise constant)
        Q derived via Van Loan method
    """
    
    def __init__(self, dt: float, sigma_a: float = 1.0):
        """
        Args:
            dt: sampling interval [s]
            sigma_a: process noise std (acceleration noise) [m/s^2]
        """
        self.dt = dt
        self.sigma_a = sigma_a
        self.n_state = 4
        self.n_meas = 2
        
        # Continuous state transition
        self.F_c = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=float)
        
        # Discrete state transition (exact ZOH)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise (Van Loan / piecewise white acceleration)
        # Q = sigma_a^2 * [dt^4/4  0       dt^3/2  0    ]
        #                  [0       dt^4/4  0       dt^3/2]
        #                  [dt^3/2  0       dt^2    0    ]
        #                  [0       dt^3/2  0       dt^2  ]
        self.Q = sigma_a**2 * np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ])
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """State transition function (linear)"""
        return self.F @ x
    
    def F_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of f (= F for linear model)"""
        return self.F
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function"""
        return self.H @ x
    
    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h"""
        return self.H
    
    def eigenvalue_stability(self) -> Tuple[np.ndarray, bool]:
        """Check stability: all |eigenvalues| ≤ 1 for discrete system"""
        eigs = np.linalg.eigvals(self.F)
        stable = np.all(np.abs(eigs) <= 1.0 + 1e-10)
        return eigs, stable
    
    def observability_matrix(self) -> np.ndarray:
        """
        Observability matrix O = [H; HF; HF^2; ...]
        System is observable iff rank(O) = n
        """
        n = self.n_state
        O = np.zeros((self.n_meas * n, n))
        HFk = self.H.copy()
        for k in range(n):
            O[k*self.n_meas:(k+1)*self.n_meas, :] = HFk
            HFk = HFk @ self.F
        return O
    
    def controllability_matrix(self) -> np.ndarray:
        """
        Controllability matrix C = [B, FB, F^2B, ...]
        Using process noise input B = [0,0;0,0;1,0;0,1]*sqrt(Q_c)
        """
        n = self.n_state
        B = np.array([[0,0],[0,0],[1,0],[0,1]], dtype=float) * self.sigma_a
        C = np.zeros((n, n * 2))
        FkB = B.copy()
        for k in range(n):
            C[:, k*2:(k+1)*2] = FkB
            FkB = self.F @ FkB
        return C
    
    def is_observable(self) -> Tuple[bool, int]:
        O = self.observability_matrix()
        rank = np.linalg.matrix_rank(O)
        return rank == self.n_state, rank
    
    def initial_state(self, pos: np.ndarray, vel: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.zeros(self.n_state)
        x[:2] = pos
        if vel is not None:
            x[2:] = vel
        return x


class ConstantAccelerationModel:
    """
    Constant Acceleration (CA / Singer) Model - Linear
    
    State: x = [px, py, vx, vy, ax, ay]^T
    
    Continuous SDE (acceleration driven by white noise jerk):
        ẍ = w,  w ~ N(0, sigma_j^2)
    """
    
    def __init__(self, dt: float, sigma_j: float = 0.5):
        self.dt = dt
        self.sigma_j = sigma_j
        self.n_state = 6
        self.n_meas = 2
        
        dt2 = dt**2
        dt3 = dt**3
        
        self.F = np.array([
            [1, 0, dt, 0, dt2/2, 0    ],
            [0, 1, 0, dt, 0,    dt2/2 ],
            [0, 0, 1, 0, dt,    0     ],
            [0, 0, 0, 1, 0,     dt    ],
            [0, 0, 0, 0, 1,     0     ],
            [0, 0, 0, 0, 0,     1     ]
        ], dtype=float)
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)
        
        # Process noise (jerk input)
        G = np.array([dt3/6, 0, dt2/2, 0, dt, 0,
                      0, dt3/6, 0, dt2/2, 0, dt]).reshape(6,2)
        self.Q = sigma_j**2 * G @ G.T
    
    def f(self, x): return self.F @ x
    def F_jacobian(self, x): return self.F
    def h(self, x): return self.H @ x
    def H_jacobian(self, x): return self.H
    
    def eigenvalue_stability(self):
        eigs = np.linalg.eigvals(self.F)
        return eigs, np.all(np.abs(eigs) <= 1.0 + 1e-10)
    
    def observability_matrix(self):
        n = self.n_state
        O = np.zeros((self.n_meas * n, n))
        HFk = self.H.copy()
        for k in range(n):
            O[k*self.n_meas:(k+1)*self.n_meas, :] = HFk
            HFk = HFk @ self.F
        return O
    
    def is_observable(self):
        O = self.observability_matrix()
        rank = np.linalg.matrix_rank(O)
        return rank == self.n_state, rank


class CoordinatedTurnModel:
    """
    Coordinated Turn (CT) Model - NONLINEAR
    
    State: x = [px, py, vx, vy, ω]^T
    where ω = turn rate [rad/s]
    
    Nonlinear dynamics (exact integration over dt):
        px_{k+1} = px_k + sin(ω*dt)/ω * vx_k - (1-cos(ω*dt))/ω * vy_k
        py_{k+1} = py_k + (1-cos(ω*dt))/ω * vx_k + sin(ω*dt)/ω * vy_k
        vx_{k+1} = cos(ω*dt)*vx_k - sin(ω*dt)*vy_k
        vy_{k+1} = sin(ω*dt)*vx_k + cos(ω*dt)*vy_k
        ω_{k+1}  = ω_k
    
    Requires EKF or UKF (nonlinear f)
    """
    
    def __init__(self, dt: float, sigma_a: float = 0.5, sigma_omega: float = 0.01):
        self.dt = dt
        self.sigma_a = sigma_a
        self.sigma_omega = sigma_omega
        self.n_state = 5
        self.n_meas = 2
        
        # Process noise covariance
        self.Q = np.diag([
            sigma_a**2 * dt**2 / 4,
            sigma_a**2 * dt**2 / 4,
            sigma_a**2,
            sigma_a**2,
            sigma_omega**2 * dt**2
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ], dtype=float)
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear state transition (exact for CT model)"""
        px, py, vx, vy, omega = x
        dt = self.dt
        
        # Handle near-zero turn rate (Taylor expansion for numerical stability)
        if abs(omega) < 1e-6:
            # Limit as omega -> 0: straight-line motion
            px_new = px + vx * dt
            py_new = py + vy * dt
            vx_new = vx
            vy_new = vy
        else:
            s = np.sin(omega * dt)
            c = np.cos(omega * dt)
            sin_o = s / omega
            cos_o = (1 - c) / omega
            
            px_new = px + sin_o * vx - cos_o * vy
            py_new = py + cos_o * vx + sin_o * vy
            vx_new = c * vx - s * vy
            vy_new = s * vx + c * vy
        
        return np.array([px_new, py_new, vx_new, vy_new, omega])
    
    def F_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian ∂f/∂x for EKF linearization
        Computed analytically.
        """
        px, py, vx, vy, omega = x
        dt = self.dt
        
        J = np.eye(5)
        
        if abs(omega) < 1e-6:
            J[0, 2] = dt
            J[1, 3] = dt
            J[0, 4] = -dt**2 * vy / 2
            J[1, 4] =  dt**2 * vx / 2
        else:
            s = np.sin(omega * dt)
            c = np.cos(omega * dt)
            o2 = omega**2
            
            # ∂px_new/∂vx, ∂px_new/∂vy, ∂px_new/∂ω
            J[0, 2] = s / omega
            J[0, 3] = -(1 - c) / omega
            J[0, 4] = (dt * c * omega - s) / o2 * vx - (dt * s * omega - (1-c)) / o2 * vy
            
            # ∂py_new/∂vx, ∂py_new/∂vy, ∂py_new/∂ω
            J[1, 2] = (1 - c) / omega
            J[1, 3] = s / omega
            J[1, 4] = (dt * s * omega - (1-c)) / o2 * vx + (dt * c * omega - s) / o2 * vy
            
            # ∂vx_new/∂vx, ∂vx_new/∂vy, ∂vx_new/∂ω
            J[2, 2] = c
            J[2, 3] = -s
            J[2, 4] = -dt * s * vx - dt * c * vy
            
            # ∂vy_new/∂vx, ∂vy_new/∂vy, ∂vy_new/∂ω
            J[3, 2] = s
            J[3, 3] = c
            J[3, 4] = dt * c * vx - dt * s * vy
        
        return J
    
    def h(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x
    
    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.H
    
    def initial_state(self, pos, vel, omega=0.0):
        return np.array([pos[0], pos[1], vel[0], vel[1], omega])


def simulate_trajectory(model, x0: np.ndarray, T: int, R: np.ndarray,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate ground truth and measurements.
    
    Args:
        model: motion model
        x0: initial state
        T: number of timesteps
        R: measurement noise covariance
        seed: random seed
    
    Returns:
        states: (T+1, n) true states
        measurements: (T, m) noisy measurements
    """
    rng = np.random.default_rng(seed)
    n = model.n_state
    m = model.n_meas
    
    states = np.zeros((T + 1, n))
    measurements = np.zeros((T, m))
    states[0] = x0
    
    # Cholesky for numerically stable multivariate sampling
    L_Q = np.linalg.cholesky(model.Q + 1e-12 * np.eye(n))
    L_R = np.linalg.cholesky(R)
    
    for t in range(T):
        w = L_Q @ rng.standard_normal(n)
        states[t+1] = model.f(states[t]) + w
        
        v = L_R @ rng.standard_normal(m)
        measurements[t] = model.h(states[t+1]) + v
    
    return states, measurements
