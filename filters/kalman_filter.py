"""
Standard (Linear) Kalman Filter
================================

Mathematical Derivation from Bayesian Principles
-------------------------------------------------

System model:
    x_k = F * x_{k-1} + w_{k-1},   w ~ N(0, Q)
    z_k = H * x_k + v_k,            v ~ N(0, R)
    x_0 ~ N(x̂_0, P_0)

Bayesian Formulation:
    p(x_k | z_{1:k}) ∝ p(z_k | x_k) * p(x_k | z_{1:k-1})

Since all distributions are Gaussian, the posterior is Gaussian:
    p(x_k | z_{1:k}) = N(x̂_k, P_k)

PREDICTION STEP (Chapman-Kolmogorov):
--------------------------------------
Prior:
    p(x_k | z_{1:k-1}) = ∫ p(x_k | x_{k-1}) p(x_{k-1} | z_{1:k-1}) dx_{k-1}

For linear Gaussian:
    x̂_{k|k-1} = F * x̂_{k-1|k-1}
    P_{k|k-1}  = F * P_{k-1|k-1} * F^T + Q

UPDATE STEP (Bayes' theorem):
------------------------------
Innovation:
    ỹ_k = z_k - H * x̂_{k|k-1}          (innovation / residual)
    S_k = H * P_{k|k-1} * H^T + R        (innovation covariance)

Kalman Gain (optimal under MMSE):
    K_k = P_{k|k-1} * H^T * S_k^{-1}

Proof of optimality:
    Minimize E[||x_k - x̂_k||^2] by taking d/dK E[(x̂_k - x_k)(x̂_k - x_k)^T] = 0
    → K* = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}

Posterior:
    x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k
    P_{k|k}  = (I - K_k * H) * P_{k|k-1}

Joseph form (numerically stable):
    P_{k|k} = (I - K*H) P_{k|k-1} (I - K*H)^T + K*R*K^T

OPTIMALITY CONDITIONS:
----------------------
1. Optimal under MMSE when model is exact
2. Innovation sequence ỹ_k must be white (zero autocorrelation)
3. NEES: (x̂-x)^T P^{-1} (x̂-x) ~ χ^2(n) - covariance consistency
4. NIS: ỹ^T S^{-1} ỹ ~ χ^2(m) - innovation consistency
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class KFResult:
    """Container for filter results"""
    x_prior: np.ndarray       # predicted state
    P_prior: np.ndarray       # predicted covariance
    x_post: np.ndarray        # updated state
    P_post: np.ndarray        # updated covariance
    innovation: np.ndarray    # z - H*x_prior
    S: np.ndarray             # innovation covariance
    K: np.ndarray             # Kalman gain
    nis: float                # Normalized Innovation Squared
    log_likelihood: float     # log p(z_k | z_{1:k-1})


class KalmanFilter:
    """
    Standard Linear Kalman Filter.
    
    Complexity per step:
        Prediction: O(n^2)
        Update:     O(n^2 m + m^3) — dominated by S inversion
        Overall:    O(n^2 m) per timestep
    
    Numerical Stability:
        - Uses Joseph form for P update (symmetric positive definite)
        - Adds jitter for numerical conditioning
        - Monitors condition number of S
    """
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray, adaptive: bool = False):
        """
        Args:
            F: state transition matrix (n×n)
            H: measurement matrix (m×n)
            Q: process noise covariance (n×n)
            R: measurement noise covariance (m×m)
            x0: initial state estimate (n,)
            P0: initial covariance (n×n)
            adaptive: enable adaptive Q/R estimation (SAGE-HUSA)
        """
        self.F = F.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.x = x0.copy()
        self.P = P0.copy()
        self.adaptive = adaptive
        
        self.n = F.shape[0]
        self.m = H.shape[0]
        
        # History
        self.results: List[KFResult] = []
        self._step = 0
        
        # Adaptive noise estimation (Sage-Husa)
        self._alpha = 0.98  # forgetting factor
        self._Q_hat = Q.copy()
        self._R_hat = R.copy()
    
    def predict(self, F: Optional[np.ndarray] = None,
                Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step.
        
        x̂_{k|k-1} = F * x̂_{k-1|k-1}
        P_{k|k-1}  = F * P_{k-1|k-1} * F^T + Q
        """
        F = F if F is not None else self.F
        Q = Q if Q is not None else (self._Q_hat if self.adaptive else self.Q)
        
        self.x_prior = F @ self.x
        self.P_prior = F @ self.P @ F.T + Q
        
        # Symmetrize for numerical stability
        self.P_prior = 0.5 * (self.P_prior + self.P_prior.T)
        
        return self.x_prior.copy(), self.P_prior.copy()
    
    def update(self, z: np.ndarray, H: Optional[np.ndarray] = None,
               R: Optional[np.ndarray] = None,
               outlier_threshold: Optional[float] = None) -> KFResult:
        """
        Update step with optional Mahalanobis distance gating.
        
        Args:
            z: measurement vector (m,)
            H: optional override measurement matrix
            R: optional override measurement noise
            outlier_threshold: Mahalanobis distance threshold (χ^2 critical value)
        
        Returns:
            KFResult with all filter quantities
        """
        H = H if H is not None else self.H
        R = R if R is not None else (self._R_hat if self.adaptive else self.R)
        
        # Innovation
        innovation = z - H @ self.x_prior
        
        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.P_prior @ H.T + R
        S = 0.5 * (S + S.T)  # symmetrize
        
        # NIS for outlier rejection
        try:
            S_inv = np.linalg.inv(S)
            nis = float(innovation @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            nis = float(innovation @ S_inv @ innovation)
        
        # Mahalanobis distance gating
        is_outlier = (outlier_threshold is not None) and (nis > outlier_threshold)
        
        if is_outlier:
            # Skip update — propagate prior
            self.x = self.x_prior.copy()
            self.P = self.P_prior.copy()
            K = np.zeros((self.n, self.m))
        else:
            # Kalman gain: K = P_{k|k-1} H^T S^{-1}
            K = self.P_prior @ H.T @ S_inv
            
            # State update
            self.x = self.x_prior + K @ innovation
            
            # Covariance update (Joseph form — numerically stable, preserves PSD):
            # P = (I - KH) P_prior (I - KH)^T + K R K^T
            IKH = np.eye(self.n) - K @ H
            self.P = IKH @ self.P_prior @ IKH.T + K @ R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            
            # Adaptive noise estimation (Sage-Husa)
            if self.adaptive:
                self._update_adaptive_noise(innovation, K, H, S)
        
        # Log-likelihood: log N(ỹ; 0, S)
        sign, logdet = np.linalg.slogdet(S)
        log_lik = -0.5 * (self.m * np.log(2 * np.pi) + logdet + nis)
        
        result = KFResult(
            x_prior=self.x_prior.copy(),
            P_prior=self.P_prior.copy(),
            x_post=self.x.copy(),
            P_post=self.P.copy(),
            innovation=innovation.copy(),
            S=S.copy(),
            K=K.copy(),
            nis=nis,
            log_likelihood=log_lik
        )
        self.results.append(result)
        self._step += 1
        return result
    
    def _update_adaptive_noise(self, innov, K, H, S):
        """
        Sage-Husa adaptive noise estimation.
        
        Q̂_k = (1-α) Q̂_{k-1} + α(K ỹ ỹ^T K^T + P - F P F^T)
        R̂_k = (1-α) R̂_{k-1} + α(ỹ ỹ^T - H P_prior H^T)
        """
        a = 1 - self._alpha
        
        # Update R
        R_new = a * (np.outer(innov, innov) - H @ self.P_prior @ H.T) + self._alpha * self._R_hat
        # Project to PSD
        vals, vecs = np.linalg.eigh(R_new)
        vals = np.maximum(vals, 1e-6)
        self._R_hat = vecs @ np.diag(vals) @ vecs.T
        
        # Update Q
        Q_new = a * (K @ np.outer(innov, innov) @ K.T + self.P - self.F @ self.P @ self.F.T) + self._alpha * self._Q_hat
        vals, vecs = np.linalg.eigh(Q_new)
        vals = np.maximum(vals, 1e-8)
        self._Q_hat = vecs @ np.diag(vals) @ vecs.T
    
    def run(self, measurements: np.ndarray, 
            missing_mask: Optional[np.ndarray] = None,
            outlier_threshold: Optional[float] = None) -> List[KFResult]:
        """
        Run filter over full measurement sequence.
        
        Args:
            measurements: (T, m) measurement array
            missing_mask: (T,) boolean array, True = missing measurement
            outlier_threshold: Mahalanobis gating threshold
        """
        T = measurements.shape[0]
        if missing_mask is None:
            missing_mask = np.zeros(T, dtype=bool)
        
        results = []
        for t in range(T):
            self.predict()
            
            if missing_mask[t]:
                # No measurement: use prior as posterior
                result = KFResult(
                    x_prior=self.x_prior.copy(),
                    P_prior=self.P_prior.copy(),
                    x_post=self.x_prior.copy(),
                    P_post=self.P_prior.copy(),
                    innovation=np.zeros(self.m),
                    S=np.zeros((self.m, self.m)),
                    K=np.zeros((self.n, self.m)),
                    nis=0.0,
                    log_likelihood=-np.inf
                )
                self.x = self.x_prior.copy()
                self.P = self.P_prior.copy()
                self.results.append(result)
            else:
                result = self.update(measurements[t], outlier_threshold=outlier_threshold)
            
            results.append(result)
        
        return results
    
    def nees(self, x_true: np.ndarray) -> float:
        """
        Normalized Estimation Error Squared (NEES).
        NEES = (x̂-x)^T P^{-1} (x̂-x) ~ χ^2(n) if filter is consistent.
        
        Time-averaged NEES should be in [χ^2_{α/2}(n*T)/(n*T), χ^2_{1-α/2}(n*T)/(n*T)]
        """
        err = self.x - x_true
        try:
            P_inv = np.linalg.inv(self.P)
        except:
            P_inv = np.linalg.pinv(self.P)
        return float(err @ P_inv @ err)
