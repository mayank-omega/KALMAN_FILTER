"""
Unscented Kalman Filter (UKF)
==============================

Key Insight:
------------
"It is easier to approximate a Gaussian distribution than to approximate 
an arbitrary nonlinear function." — Julier & Uhlmann (1997)

Instead of linearizing f and h (EKF), the UKF uses the Unscented Transform (UT):
    - Deterministically select 2n+1 sigma points that exactly capture mean and covariance
    - Propagate each sigma point through the nonlinear function
    - Recover mean and covariance from transformed sigma points (second-order accurate!)

UNSCENTED TRANSFORM:
--------------------
Given x ~ N(x̂, P), choose sigma points Χ_i and weights W_i:

    Χ_0 = x̂                                   (mean)
    Χ_i = x̂ + (√((n+λ)P))_i,   i=1,...,n
    Χ_i = x̂ - (√((n+λ)P))_{i-n}, i=n+1,...,2n

where λ = α²(n+κ) - n  (scaling parameter)

Weights:
    W_0^m = λ/(n+λ)
    W_0^c = λ/(n+λ) + (1-α²+β)
    W_i^m = W_i^c = 1/(2(n+λ)),  i=1,...,2n

Parameters:
    α ∈ (0,1]: spread of sigma points (typically 1e-3)
    β = 2:     optimal for Gaussian distributions (incorporates 4th-order info)
    κ ≥ 0:     secondary scaling (κ = 3-n is common)

TRANSFORM:
    ŷ = Σ W_i^m f(Χ_i)                   (mean through nonlinearity)
    P_yy = Σ W_i^c (f(Χ_i)-ŷ)(f(Χ_i)-ŷ)^T  (covariance)

ACCURACY COMPARISON:
--------------------
EKF:  accurate to first order (O(dt²) error in mean, O(dt²) in covariance)
UKF:  accurate to third order for Gaussian distributions!
      Mean: O(dt³) error, Covariance: O(dt³) error — superior to EKF

PREDICTION STEP:
----------------
    [Χ_{k-1|k-1}] = sigma_points(x̂_{k-1}, P_{k-1})
    Χ_{k|k-1}^i   = f(Χ_{k-1|k-1}^i)   (propagate each sigma point)
    x̂_{k|k-1}    = Σ W_i^m Χ_{k|k-1}^i
    P_{k|k-1}     = Σ W_i^c (Χ_i - x̂)(Χ_i - x̂)^T + Q

UPDATE STEP:
------------
    [Χ_{k|k-1}] = sigma_points(x̂_{k|k-1}, P_{k|k-1})  [re-compute for update]
    γ_i = h(Χ_i)
    ŷ = Σ W_i^m γ_i
    S = Σ W_i^c (γ_i - ŷ)(γ_i - ŷ)^T + R
    P_xy = Σ W_i^c (Χ_i - x̂)(γ_i - ŷ)^T
    K = P_xy S^{-1}
    x̂_{k|k} = x̂_{k|k-1} + K(z - ŷ)
    P_{k|k} = P_{k|k-1} - K S K^T
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
from .kalman_filter import KFResult


class UnscentedKalmanFilter:
    """
    UKF using the Scaled Unscented Transform.
    No Jacobians needed — works for any differentiable f and h.
    """
    
    def __init__(self, f: Callable, h: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        Args:
            f: state transition function f(x) -> x'
            h: measurement function h(x) -> z
            Q: process noise covariance (n×n)
            R: measurement noise covariance (m×m)
            x0: initial state (n,)
            P0: initial covariance (n×n)
            alpha: sigma point spread (1e-3 to 1.0)
            beta: distribution parameter (2 optimal for Gaussian)
            kappa: secondary scaling (0 or 3-n)
        """
        self.f = f
        self.h = h
        self.Q = Q.copy()
        self.R = R.copy()
        self.x = x0.copy()
        self.P = P0.copy()
        
        self.n = len(x0)
        self.m = R.shape[0]
        
        # UKF hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived parameters
        self.lam = alpha**2 * (self.n + kappa) - self.n
        
        # Weights
        n, lam = self.n, self.lam
        self.W_m = np.full(2*n+1, 1.0 / (2*(n+lam)))
        self.W_c = np.full(2*n+1, 1.0 / (2*(n+lam)))
        self.W_m[0] = lam / (n + lam)
        self.W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        
        self.results: List[KFResult] = []
    
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute 2n+1 sigma points via Cholesky factorization.
        
        Χ_0 = x̂
        Χ_i = x̂ + row_i(sqrt_matrix),   i=1,...,n
        Χ_i = x̂ - row_{i-n}(sqrt_matrix), i=n+1,...,2n
        
        where sqrt_matrix = cholesky((n+λ)P)
        """
        n = self.n
        scale = n + self.lam
        
        # Cholesky decomposition (numerically stable square root)
        try:
            L = np.linalg.cholesky(scale * P)
        except np.linalg.LinAlgError:
            # Add jitter if not PSD
            P_reg = P + 1e-8 * np.eye(n)
            L = np.linalg.cholesky(scale * P_reg)
        
        sigma_pts = np.zeros((2*n+1, n))
        sigma_pts[0] = x
        for i in range(n):
            sigma_pts[i+1]   = x + L[:, i]
            sigma_pts[n+i+1] = x - L[:, i]
        
        return sigma_pts
    
    def _unscented_transform(self, sigma_pts: np.ndarray, func: Callable,
                             noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply unscented transform through func.
        
        Returns:
            y_mean: weighted mean of transformed sigma points
            P_yy:   weighted covariance + noise
            Y:      transformed sigma points
        """
        Y = np.array([func(sigma_pts[i]) for i in range(len(sigma_pts))])
        
        # Weighted mean
        y_mean = np.einsum('i,ij->j', self.W_m, Y)
        
        # Weighted covariance
        dy = Y - y_mean
        P_yy = np.einsum('i,ij,ik->jk', self.W_c, dy, dy) + noise_cov
        P_yy = 0.5 * (P_yy + P_yy.T)
        
        return y_mean, P_yy, Y
    
    def predict(self, Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF prediction:
        1. Generate sigma points from current estimate
        2. Propagate through f
        3. Recover predicted mean and covariance
        """
        Q = Q if Q is not None else self.Q
        
        sigma_pts = self._sigma_points(self.x, self.P)
        
        x_prior, P_prior, self._sigma_f = self._unscented_transform(sigma_pts, self.f, Q)
        
        self.x_prior = x_prior
        self.P_prior = P_prior
        self._sigma_pts_pred = sigma_pts  # for cross-covariance in update
        
        return x_prior.copy(), P_prior.copy()
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None,
               outlier_threshold: Optional[float] = None) -> KFResult:
        """
        UKF update:
        1. Re-generate sigma points from prediction
        2. Propagate through h
        3. Compute cross-covariance and Kalman gain
        """
        R = R if R is not None else self.R
        
        # Re-sigma from predicted distribution
        sigma_pts = self._sigma_points(self.x_prior, self.P_prior)
        
        # Transform through measurement function
        z_pred, S, Z = self._unscented_transform(sigma_pts, self.h, R)
        
        # Cross-covariance: P_xz = Σ W_i^c (Χ_i - x̂)(γ_i - ẑ)^T
        dx = sigma_pts - self.x_prior
        dz = Z - z_pred
        P_xz = np.einsum('i,ij,ik->jk', self.W_c, dx, dz)
        
        # Innovation
        innovation = z - z_pred
        
        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.linalg.pinv(S)
        
        nis = float(innovation @ S_inv @ innovation)
        
        is_outlier = (outlier_threshold is not None) and (nis > outlier_threshold)
        
        if is_outlier:
            self.x = self.x_prior.copy()
            self.P = self.P_prior.copy()
            K = np.zeros((self.n, self.m))
        else:
            # Kalman gain: K = P_xz S^{-1}
            K = P_xz @ S_inv
            
            self.x = self.x_prior + K @ innovation
            self.P = self.P_prior - K @ S @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            
            # Ensure PSD
            vals, vecs = np.linalg.eigh(self.P)
            vals = np.maximum(vals, 1e-10)
            self.P = vecs @ np.diag(vals) @ vecs.T
        
        sign, logdet = np.linalg.slogdet(S)
        log_lik = -0.5 * (self.m * np.log(2*np.pi) + logdet + nis)
        
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
        return result
    
    def run(self, measurements: np.ndarray,
            missing_mask: Optional[np.ndarray] = None,
            outlier_threshold: Optional[float] = None) -> List[KFResult]:
        T = measurements.shape[0]
        if missing_mask is None:
            missing_mask = np.zeros(T, dtype=bool)
        
        results = []
        for t in range(T):
            self.predict()
            if missing_mask[t]:
                result = KFResult(
                    x_prior=self.x_prior.copy(), P_prior=self.P_prior.copy(),
                    x_post=self.x_prior.copy(), P_post=self.P_prior.copy(),
                    innovation=np.zeros(self.m), S=np.zeros((self.m,self.m)),
                    K=np.zeros((self.n,self.m)), nis=0.0, log_likelihood=-np.inf
                )
                self.x = self.x_prior.copy()
                self.P = self.P_prior.copy()
                self.results.append(result)
            else:
                result = self.update(measurements[t], outlier_threshold=outlier_threshold)
            results.append(result)
        return results
