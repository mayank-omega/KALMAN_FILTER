"""
Extended Kalman Filter (EKF)
=============================

Motivation:
-----------
For nonlinear systems:
    x_k = f(x_{k-1}) + w_{k-1},   w ~ N(0, Q)
    z_k = h(x_k) + v_k,            v ~ N(0, R)

The posterior p(x_k | z_{1:k}) is no longer Gaussian.

EKF APPROXIMATION:
------------------
Linearize f and h around the current estimate using first-order Taylor expansion:

    f(x) ≈ f(x̂) + F_k (x - x̂),   F_k = ∂f/∂x|_{x=x̂}   (Jacobian)
    h(x) ≈ h(x̂) + H_k (x - x̂),   H_k = ∂h/∂x|_{x=x̂}

PREDICTION STEP:
----------------
    x̂_{k|k-1} = f(x̂_{k-1|k-1})             (propagate through nonlinear f)
    P_{k|k-1}  = F_k P_{k-1|k-1} F_k^T + Q  (linearized covariance)

UPDATE STEP:
------------
    ỹ_k = z_k - h(x̂_{k|k-1})                (innovation via nonlinear h)
    S_k = H_k P_{k|k-1} H_k^T + R
    K_k = P_{k|k-1} H_k^T S_k^{-1}
    x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k
    P_{k|k}  = (I - K_k H_k) P_{k|k-1}

LIMITATIONS:
------------
1. Only first-order accurate (ignores higher-order terms)
2. Can diverge if nonlinearity is severe or initial error is large
3. Requires analytical Jacobians (or numerical differentiation)
4. Not guaranteed to be MMSE optimal (unlike KF)

COMPUTATIONAL COMPLEXITY: O(n^2 m + m^3) per step (same as KF)
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
from .kalman_filter import KFResult


class ExtendedKalmanFilter:
    """
    EKF for nonlinear state estimation.
    Requires user-provided Jacobian functions.
    """
    
    def __init__(self, f: Callable, h: Callable,
                 F_jac: Callable, H_jac: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray,
                 numerical_jac: bool = False,
                 eps: float = 1e-5):
        """
        Args:
            f: nonlinear state transition f(x) -> x'
            h: nonlinear measurement function h(x) -> z
            F_jac: Jacobian of f, F_jac(x) -> (n×n)
            H_jac: Jacobian of h, H_jac(x) -> (m×n)
            Q: process noise covariance
            R: measurement noise covariance
            x0: initial state
            P0: initial covariance
            numerical_jac: use numerical Jacobians (finite difference)
            eps: finite difference step for numerical Jacobians
        """
        self.f = f
        self.h = h
        self._F_jac = F_jac
        self._H_jac = H_jac
        self.Q = Q.copy()
        self.R = R.copy()
        self.x = x0.copy()
        self.P = P0.copy()
        self.numerical_jac = numerical_jac
        self.eps = eps
        
        self.n = len(x0)
        self.m = R.shape[0]
        
        self.results: List[KFResult] = []
    
    def _numerical_jacobian(self, func: Callable, x: np.ndarray) -> np.ndarray:
        """Central difference Jacobian: J_ij = (f_i(x+eps*e_j) - f_i(x-eps*e_j)) / (2*eps)"""
        fx = func(x)
        m = len(fx)
        n = len(x)
        J = np.zeros((m, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = self.eps
            J[:, j] = (func(x + e) - func(x - e)) / (2 * self.eps)
        return J
    
    def F_jacobian(self, x: np.ndarray) -> np.ndarray:
        if self.numerical_jac:
            return self._numerical_jacobian(self.f, x)
        return self._F_jac(x)
    
    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        if self.numerical_jac:
            return self._numerical_jacobian(self.h, x)
        return self._H_jac(x)
    
    def predict(self, Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction:
            x̂_{k|k-1} = f(x̂_{k-1})
            P_{k|k-1}  = F_k P F_k^T + Q
        """
        Q = Q if Q is not None else self.Q
        
        self.x_prior = self.f(self.x)
        Fk = self.F_jacobian(self.x)  # linearize at current estimate
        self.P_prior = Fk @ self.P @ Fk.T + Q
        self.P_prior = 0.5 * (self.P_prior + self.P_prior.T)
        self._Fk = Fk
        
        return self.x_prior.copy(), self.P_prior.copy()
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None,
               outlier_threshold: Optional[float] = None) -> KFResult:
        """
        EKF update:
            ỹ = z - h(x̂_{k|k-1})
            H_k = ∂h/∂x at x̂_{k|k-1}
            S = H_k P H_k^T + R
            K = P H_k^T S^{-1}
            x̂ = x̂_{k|k-1} + K ỹ
            P = (I - KH) P (I-KH)^T + K R K^T
        """
        R = R if R is not None else self.R
        Hk = self.H_jacobian(self.x_prior)
        
        innovation = z - self.h(self.x_prior)
        S = Hk @ self.P_prior @ Hk.T + R
        S = 0.5 * (S + S.T)
        
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
            K = self.P_prior @ Hk.T @ S_inv
            self.x = self.x_prior + K @ innovation
            
            IKH = np.eye(self.n) - K @ Hk
            self.P = IKH @ self.P_prior @ IKH.T + K @ R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
        
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
