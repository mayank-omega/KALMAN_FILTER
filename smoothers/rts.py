"""
Rauch-Tung-Striebel (RTS) Smoother
=====================================

The Kalman filter is causal (uses measurements up to time k).
The RTS smoother refines estimates using ALL measurements (batch, offline).

FORWARD PASS: Standard Kalman Filter
    x̂_{k|k},  P_{k|k}     (filtered estimates)
    x̂_{k|k-1}, P_{k|k-1}  (predicted estimates)

BACKWARD PASS (RTS equations):
    Starting from k=T-1 down to k=0:
    
    Smoother gain:
        G_k = P_{k|k} F^T P_{k+1|k}^{-1}
    
    Smoothed state:
        x̂_{k|T} = x̂_{k|k} + G_k (x̂_{k+1|T} - x̂_{k+1|k})
    
    Smoothed covariance:
        P_{k|T} = P_{k|k} + G_k (P_{k+1|T} - P_{k+1|k}) G_k^T

DERIVATION:
-----------
From Bayesian smoothing:
    p(x_k | z_{1:T}) ∝ p(x_k | z_{1:k}) * p(z_{k+1:T} | x_k)

The backward information factor p(z_{k+1:T} | x_k) is Gaussian due to linear Gaussian model.
The product of two Gaussians is Gaussian, yielding the RTS recursion.

IMPROVEMENT:
-----------
Smoother is ALWAYS better than or equal to filter:
    P_{k|T} ≼ P_{k|k}  (positive semidefinite inequality)

because smoother uses more information. Equality only at k=T.

COMPLEXITY: O(n^3) per step backward (matrix products), O(T*n^3) total
"""

import numpy as np
from typing import List, Tuple, Optional
from filters.kalman_filter import KFResult


class RTSSmoother:
    """
    RTS Smoother for Linear Gaussian Systems.
    Requires the full forward Kalman filter pass first.
    """
    
    def __init__(self, F: np.ndarray):
        """
        Args:
            F: state transition matrix (n×n)
        """
        self.F = F.copy()
        self.n = F.shape[0]
    
    def smooth(self, kf_results: List[KFResult]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run backward RTS smoother.
        
        Args:
            kf_results: list of KFResult from forward KF pass
        
        Returns:
            x_smooth: (T, n) smoothed state estimates
            P_smooth: (T, n, n) smoothed covariance matrices
        """
        T = len(kf_results)
        n = self.n
        
        x_smooth = np.zeros((T, n))
        P_smooth = np.zeros((T, n, n))
        
        # Initialize with last filtered estimate
        x_smooth[-1] = kf_results[-1].x_post
        P_smooth[-1] = kf_results[-1].P_post
        
        for k in range(T-2, -1, -1):
            x_k  = kf_results[k].x_post
            P_k  = kf_results[k].P_post
            x_k1_prior = kf_results[k+1].x_prior
            P_k1_prior = kf_results[k+1].P_prior
            
            # Smoother gain: G_k = P_{k|k} F^T P_{k+1|k}^{-1}
            try:
                P_pred_inv = np.linalg.inv(P_k1_prior)
            except:
                P_pred_inv = np.linalg.pinv(P_k1_prior)
            
            G_k = P_k @ self.F.T @ P_pred_inv
            
            # Smoothed state
            x_smooth[k] = x_k + G_k @ (x_smooth[k+1] - x_k1_prior)
            
            # Smoothed covariance
            dP = P_smooth[k+1] - P_k1_prior
            P_smooth[k] = P_k + G_k @ dP @ G_k.T
            P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)
        
        return x_smooth, P_smooth
    
    def smooth_gain(self, kf_results: List[KFResult]) -> np.ndarray:
        """Compute smoother gains G_k for analysis"""
        T = len(kf_results)
        gains = np.zeros((T-1, self.n, self.n))
        for k in range(T-1):
            P_k = kf_results[k].P_post
            P_k1_prior = kf_results[k+1].P_prior
            try:
                P_pred_inv = np.linalg.inv(P_k1_prior)
            except:
                P_pred_inv = np.linalg.pinv(P_k1_prior)
            gains[k] = P_k @ self.F.T @ P_pred_inv
        return gains


class UKFRTSSmoother:
    """
    RTS Smoother for Nonlinear Systems (UKF-based).
    Uses cross-covariance from UKF sigma points for backward pass.
    """
    
    def __init__(self, f: callable, n: int, alpha: float = 1e-3, 
                 beta: float = 2.0, kappa: float = 0.0):
        self.f = f
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Compute weights
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam
        self.W_m = np.full(2*n+1, 1.0/(2*(n+lam)))
        self.W_c = np.full(2*n+1, 1.0/(2*(n+lam)))
        self.W_m[0] = lam / (n+lam)
        self.W_c[0] = lam / (n+lam) + (1 - alpha**2 + beta)
    
    def _sigma_points(self, x, P):
        n, lam = self.n, self.lam
        try:
            L = np.linalg.cholesky((n+lam)*P)
        except:
            L = np.linalg.cholesky((n+lam)*(P + 1e-8*np.eye(n)))
        pts = np.zeros((2*n+1, n))
        pts[0] = x
        for i in range(n):
            pts[i+1] = x + L[:, i]
            pts[n+i+1] = x - L[:, i]
        return pts
    
    def smooth(self, ukf_results: List[KFResult], Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF-RTS smoother backward pass.
        
        Smoother gain uses cross-covariance computed from sigma points:
            P_{x,x^+} = Σ W_i^c (Χ_i - x̂)(f(Χ_i) - x̂^+)^T
            G_k = P_{x,x^+} P_{k+1|k}^{-1}
        """
        T = len(ukf_results)
        x_smooth = np.zeros((T, self.n))
        P_smooth = np.zeros((T, self.n, self.n))
        
        x_smooth[-1] = ukf_results[-1].x_post
        P_smooth[-1] = ukf_results[-1].P_post
        
        for k in range(T-2, -1, -1):
            x_k = ukf_results[k].x_post
            P_k = ukf_results[k].P_post
            
            # Generate sigma points at filtered estimate
            sigma_pts = self._sigma_points(x_k, P_k)
            sigma_f = np.array([self.f(sigma_pts[i]) for i in range(2*self.n+1)])
            
            x_pred = np.einsum('i,ij->j', self.W_m, sigma_f)
            
            # Cross-covariance
            dx = sigma_pts - x_k
            df = sigma_f - x_pred
            P_cross = np.einsum('i,ij,ik->jk', self.W_c, dx, df)
            
            P_k1_prior = ukf_results[k+1].P_prior
            
            try:
                G_k = P_cross @ np.linalg.inv(P_k1_prior)
            except:
                G_k = P_cross @ np.linalg.pinv(P_k1_prior)
            
            x_smooth[k] = x_k + G_k @ (x_smooth[k+1] - ukf_results[k+1].x_prior)
            dP = P_smooth[k+1] - P_k1_prior
            P_smooth[k] = P_k + G_k @ dP @ G_k.T
            P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)
        
        return x_smooth, P_smooth
