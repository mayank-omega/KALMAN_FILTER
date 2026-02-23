"""
Particle Filter (Sequential Monte Carlo)
=========================================

Baseline for comparison against KF/EKF/UKF.

Algorithm:
----------
Represent p(x_k | z_{1:k}) by N weighted particles {x_i, w_i}_{i=1}^N

1. INITIALIZATION:
   x_i^0 ~ p(x_0),  w_i = 1/N

2. PREDICTION (proposal via prior):
   x_i^- = f(x_i^{k-1}) + w,   w ~ N(0, Q)

3. WEIGHT UPDATE (importance weight):
   w_i ← w_i * p(z_k | x_i^-) = w_i * N(z_k; h(x_i^-), R)

4. NORMALIZE:
   w_i ← w_i / Σ w_j

5. ESTIMATE:
   x̂_k = Σ w_i * x_i

6. RESAMPLING (when effective sample size < N/2):
   N_eff = 1 / Σ w_i²
   Resample N particles from {x_i, w_i} (systematic resampling)
   Reset weights to 1/N

ADVANTAGES over KF/EKF/UKF:
----------------------------
- Non-parametric: handles non-Gaussian, multimodal distributions
- No Jacobians or sigma points needed
- Can represent any posterior shape

DISADVANTAGES:
--------------
- O(N) per step where N can be 1000-10000 particles
- Curse of dimensionality: N grows exponentially with state dim
- Particle degeneracy if proposal is poor
- Weight variance increases over time (particle impoverishment)

COMPUTATIONAL COMPARISON (n_state=4, T=150):
    KF:  O(n^2 m)  ≈ 48 ops/step   → ~microseconds
    EKF: O(n^2 m)  ≈ 48 ops/step   → ~microseconds  
    UKF: O(n^3)    ≈ 576 ops/step  → ~100 microseconds
    PF:  O(N*n)    with N=1000 → ~milliseconds
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from .kalman_filter import KFResult


class ParticleFilter:
    """
    Bootstrap Particle Filter with systematic resampling.
    Uses prior as proposal distribution.
    """
    
    def __init__(self, f: Callable, h: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray,
                 N: int = 1000,
                 resample_threshold: float = 0.5):
        """
        Args:
            f: state transition function f(x) -> x'
            h: measurement function h(x) -> z
            Q: process noise covariance
            R: measurement noise covariance
            x0: initial state mean
            P0: initial covariance
            N: number of particles
            resample_threshold: fraction of N_eff below which to resample
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.N = N
        self.thresh = resample_threshold
        
        self.n = len(x0)
        self.m = R.shape[0]
        
        # Initialize particles from prior
        L = np.linalg.cholesky(P0)
        self.particles = x0 + (L @ np.random.randn(self.n, N)).T
        self.weights = np.ones(N) / N
        
        self._L_Q = np.linalg.cholesky(Q + 1e-9 * np.eye(Q.shape[0]))
        self._R_inv = np.linalg.inv(R)
        self._log_det_R = np.linalg.slogdet(R)[1]
        
        self.results: List[KFResult] = []
    
    def _log_likelihood(self, z: np.ndarray, x: np.ndarray) -> float:
        """log N(z; h(x), R)"""
        innov = z - self.h(x)
        return -0.5 * (self.m * np.log(2*np.pi) + self._log_det_R + 
                       innov @ self._R_inv @ innov)
    
    def _systematic_resample(self) -> np.ndarray:
        """
        Systematic resampling (low-variance, O(N)):
        Draw single uniform u ~ U[0, 1/N], then equally space N samples.
        """
        cumsum = np.cumsum(self.weights)
        positions = (np.arange(self.N) + np.random.uniform(0, 1)) / self.N
        indices = np.searchsorted(cumsum, positions)
        return np.clip(indices, 0, self.N - 1)
    
    def predict(self) -> np.ndarray:
        """Propagate all particles through dynamics + noise"""
        noise = (self._L_Q @ np.random.randn(self.n, self.N)).T
        self.particles = np.array([self.f(p) for p in self.particles]) + noise
        return self._estimate()
    
    def update(self, z: np.ndarray) -> 'KFResult':
        """Update weights based on measurement likelihood"""
        log_weights = np.array([self._log_likelihood(z, p) for p in self.particles])
        
        # Numerical stability: subtract max before exp
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)
        
        # Effective sample size
        N_eff = 1.0 / np.sum(self.weights**2)
        
        if N_eff < self.thresh * self.N:
            indices = self._systematic_resample()
            self.particles = self.particles[indices].copy()
            self.weights = np.ones(self.N) / self.N
        
        x_est = self._estimate()
        P_est = self._covariance(x_est)
        
        innov = z - self.h(x_est)
        S = self.h(np.eye(self.n)) @ P_est @ self.h(np.eye(self.n)).T if False else \
            np.eye(self.m) * np.var(innov)  # approximate
        
        result = KFResult(
            x_prior=x_est, P_prior=P_est,
            x_post=x_est, P_post=P_est,
            innovation=innov, S=np.eye(self.m),
            K=np.zeros((self.n, self.m)),
            nis=0.0, log_likelihood=0.0
        )
        self.results.append(result)
        return result
    
    def _estimate(self) -> np.ndarray:
        """Weighted mean estimate"""
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def _covariance(self, mean: np.ndarray) -> np.ndarray:
        """Weighted sample covariance"""
        diffs = self.particles - mean
        return np.einsum('i,ij,ik->jk', self.weights, diffs, diffs)
    
    def run(self, measurements: np.ndarray,
            missing_mask: Optional[np.ndarray] = None,
            outlier_threshold: Optional[float] = None) -> List[KFResult]:
        T = len(measurements)
        if missing_mask is None:
            missing_mask = np.zeros(T, dtype=bool)
        
        results = []
        for t in range(T):
            self.predict()
            if not missing_mask[t]:
                result = self.update(measurements[t])
            else:
                x_est = self._estimate()
                P_est = self._covariance(x_est)
                result = KFResult(
                    x_prior=x_est, P_prior=P_est, x_post=x_est, P_post=P_est,
                    innovation=np.zeros(self.m), S=np.zeros((self.m,self.m)),
                    K=np.zeros((self.n,self.m)), nis=0.0, log_likelihood=-np.inf
                )
                self.results.append(result)
            results.append(result)
        return results
