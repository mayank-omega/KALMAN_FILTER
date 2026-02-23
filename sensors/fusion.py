"""
Multi-Sensor Fusion: GPS + IMU
================================

Architecture:
    - GPS: position measurements at low rate (1 Hz), high noise
    - IMU: acceleration measurements at high rate (100 Hz), low noise
    
    Combined state: x = [px, py, vx, vy, ax, ay, bax, bay]
    where bax, bay = accelerometer biases (slowly drifting)

Fusion Strategy:
    - Tight coupling via augmented state
    - GPS updates whenever available (gating on Mahalanobis distance)
    - IMU drives the prediction step at full rate
    
Sensor Models:
    GPS:  z_gps = [px, py] + N(0, R_gps),  R_gps ~ diag(5², 5²) m²
    IMU:  z_imu = [ax, ay] + bias + N(0, R_imu),  R_imu ~ diag(0.1², 0.1²) m²/s²

Missing data handled by skipping the update step.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from ..filters.kalman_filter import KalmanFilter, KFResult


class GPSIMUFusion:
    """
    Loosely-coupled GPS/IMU fusion via Extended Kalman Filter on augmented state.
    
    State: [px, py, vx, vy, bax, bay]  (position, velocity, accel bias)
    GPS updates: position (2 measurements)
    IMU updates: used in prediction, bias corrected
    """
    
    def __init__(self, dt_imu: float = 0.01, dt_gps: float = 1.0,
                 sigma_process: float = 0.5,
                 sigma_gps: float = 3.0,      # GPS position noise [m]
                 sigma_imu: float = 0.1,      # IMU noise [m/s²]
                 sigma_bias: float = 0.01):   # Bias random walk
        
        self.dt_imu = dt_imu
        self.dt_gps = dt_gps
        self.ratio = int(dt_gps / dt_imu)
        
        self.n_state = 6  # [px, py, vx, vy, bax, bay]
        
        # State: px, py, vx, vy, bax, bay
        dt = dt_imu
        self.F = np.array([
            [1, 0, dt, 0, dt**2/2, 0       ],
            [0, 1, 0, dt, 0,       dt**2/2 ],
            [0, 0, 1, 0,  dt,      0       ],
            [0, 0, 0, 1,  0,       dt      ],
            [0, 0, 0, 0,  1,       0       ],  # bias random walk
            [0, 0, 0, 0,  0,       1       ]
        ], dtype=float)
        
        # GPS measurement: observe position only
        self.H_gps = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)
        
        # IMU-corrected observation (acceleration - bias)
        self.H_imu = np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # Process noise
        G = np.zeros((6, 2))
        G[0, 0] = dt**2/2; G[1, 1] = dt**2/2
        G[2, 0] = dt;      G[3, 1] = dt
        G[4, 0] = dt;      G[5, 1] = dt   # bias process noise
        
        self.Q = sigma_process**2 * G @ G.T
        self.Q[4, 4] += sigma_bias**2 * dt
        self.Q[5, 5] += sigma_bias**2 * dt
        
        # Measurement noises
        self.R_gps = np.diag([sigma_gps**2, sigma_gps**2])
        self.R_imu = np.diag([sigma_imu**2, sigma_imu**2])
        
        # Store sensor noises for simulation
        self.sigma_gps = sigma_gps
        self.sigma_imu = sigma_imu
    
    def simulate(self, T_total: float, true_trajectory: callable,
                 seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Simulate GPS+IMU data along a true trajectory.
        
        Args:
            T_total: total simulation time [s]
            true_trajectory: callable t -> (px, py, vx, vy, ax, ay)
            seed: random seed
        
        Returns:
            dict with gps_measurements, imu_measurements, true_states, times
        """
        rng = np.random.default_rng(seed)
        
        n_imu = int(T_total / self.dt_imu)
        n_gps = int(T_total / self.dt_gps)
        
        t_imu = np.arange(n_imu) * self.dt_imu
        t_gps = np.arange(n_gps) * self.dt_gps
        
        true_states_imu = np.array([true_trajectory(t) for t in t_imu])
        
        # GPS measurements (position + noise)
        gps_times = np.arange(n_gps) * self.dt_gps
        gps_idx = (gps_times / self.dt_imu).astype(int)
        gps_idx = np.clip(gps_idx, 0, n_imu-1)
        
        gps_true = true_states_imu[gps_idx, :2]
        gps_meas = gps_true + rng.normal(0, self.sigma_gps, gps_true.shape)
        
        # IMU measurements (acceleration + noise + bias)
        true_accel = true_states_imu[:, 4:6]
        bias = np.array([0.05, -0.03])  # constant bias for simulation
        imu_meas = true_accel + bias + rng.normal(0, self.sigma_imu, true_accel.shape)
        
        return {
            'gps_measurements': gps_meas,
            'gps_times': gps_times,
            'imu_measurements': imu_meas,
            'imu_times': t_imu,
            'true_states': true_states_imu,
            'true_bias': bias
        }
    
    def run_fusion(self, sim_data: Dict, x0: np.ndarray, P0: np.ndarray,
                   mahal_threshold: float = 9.21) -> Dict[str, np.ndarray]:
        """
        Run GPS/IMU fusion filter.
        
        Args:
            sim_data: output from simulate()
            x0: initial state
            P0: initial covariance
            mahal_threshold: chi-squared threshold for GPS gating (dof=2, 99%)
        
        Returns:
            dict with filtered states, covariances, innovations
        """
        n_imu = len(sim_data['imu_times'])
        gps_meas = sim_data['gps_measurements']
        imu_meas = sim_data['imu_measurements']
        
        x = x0.copy()
        P = P0.copy()
        
        states_filt = []
        covs_filt = []
        gps_updates = []
        
        gps_counter = 0
        
        for k in range(n_imu):
            t = sim_data['imu_times'][k]
            
            # PREDICTION using IMU as control input
            # Correct acceleration by removing estimated bias
            a_corrected = imu_meas[k] - x[4:6]
            
            # Augment F with IMU control
            x_pred = self.F @ x
            # Override velocity prediction with IMU
            x_pred[2] += self.dt_imu * a_corrected[0]
            x_pred[3] += self.dt_imu * a_corrected[1]
            
            P_pred = self.F @ P @ self.F.T + self.Q
            P_pred = 0.5 * (P_pred + P_pred.T)
            
            x = x_pred
            P = P_pred
            
            # GPS UPDATE (when available)
            if (gps_counter < len(gps_meas) and 
                abs(t - sim_data['gps_times'][gps_counter]) < self.dt_imu/2):
                
                z = gps_meas[gps_counter]
                H = self.H_gps
                R = self.R_gps
                
                innov = z - H @ x
                S = H @ P @ H.T + R
                
                try:
                    S_inv = np.linalg.inv(S)
                except:
                    S_inv = np.linalg.pinv(S)
                
                mahal = float(innov @ S_inv @ innov)
                
                # Mahalanobis gating
                if mahal < mahal_threshold:
                    K = P @ H.T @ S_inv
                    x = x + K @ innov
                    IKH = np.eye(self.n_state) - K @ H
                    P = IKH @ P @ IKH.T + K @ R @ K.T
                    P = 0.5 * (P + P.T)
                    
                    gps_updates.append({
                        'time': t, 'innovation': innov, 'mahal': mahal, 'accepted': True
                    })
                else:
                    gps_updates.append({
                        'time': t, 'innovation': innov, 'mahal': mahal, 'accepted': False
                    })
                
                gps_counter += 1
            
            states_filt.append(x.copy())
            covs_filt.append(P.copy())
        
        return {
            'states': np.array(states_filt),
            'covariances': np.array(covs_filt),
            'gps_updates': gps_updates,
            'times': sim_data['imu_times']
        }
