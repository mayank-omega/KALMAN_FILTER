"""
Main Demonstration Script
==========================
Runs all filters, Monte Carlo analysis, and generates results.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import (ConstantVelocityModel, ConstantAccelerationModel,
                          CoordinatedTurnModel, simulate_trajectory)
from filters.kalman_filter import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from filters.particle_filter import ParticleFilter
from smoothers.rts import RTSSmoother, UKFRTSSmoother
from analysis.monte_carlo import MonteCarloEvaluator


def print_header(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def run_single_scenario(scenario_name, model, x0, T, R, filter_types=None, seed=42):
    print_header(f"Scenario: {scenario_name}")
    
    # Simulate
    x_true, measurements = simulate_trajectory(model, x0, T, R, seed=seed)
    
    print(f"\nSimulation: T={T} steps, dt={model.dt}s")
    print(f"True trajectory: {x_true[0,:2]} → {x_true[-1,:2]}")
    print(f"Measurement noise σ: {np.sqrt(R[0,0]):.1f} m")
    
    n = model.n_state
    P0 = np.eye(n) * 10.0
    
    results_all = {}
    
    # ── KF ──────────────────────────────────────────────────────
    if hasattr(model, 'F'):
        print("\n[KF] Standard Kalman Filter")
        kf = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
        t0 = time.perf_counter()
        kf_results = kf.run(measurements, outlier_threshold=9.21)
        kf_time = (time.perf_counter() - t0) * 1000
        
        pos_err = [np.linalg.norm(r.x_post[:2] - x_true[t+1,:2]) for t, r in enumerate(kf_results)]
        print(f"  RMSE pos: {np.sqrt(np.mean(np.array(pos_err)**2)):.3f} m")
        print(f"  Mean NIS: {np.mean([r.nis for r in kf_results]):.3f}")
        print(f"  Time: {kf_time:.2f} ms for {T} steps")
        results_all['KF'] = kf_results
        
        # RTS Smoother
        sm = RTSSmoother(F=model.F)
        xs, Ps = sm.smooth(kf_results)
        sm_err = [np.linalg.norm(xs[t,:2] - x_true[t+1,:2]) for t in range(T)]
        print(f"  RMSE pos (smoothed): {np.sqrt(np.mean(np.array(sm_err)**2)):.3f} m")
    
    # ── EKF ─────────────────────────────────────────────────────
    print("\n[EKF] Extended Kalman Filter")
    ekf = ExtendedKalmanFilter(f=model.f, h=model.h, F_jac=model.F_jacobian,
                                H_jac=model.H_jacobian, Q=model.Q, R=R,
                                x0=x0.copy(), P0=P0.copy())
    t0 = time.perf_counter()
    ekf_results = ekf.run(measurements, outlier_threshold=9.21)
    ekf_time = (time.perf_counter() - t0) * 1000
    
    pos_err = [np.linalg.norm(r.x_post[:2] - x_true[t+1,:2]) for t, r in enumerate(ekf_results)]
    print(f"  RMSE pos: {np.sqrt(np.mean(np.array(pos_err)**2)):.3f} m")
    print(f"  Mean NIS: {np.mean([r.nis for r in ekf_results]):.3f}")
    print(f"  Time: {ekf_time:.2f} ms for {T} steps")
    results_all['EKF'] = ekf_results
    
    # ── UKF ─────────────────────────────────────────────────────
    print("\n[UKF] Unscented Kalman Filter")
    ukf = UnscentedKalmanFilter(f=model.f, h=model.h, Q=model.Q, R=R,
                                 x0=x0.copy(), P0=P0.copy(), alpha=1e-3, beta=2.0)
    t0 = time.perf_counter()
    ukf_results = ukf.run(measurements, outlier_threshold=9.21)
    ukf_time = (time.perf_counter() - t0) * 1000
    
    pos_err = [np.linalg.norm(r.x_post[:2] - x_true[t+1,:2]) for t, r in enumerate(ukf_results)]
    print(f"  RMSE pos: {np.sqrt(np.mean(np.array(pos_err)**2)):.3f} m")
    print(f"  Mean NIS: {np.mean([r.nis for r in ukf_results]):.3f}")
    print(f"  Time: {ukf_time:.2f} ms for {T} steps")
    results_all['UKF'] = ukf_results
    
    # UKF-RTS Smoother
    sm_ukf = UKFRTSSmoother(f=model.f, n=n)
    xs_ukf, Ps_ukf = sm_ukf.smooth(ukf_results, Q=model.Q)
    sm_err = [np.linalg.norm(xs_ukf[t,:2] - x_true[t+1,:2]) for t in range(T)]
    print(f"  RMSE pos (UKF-smoothed): {np.sqrt(np.mean(np.array(sm_err)**2)):.3f} m")
    
    # ── PF ──────────────────────────────────────────────────────
    print("\n[PF] Particle Filter (N=1000)")
    pf = ParticleFilter(f=model.f, h=model.h, Q=model.Q, R=R,
                        x0=x0.copy(), P0=P0.copy(), N=1000)
    t0 = time.perf_counter()
    pf_results = pf.run(measurements)
    pf_time = (time.perf_counter() - t0) * 1000
    
    pos_err = [np.linalg.norm(r.x_post[:2] - x_true[t+1,:2]) for t, r in enumerate(pf_results)]
    print(f"  RMSE pos: {np.sqrt(np.mean(np.array(pos_err)**2)):.3f} m")
    print(f"  Time: {pf_time:.2f} ms for {T} steps ({pf_time/T:.2f} ms/step)")
    
    return results_all, x_true, measurements


def run_monte_carlo_analysis(model, x0, T, R, n_runs=500):
    print_header(f"Monte Carlo Analysis ({n_runs} runs)")
    
    n = model.n_state
    P0 = np.eye(n) * 10.0
    
    def sim_factory(seed):
        rng = np.random.default_rng(seed)
        x0_r = x0.copy()
        x0_r[:2] += rng.normal(0, 2, 2)
        return simulate_trajectory(model, x0_r, T, R, seed=seed)
    
    filter_configs = {
        'KF': {'factory': lambda seed: KalmanFilter(
            F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy()
        )} if hasattr(model, 'F') else None,
        'EKF': {'factory': lambda seed: ExtendedKalmanFilter(
            f=model.f, h=model.h, F_jac=model.F_jacobian, H_jac=model.H_jacobian,
            Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy()
        )},
        'UKF': {'factory': lambda seed: UnscentedKalmanFilter(
            f=model.f, h=model.h, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy()
        )},
    }
    filter_configs = {k: v for k, v in filter_configs.items() if v is not None}
    
    mc = MonteCarloEvaluator(n_runs=n_runs, n_state=n, n_meas=2)
    all_stats = {}
    
    for name, config in filter_configs.items():
        print(f"\nRunning {name}...")
        stats = mc.run(config['factory'], sim_factory, T=T, verbose=True)
        all_stats[name] = stats
    
    # Summary table
    print("\n\n" + "─"*65)
    print(f"{'Filter':<8} {'RMSE_pos':>10} {'RMSE_vel':>10} {'NEES':>8} "
          f"{'NEES_cons':>12} {'NIS':>8} {'NIS_cons':>10} {'ms/run':>8}")
    print("─"*65)
    
    for name, s in all_stats.items():
        nees_lo, nees_hi = s.nees_bounds
        nis_lo, nis_hi = s.nis_bounds
        print(f"{name:<8} {s.rmse_pos:>10.4f} {s.rmse_vel:>10.4f} "
              f"{s.mean_nees:>8.3f} "
              f"{'✓' if s.nees_consistent else '✗':>5}[{nees_lo:.2f},{nees_hi:.2f}] "
              f"{s.mean_nis:>8.3f} "
              f"{'✓' if s.nis_consistent else '✗':>8} "
              f"{s.computation_time*1000:>8.3f}")
    print("─"*65)
    
    return all_stats


def main():
    print_header("KALMAN FILTER TRACKING SYSTEM — PRODUCTION DEMO")
    print("Implementing: KF | EKF | UKF | Particle Filter | RTS Smoother")
    print("Scenarios: Constant Velocity | Constant Acceleration | Coordinated Turn")
    
    # ── SCENARIO 1: Constant Velocity ──────────────────────────────────────────
    cv_model = ConstantVelocityModel(dt=0.1, sigma_a=1.0)
    x0_cv = np.array([0., 0., 15., 5.])
    R_cv = np.eye(2) * 25.0  # 5m position noise
    
    obs, rank = cv_model.is_observable()
    eigs, stable = cv_model.eigenvalue_stability()
    print(f"\n[CV] Observable: {obs} (rank={rank}/{cv_model.n_state}), Stable: {stable}")
    
    cv_results, cv_true, cv_meas = run_single_scenario(
        "Constant Velocity", cv_model, x0_cv, T=150, R=R_cv
    )
    
    # ── SCENARIO 2: Constant Acceleration ──────────────────────────────────────
    ca_model = ConstantAccelerationModel(dt=0.1, sigma_j=0.5)
    x0_ca = np.array([0., 0., 10., 3., 0.5, 0.2])
    R_ca = np.eye(2) * 25.0
    
    obs, rank = ca_model.is_observable()
    print(f"\n[CA] Observable: {obs} (rank={rank}/{ca_model.n_state})")
    
    ca_results, ca_true, ca_meas = run_single_scenario(
        "Constant Acceleration", ca_model, x0_ca, T=150, R=R_ca
    )
    
    # ── SCENARIO 3: Coordinated Turn (NONLINEAR) ────────────────────────────────
    ct_model = CoordinatedTurnModel(dt=0.1, sigma_a=0.5, sigma_omega=0.01)
    x0_ct = np.array([0., 0., 15., 0., 0.1])  # 0.1 rad/s turn rate
    R_ct = np.eye(2) * 25.0
    
    print(f"\n[CT] Nonlinear model — EKF/UKF required")
    
    ct_results, ct_true, ct_meas = run_single_scenario(
        "Coordinated Turn (Nonlinear)", ct_model, x0_ct, T=150, R=R_ct
    )
    
    # ── MONTE CARLO ANALYSIS ───────────────────────────────────────────────────
    print("\n\nRunning Monte Carlo on Coordinated Turn (most challenging)...")
    mc_stats = run_monte_carlo_analysis(ct_model, x0_ct, T=100, R=R_ct, n_runs=200)
    
    # ── STABILITY & OBSERVABILITY SUMMARY ─────────────────────────────────────
    print_header("SYSTEM ANALYSIS SUMMARY")
    
    for model_name, model in [("CV", cv_model), ("CA", ca_model)]:
        obs, rank = model.is_observable()
        eigs, stable = model.eigenvalue_stability()
        cond_F = np.linalg.cond(model.F)
        print(f"\n{model_name}:")
        print(f"  Observable: {obs} | Rank: {rank}/{model.n_state}")
        print(f"  Eigenvalues: {[f'{abs(e):.4f}' for e in eigs]}")
        print(f"  Stable: {stable} | cond(F): {cond_F:.2e}")
    
    print_header("COMPUTATIONAL COMPLEXITY SUMMARY")
    print("  KF/EKF:  O(n²m + m³) per step   — milliseconds")
    print("  UKF:     O((2n+1)·n²) per step  — milliseconds")
    print("  PF(N):   O(N·(n+m)) per step    — N×slower")
    print("")
    print("  KF:  optimal for linear Gaussian — exact MMSE")
    print("  EKF: 1st-order Taylor approx — can diverge")
    print("  UKF: 3rd-order accurate — near-optimal for mild nonlinearity")
    print("  PF:  nonparametric — handles multimodal posterior")
    print("")
    print("Dashboard: streamlit run dashboard/app.py")
    print("API:       uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()