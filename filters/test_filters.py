"""
Test Suite for Kalman Filter System
=====================================
Tests mathematical correctness, numerical stability, consistency.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import ConstantVelocityModel, CoordinatedTurnModel, simulate_trajectory
from filters.kalman_filter import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from smoothers.rts import RTSSmoother


def test_kf_linear_consistency():
    """KF should match EKF and UKF on linear system"""
    model = ConstantVelocityModel(dt=0.1, sigma_a=1.0)
    x0 = np.array([0., 0., 10., 5.])
    P0 = np.eye(4) * 10.0
    R = np.eye(2) * 25.0
    
    x_true, meas = simulate_trajectory(model, x0, 50, R, seed=1)
    
    kf = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
    ekf = ExtendedKalmanFilter(f=model.f, h=model.h, F_jac=model.F_jacobian,
                                H_jac=model.H_jacobian, Q=model.Q, R=R,
                                x0=x0.copy(), P0=P0.copy())
    ukf = UnscentedKalmanFilter(f=model.f, h=model.h, Q=model.Q, R=R,
                                 x0=x0.copy(), P0=P0.copy())
    
    kf_r = kf.run(meas)
    ekf_r = ekf.run(meas)
    ukf_r = ukf.run(meas)
    
    for t in range(len(meas)):
        np.testing.assert_allclose(kf_r[t].x_post, ekf_r[t].x_post, atol=1e-10,
                                    err_msg=f"KF != EKF at step {t}")
        np.testing.assert_allclose(kf_r[t].x_post, ukf_r[t].x_post, atol=1e-4,
                                    err_msg=f"KF != UKF at step {t}")
    print("✓ test_kf_linear_consistency PASSED")


def test_covariance_psd():
    """Covariance must remain positive semi-definite throughout"""
    model = CoordinatedTurnModel(dt=0.1)
    x0 = np.array([0., 0., 10., 5., 0.1])
    P0 = np.eye(5) * 10.0
    R = np.eye(2) * 25.0
    
    x_true, meas = simulate_trajectory(model, x0, 100, R, seed=2)
    
    ukf = UnscentedKalmanFilter(f=model.f, h=model.h, Q=model.Q, R=R,
                                 x0=x0.copy(), P0=P0.copy())
    results = ukf.run(meas)
    
    for t, r in enumerate(results):
        # Check PSD via eigenvalues
        eigs = np.linalg.eigvalsh(r.P_post)
        assert np.all(eigs >= -1e-8), f"P not PSD at step {t}: min eig = {eigs.min():.2e}"
    print("✓ test_covariance_psd PASSED")


def test_rts_smoother_improvement():
    """RTS smoother must have <= error than filter"""
    model = ConstantVelocityModel(dt=0.1, sigma_a=0.5)
    x0 = np.array([0., 0., 15., 5.])
    P0 = np.eye(4) * 10.0
    R = np.eye(2) * 25.0
    
    x_true, meas = simulate_trajectory(model, x0, 100, R, seed=3)
    
    kf = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
    results = kf.run(meas)
    
    sm = RTSSmoother(F=model.F)
    xs, Ps = sm.smooth(results)
    
    filter_rmse = np.sqrt(np.mean([np.sum((r.x_post[:2] - x_true[t+1,:2])**2) 
                                    for t, r in enumerate(results)]))
    smooth_rmse = np.sqrt(np.mean([np.sum((xs[t,:2] - x_true[t+1,:2])**2) 
                                    for t in range(len(results))]))
    
    assert smooth_rmse <= filter_rmse + 1e-10, \
        f"Smoother RMSE ({smooth_rmse:.4f}) > Filter RMSE ({filter_rmse:.4f})"
    print(f"✓ test_rts_smoother_improvement PASSED "
          f"(Filter={filter_rmse:.3f}m, Smoother={smooth_rmse:.3f}m)")


def test_jacobian_numerical_match():
    """Analytical Jacobian should match numerical Jacobian within tolerance"""
    from filters.ekf import ExtendedKalmanFilter
    model = CoordinatedTurnModel(dt=0.1)
    
    ekf_analytical = ExtendedKalmanFilter(
        f=model.f, h=model.h, F_jac=model.F_jacobian, H_jac=model.H_jacobian,
        Q=model.Q, R=np.eye(2), x0=np.zeros(5), P0=np.eye(5), numerical_jac=False
    )
    ekf_numerical = ExtendedKalmanFilter(
        f=model.f, h=model.h, F_jac=model.F_jacobian, H_jac=model.H_jacobian,
        Q=model.Q, R=np.eye(2), x0=np.zeros(5), P0=np.eye(5), numerical_jac=True
    )
    
    x_test = np.array([10., 5., 15., 3., 0.15])
    
    J_anal = ekf_analytical.F_jacobian(x_test)
    J_num = ekf_numerical.F_jacobian(x_test)
    
    np.testing.assert_allclose(J_anal, J_num, atol=1e-5,
                                err_msg="Analytical vs numerical Jacobian mismatch")
    print("✓ test_jacobian_numerical_match PASSED")


def test_observability():
    """CV and CA models must be fully observable"""
    from core.models import ConstantVelocityModel, ConstantAccelerationModel
    
    cv = ConstantVelocityModel(dt=0.1)
    ca = ConstantAccelerationModel(dt=0.1)
    
    obs_cv, rank_cv = cv.is_observable()
    obs_ca, rank_ca = ca.is_observable()
    
    assert obs_cv and rank_cv == 4, f"CV not observable: rank={rank_cv}"
    assert obs_ca and rank_ca == 6, f"CA not observable: rank={rank_ca}"
    print(f"✓ test_observability PASSED (CV rank={rank_cv}, CA rank={rank_ca})")


def test_outlier_rejection():
    """Mahalanobis gating should reject extreme outliers"""
    model = ConstantVelocityModel(dt=0.1)
    x0 = np.array([0., 0., 10., 5.])
    P0 = np.eye(4) * 5.0
    R = np.eye(2) * 4.0
    
    x_true, meas = simulate_trajectory(model, x0, 50, R, seed=4)
    
    # Inject outlier at step 25
    meas_outlier = meas.copy()
    meas_outlier[25] = np.array([1000., 1000.])
    
    kf_no_gate = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
    kf_gated   = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
    
    r_no_gate = kf_no_gate.run(meas_outlier, outlier_threshold=None)
    r_gated   = kf_gated.run(meas_outlier, outlier_threshold=9.21)
    
    # At step 25, gated filter should be closer to truth
    err_no_gate = np.linalg.norm(r_no_gate[25].x_post[:2] - x_true[26,:2])
    err_gated   = np.linalg.norm(r_gated[25].x_post[:2]   - x_true[26,:2])
    
    assert err_gated < err_no_gate, \
        f"Gating didn't help: gated={err_gated:.1f} >= no_gate={err_no_gate:.1f}"
    print(f"✓ test_outlier_rejection PASSED "
          f"(gated error {err_gated:.1f}m < ungated {err_no_gate:.1f}m)")


def test_missing_data():
    """Filter should handle missing measurements gracefully"""
    model = ConstantVelocityModel(dt=0.1)
    x0 = np.array([0., 0., 10., 5.])
    P0 = np.eye(4) * 5.0
    R = np.eye(2) * 4.0
    
    x_true, meas = simulate_trajectory(model, x0, 50, R, seed=5)
    
    missing_mask = np.zeros(50, dtype=bool)
    missing_mask[20:25] = True  # 5 consecutive missing
    
    kf = KalmanFilter(F=model.F, H=model.H, Q=model.Q, R=R, x0=x0.copy(), P0=P0.copy())
    results = kf.run(meas, missing_mask=missing_mask)
    
    # During missing period, covariance should increase (uncertainty grows)
    P_before = results[19].P_post
    P_after_gap = results[24].P_post
    
    assert np.trace(P_after_gap) > np.trace(P_before), \
        "Covariance should increase during missing measurement period"
    print("✓ test_missing_data PASSED (covariance grows during gap)")


def run_all_tests():
    print("="*60)
    print("KALMAN FILTER SYSTEM — TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_kf_linear_consistency,
        test_covariance_psd,
        test_rts_smoother_improvement,
        test_jacobian_numerical_match,
        test_observability,
        test_outlier_rejection,
        test_missing_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    print("="*60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
