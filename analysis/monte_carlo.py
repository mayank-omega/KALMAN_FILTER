"""
Monte Carlo Analysis & Statistical Evaluation
==============================================

Statistical Tests:
------------------

1. NEES (Normalized Estimation Error Squared):
   ε_k = (x̂_k - x_k)^T P_k^{-1} (x̂_k - x_k) ~ χ²(n)
   
   Time-averaged NEES: ε̄ = (1/T) Σ ε_k ~ χ²(n·T)/(n·T) → n (as T→∞)
   
   Consistency bounds (α=0.05 significance):
   [χ²_{α/2}(n·N)/(n·N), χ²_{1-α/2}(n·N)/(n·N)]
   where N = number of Monte Carlo runs

2. NIS (Normalized Innovation Squared):
   ν_k = ỹ_k^T S_k^{-1} ỹ_k ~ χ²(m)
   
   If NIS > threshold: filter inconsistent (underconfident or overconfident)

3. RMSE:
   RMSE_pos = √(E[(px-p̂x)² + (py-p̂y)²])
   RMSE_vel = √(E[(vx-v̂x)² + (vy-v̂y)²])

Filter Comparison:
------------------
   KF:  optimal for linear; baseline
   EKF: linearization error; can diverge for strong nonlinearity  
   UKF: 3rd-order accurate; more robust than EKF
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class FilterStats:
    """Aggregate filter performance statistics"""
    rmse_pos: float
    rmse_vel: float
    mean_nees: float
    std_nees: float
    mean_nis: float
    std_nis: float
    nees_bounds: Tuple[float, float]  # 95% bounds
    nis_bounds: Tuple[float, float]
    nees_consistent: bool
    nis_consistent: bool
    computation_time: float  # seconds per run
    nees_trajectory: np.ndarray
    nis_trajectory: np.ndarray
    rmse_trajectory: np.ndarray
    log_likelihood_total: float


def chi2_bounds(n_dof: int, N: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute chi-squared bounds for NEES/NIS consistency test.
    
    For average NEES over N runs and T timesteps, each with n_dof:
    Total dof = n_dof * N
    Bounds at significance α
    
    Uses Wilson-Hilferty normal approximation to chi-squared
    """
    k = n_dof * N
    
    def chi2_quantile(p, k):
        """Wilson-Hilferty approximation to chi-squared quantile"""
        z = _normal_quantile(p)
        h = 2.0/(9*k)
        return k * (1 - h + z * np.sqrt(h))**3
    
    lower = chi2_quantile(alpha/2, k) / k
    upper = chi2_quantile(1-alpha/2, k) / k
    return lower, upper


def _normal_quantile(p: float) -> float:
    """Rational approximation to standard normal quantile (Beasley-Springer-Moro)"""
    # Coefficients
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
    else:
        r = p if y > 0 else 1-p
        r = np.log(-np.log(r))
        x = c[0]+r*(c[1]+r*(c[2]+r*(c[3]+r*(c[4]+r*(c[5]+r*(c[6]+r*(c[7]+r*c[8])))))))
        if y < 0:
            x = -x
    return x


class MonteCarloEvaluator:
    """
    Monte Carlo framework for filter performance evaluation.
    
    Runs N independent simulations, evaluates NEES, NIS, RMSE,
    and tests for filter consistency.
    """
    
    def __init__(self, n_runs: int = 10000, n_state: int = 4, n_meas: int = 2):
        self.n_runs = n_runs
        self.n_state = n_state
        self.n_meas = n_meas
    
    def run(self, filter_factory: Callable, sim_factory: Callable,
            T: int, pos_idx: List[int] = [0, 1],
            vel_idx: List[int] = [2, 3],
            outlier_threshold: Optional[float] = None,
            verbose: bool = True) -> FilterStats:
        """
        Run Monte Carlo evaluation.
        
        Args:
            filter_factory: callable(seed) -> filter instance with .run() method
            sim_factory: callable(seed) -> (x_true, measurements) 
            T: timesteps per run
            pos_idx: state indices for position
            vel_idx: state indices for velocity
            outlier_threshold: Mahalanobis gating threshold
            verbose: print progress
        
        Returns:
            FilterStats with aggregate statistics
        """
        n = self.n_state
        m = self.n_meas
        N = self.n_runs
        
        all_nees = np.zeros((N, T))
        all_nis = np.zeros((N, T))
        all_pos_err = np.zeros((N, T))
        all_vel_err = np.zeros((N, T))
        all_loglik = np.zeros(N)
        
        total_time = 0.0
        
        for run_i in range(N):
            seed = run_i * 12345 + 7
            
            x_true, measurements = sim_factory(seed)
            filt = filter_factory(seed)
            
            t0 = time.perf_counter()
            results = filt.run(measurements, outlier_threshold=outlier_threshold)
            total_time += time.perf_counter() - t0
            
            for t, res in enumerate(results):
                # State index in true trajectory (1-indexed since states[0]=x0)
                x_t = x_true[t+1]
                x_hat = res.x_post
                P_hat = res.P_post
                
                # NEES
                err = x_hat - x_t
                try:
                    P_inv = np.linalg.inv(P_hat)
                    nees = float(err @ P_inv @ err)
                except:
                    nees = n  # fallback
                all_nees[run_i, t] = nees
                
                # NIS
                all_nis[run_i, t] = res.nis
                
                # Position and velocity RMSE contributions
                pos_err = x_hat[pos_idx] - x_t[pos_idx]
                all_pos_err[run_i, t] = np.sum(pos_err**2)
                
                if len(vel_idx) > 0 and max(vel_idx) < len(x_t):
                    vel_err = x_hat[vel_idx] - x_t[vel_idx]
                    all_vel_err[run_i, t] = np.sum(vel_err**2)
                
                all_loglik[run_i] += res.log_likelihood
            
            if verbose and (run_i + 1) % max(1, N//10) == 0:
                print(f"  MC run {run_i+1}/{N} | "
                      f"NEES={np.mean(all_nees[:run_i+1]):.2f} | "
                      f"RMSE_pos={np.sqrt(np.mean(all_pos_err[:run_i+1])):.2f}m")
        
        # Aggregate statistics
        # Time-averaged NEES across runs
        nees_traj = np.mean(all_nees, axis=0)  # (T,) - mean over runs
        nis_traj = np.mean(all_nis, axis=0)
        rmse_traj = np.sqrt(np.mean(all_pos_err, axis=0))
        
        mean_nees = np.mean(all_nees)
        std_nees = np.std(np.mean(all_nees, axis=1))  # std over runs
        mean_nis = np.mean(all_nis)
        std_nis = np.std(np.mean(all_nis, axis=1))
        
        # Chi-squared bounds
        nees_bounds = chi2_bounds(n, N)
        nis_bounds = chi2_bounds(m, N)
        
        nees_consistent = nees_bounds[0] <= mean_nees <= nees_bounds[1]
        nis_consistent = nis_bounds[0] <= mean_nis <= nis_bounds[1]
        
        rmse_pos = np.sqrt(np.mean(all_pos_err))
        rmse_vel = np.sqrt(np.mean(all_vel_err))
        
        return FilterStats(
            rmse_pos=rmse_pos,
            rmse_vel=rmse_vel,
            mean_nees=mean_nees,
            std_nees=std_nees,
            mean_nis=mean_nis,
            std_nis=std_nis,
            nees_bounds=nees_bounds,
            nis_bounds=nis_bounds,
            nees_consistent=nees_consistent,
            nis_consistent=nis_consistent,
            computation_time=total_time / N,
            nees_trajectory=nees_traj,
            nis_trajectory=nis_traj,
            rmse_trajectory=rmse_traj,
            log_likelihood_total=np.mean(all_loglik)
        )
    
    def compare_filters(self, filter_configs: Dict[str, Dict],
                        sim_factory: Callable, T: int,
                        pos_idx=[0,1], vel_idx=[2,3]) -> Dict[str, FilterStats]:
        """
        Compare multiple filters on identical Monte Carlo simulations.
        
        Args:
            filter_configs: dict of {name: {'factory': callable, ...}}
        
        Returns:
            dict of {name: FilterStats}
        """
        results = {}
        for name, config in filter_configs.items():
            print(f"\n{'='*50}")
            print(f"Evaluating: {name}")
            print(f"{'='*50}")
            stats = self.run(
                filter_factory=config['factory'],
                sim_factory=sim_factory,
                T=T,
                pos_idx=pos_idx,
                vel_idx=vel_idx,
                outlier_threshold=config.get('outlier_threshold')
            )
            results[name] = stats
            self._print_stats(name, stats)
        
        return results
    
    def _print_stats(self, name: str, stats: FilterStats):
        print(f"\n--- {name} Results ---")
        print(f"  RMSE Position:  {stats.rmse_pos:.4f} m")
        print(f"  RMSE Velocity:  {stats.rmse_vel:.4f} m/s")
        print(f"  Mean NEES:      {stats.mean_nees:.4f} ± {stats.std_nees:.4f}")
        print(f"  NEES Bounds:    [{stats.nees_bounds[0]:.4f}, {stats.nees_bounds[1]:.4f}]")
        print(f"  NEES Consistent: {'✓ YES' if stats.nees_consistent else '✗ NO'}")
        print(f"  Mean NIS:       {stats.mean_nis:.4f}")
        print(f"  NIS Consistent:  {'✓ YES' if stats.nis_consistent else '✗ NO'}")
        print(f"  Avg Comp Time:  {stats.computation_time*1000:.3f} ms/run")
        print(f"  Log-Likelihood: {stats.log_likelihood_total:.2f}")
    
    def error_distribution_analysis(self, filter_factory: Callable,
                                    sim_factory: Callable, T: int,
                                    n_runs: int = 1000) -> Dict:
        """
        Analyze the distribution of estimation errors.
        Test for normality, autocorrelation of innovations.
        """
        errors_pos = []
        innovations = []
        
        for i in range(n_runs):
            x_true, measurements = sim_factory(i * 99)
            filt = filter_factory(i * 99)
            results = filt.run(measurements)
            
            for t, res in enumerate(results):
                err = res.x_post[:2] - x_true[t+1, :2]
                errors_pos.append(err)
                innovations.append(res.innovation)
        
        errors_pos = np.array(errors_pos)
        innovations = np.array(innovations)
        
        # Kolmogorov-Smirnov test for normality (manual implementation)
        def ks_test(data):
            """KS test against standard normal"""
            sorted_data = np.sort((data - np.mean(data)) / (np.std(data) + 1e-10))
            n = len(sorted_data)
            ecdf = np.arange(1, n+1) / n
            cdf_vals = np.array([_normal_cdf(x) for x in sorted_data])
            D = np.max(np.abs(ecdf - cdf_vals))
            return D, D < 1.36 / np.sqrt(n)  # critical value at α=0.05
        
        ks_stat_x, normal_x = ks_test(errors_pos[:, 0])
        ks_stat_y, normal_y = ks_test(errors_pos[:, 1])
        
        # Innovation autocorrelation (lag 1)
        innov_x = innovations[:, 0]
        innov_ac = np.corrcoef(innov_x[:-1], innov_x[1:])[0, 1]
        
        return {
            'error_mean': errors_pos.mean(axis=0),
            'error_std': errors_pos.std(axis=0),
            'ks_statistic': [ks_stat_x, ks_stat_y],
            'errors_normal': [normal_x, normal_y],
            'innovation_autocorr': innov_ac,
            'innovation_mean': innovations.mean(axis=0),
            'innovation_std': innovations.std(axis=0),
            'raw_errors': errors_pos[:1000]  # sample for plotting
        }


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function"""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
