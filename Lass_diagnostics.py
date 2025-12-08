import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class LassoDiagnostics:
    """
    Class for LASSO model diagnostics - 4 plots only
    """
    
    def __init__(self, results, lambda_grid):
        self.results = results
        self.lambda_grid = lambda_grid
    
    def plot_diagnostics(self, model_name='lasso_cv', save_path=None):
        """
        Create 4 diagnostic plots: time-series, scatter, residuals, Q-Q
        """
        y_test = self.results['data']['y_test']
        y_pred = self.results['predictions'][model_name]
        residuals = y_test - y_pred
        r2_test = self.results['metrics'][model_name]['r2_test']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Time series
        axes[0, 0].plot(y_test.index, y_test.values, label='Actual', color='blue', linewidth=1)
        axes[0, 0].plot(y_pred.index, y_pred.values, label='Predicted', color='red', linewidth=0.3)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Volatility (%)')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2_test:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter
        axes[0, 1].scatter(y_test, y_pred, alpha=0.5,color = 'blue', s=10)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        axes[0, 1].set_xlabel('Actual Volatility (%)')
        axes[0, 1].set_ylabel('Predicted Volatility (%)')
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        axes[1, 0].plot(residuals.index, residuals.values, linewidth=0, color = 'orange', marker='o', markersize=2  )
        axes[1, 0].axhline(y=0, color='blue', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Residuals (%)')
        axes[1, 0].set_title(f'Residuals (Mean={residuals.mean():.4f}, Std={residuals.std():.4f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close after saving to avoid extra display
        
        return fig


class KernelAnalysis:
    """
    Class for kernel visualization - 2 kernel plots only
    """
    
    def __init__(self, results, lambda_grid):
        self.results = results
        self.lambda_grid = lambda_grid
    
    def _kernel_R1(self, t, lambda_coef_pairs):
        result = np.zeros_like(t, dtype=float)
        for lam, coef in lambda_coef_pairs:
            result += -coef * np.exp(-lam * t)
        return result
    
    def _kernel_R2(self, t, lambda_coef_pairs):
        result = np.zeros_like(t, dtype=float)
        for lam, coef in lambda_coef_pairs:
            result += coef * np.exp(-lam * t)
        return result
    
    def get_selected_features(self, model_name='lasso_cv'):
        model = self.results['models'][model_name]
        coefficients = model.coef_
        
        n_lambdas = len(self.lambda_grid)
        coef_R1 = coefficients[:n_lambdas]
        coef_R2 = coefficients[n_lambdas:]
        
        selected_R1_idx = np.where(coef_R1 != 0)[0]
        selected_R2_idx = np.where(coef_R2 != 0)[0]
        
        r1_pairs = [(self.lambda_grid[idx], coef_R1[idx]) for idx in selected_R1_idx]
        r2_pairs = [(self.lambda_grid[idx], coef_R2[idx]) for idx in selected_R2_idx]
        
        return r1_pairs, r2_pairs
    
    def plot_kernels(self, model_name='lasso_cv', t_max=100, save_path=None):
        """
        Create 2 kernel plots: R1 and R2
        """
        r1_pairs, r2_pairs = self.get_selected_features(model_name)
        
        t_1 = np.linspace(0, t_max*10, 1000)
        t_2 = np.linspace(0, t_max, 1000)

        K_R1 = self._kernel_R1(t_1, r1_pairs)
        K_R2 = self._kernel_R2(t_2, r2_pairs)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: R1 Kernel
        axes[0].plot(t_1, K_R1, linewidth=1.5, color='blue')
        axes[0].set_xlabel('Time (τ units, 1τ = 2 hours)')
        axes[0].set_ylabel('K_R1(τ)')
        axes[0].set_title('R1 Kernel: Linear Returns')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Plot 2: R2 Kernel
        axes[1].plot(t_2, K_R2, linewidth=1.5, color='red')
        axes[1].set_xlabel('Time (τ units, 1τ = 2 hours)')
        axes[1].set_ylabel('K_R2(τ)')
        axes[1].set_title('R2 Kernel: Squared Returns')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close after saving to avoid extra display
        
        return fig