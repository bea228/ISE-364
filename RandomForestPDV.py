import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

class RandomForestPDV:
    
    def __init__( self, n_estimators=100,max_depth=None):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model= None
        self.feature_importance = None
        self.scaler = StandardScaler()

    def exp_kernel(self, lam, tau):
        '''Exponential kernel: K(τ) = λ·exp(-λτ/365)'''
        K = lam * np.exp(-lam * tau / 365.0)#365 as foreign exchange is 24/7
        return K / K.sum()
    
    
    def compute_returns(self, prices:list):
        '''
        In the paper returns are defined as  r_t = (S_t - S_{t-1}) / S_{t-1}
        '''
        prices = np.array(prices)
        returns =np.diff(prices) / prices[:-1]
        returns = returns*100
        return np.concatenate([[0], returns])
    
    def find_features(self, prices):
        '''Compute features for XGBoost model'''
        returns = self.compute_returns(prices)
        n = len(returns)
        lags = np.arange(500)
        
        features = pd.DataFrame()
        
        # Multi-scale R1, R2, Sigma at different timescales
        K = self.exp_kernel(5, lags)#using a lambda of 5
        R1_l, R2_l = np.zeros(n), np.zeros(n)
        for t in range(500, n):
            past = returns[t-500:t][::-1]
            R1_l[t] = np.sum(K * past)
            R2_l[t] = np.sum(K * past ** 2)
        features[f'R1_l{5}'] = R1_l
        features[f'R2_l{5}'] = R2_l
        features[f'Sigma_l{5}'] = np.sqrt(np.maximum(R2_l, 1e-10))

        features['VoV_50'] = self.compute_vol_of_vol(returns, window=50)

        features['jump_flag'] = self.compute_jump_indicator(returns, window=50)

        return features
    def compute_raw_rvol(self, returns, window=50):
        rvol = np.zeros(len(returns))
        for t in range(window, len(returns)):
            rvol[t] = np.sqrt(np.mean(returns[t-window:t]**2))
        return rvol

    def compute_vol_of_vol(self, returns, window=50):
        rvol = self.compute_raw_rvol(returns, window)
        vov = pd.Series(rvol).rolling(window).std().fillna(0).values
        return vov
    def compute_jump_indicator(self, returns, window=50):
        rolling_std = pd.Series(returns).rolling(window).std().fillna(0)
        jumps = (np.abs(returns) > 1.75 * rolling_std).astype(int).values
        return jumps
    
    
    def fit(self, prices, volatility):
        '''
        PDV model is fit with training data
        '''
        feats = self.find_features(prices)
        

        self.features = feats
        
        x = feats[['R1_l5', 'Sigma_l5','VoV_50','jump_flag']].values[:-1]
        y = np.array(volatility)[1:]
        X_scaled = self.scaler.fit_transform(x)

        
    
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,random_state=42)
        self.model.fit(X_scaled, y)
        self.feature_importance= self.model.feature_importances_
        
        
        return self
    def predict(self, prices):
        features = self.find_features(prices)
        x = features[['R1_l5', 'Sigma_l5','VoV_50','jump_flag']].values
        X_scaled = self.scaler.transform(x)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    
    
    
    
    def plot_model_performance(self, predictions, actual_volatility, 
                          model_name='Model', n_points=500):
        """
        Create a comprehensive visualization of model performance including:
        - Time series comparison
        - Predictions vs Actual scatter plot
        - Residuals over time
        
        Parameters:
        -----------
        predictions : array-like
            Model predictions for volatility
        actual_volatility : array-like
            Actual volatility values
        model_name : str
            Name of the model for plot titles (default: 'Model')
        n_points : int
            Number of points to show in time series plot (default: 500)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The complete figure with all subplots
        """
        # Convert to arrays
        predictions = np.array(predictions)
        actual_volatility = np.array(actual_volatility)
        
        # Ensure arrays are same length
        min_len = min(len(predictions), len(actual_volatility))
        predictions = predictions[:min_len]
        actual_volatility = actual_volatility[:min_len]
        
        # Calculate R² score
        r2 = r2_score(actual_volatility, predictions)
        
        # Calculate residuals
        residuals = actual_volatility - predictions
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.3)
        
        # 1. Time Series Plot
        ax1 = fig.add_subplot(gs[0])
        time_indices = np.arange(min(n_points, len(actual_volatility)))
        ax1.plot(time_indices, actual_volatility[:n_points], 
                label='Actual', color='black', linewidth=1.5, alpha=0.8)
        ax1.plot(time_indices, predictions[:n_points], 
                label=f'{model_name} Predicted', color='red', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Volatility', fontsize=11)
        ax1.set_title(f'{model_name}: Time Series (First {n_points} points)', 
                    fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Predictions vs Actual Scatter Plot
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(actual_volatility, predictions, alpha=0.5, s=20, color='blue')
        
        # Add perfect prediction line
        min_val = min(actual_volatility.min(), predictions.min())
        max_val = max(actual_volatility.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect prediction')
        
        ax2.set_xlabel('Actual Volatility', fontsize=11)
        ax2.set_ylabel('Predicted Volatility', fontsize=11)
        ax2.set_title(f'{model_name}: Predictions vs Actual\nR² = {r2:.4f}', 
                    fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', frameon=True, fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals Over Time
        ax3 = fig.add_subplot(gs[2])
        time_indices_full = np.arange(len(residuals))
        ax3.scatter(time_indices_full, residuals, alpha=0.4, s=10, color='teal')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Time Index', fontsize=11)
        ax3.set_ylabel('Residuals', fontsize=11)
        ax3.set_title('Residuals Over Time', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
