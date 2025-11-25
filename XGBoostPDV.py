import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostPDV:
    
    def __init__(self, lookback=500, random_state=42):
        self.lookback = lookback
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def exp_kernel(self, lam, tau):
        '''Exponential kernel: K(τ) = λ·exp(-λτ/252)'''
        K = lam * np.exp(-lam * tau / 252.0)
        return K / K.sum()
    
    def compute_returns(self, prices):
        '''r_t = (S_t - S_{t-1}) / S_{t-1}'''
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return np.concatenate([[0], returns])
    
    def find_features(self, prices):
        '''Compute features for XGBoost model'''
        returns = self.compute_returns(prices)
        n = len(returns)
        lags = np.arange(self.lookback)
        
        features = pd.DataFrame()
        
        # Multi-scale R1, R2, Sigma at different timescales
        for lam in [50, 20, 10, 5]:
            K = self.exp_kernel(lam, lags)
            R1_l, R2_l = np.zeros(n), np.zeros(n)
            for t in range(self.lookback, n):
                past = returns[t-self.lookback:t][::-1]
                R1_l[t] = np.sum(K * past)
                R2_l[t] = np.sum(K * past ** 2)
            features[f'R1_l{lam}'] = R1_l
            features[f'R2_l{lam}'] = R2_l
            features[f'Sigma_l{lam}'] = np.sqrt(np.maximum(R2_l, 1e-10))
        
        return features
    
    def fit(self, prices, volatility):
        '''Fit XGBoost model'''
        features = self.find_features(prices)
        self.feature_names = list(features.columns)
        
        valid = features['R1_l50'] != 0
        X = features.loc[valid].values
        y = np.array(volatility)[valid]
        
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        split = int(len(X) * 0.8)
        dtrain = xgb.DMatrix(X[:split], label=y[:split], feature_names=self.feature_names)
        dval = xgb.DMatrix(X[split:], label=y[split:], feature_names=self.feature_names)
        
        params = {
            'max_depth': 5,
            'eta': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 20,
            'alpha': 1.0,
            'lambda': 5.0,
            'seed': self.random_state,
            'objective': 'reg:squarederror'
        }
        
        self.model = xgb.train(params, dtrain, 500, evals=[(dval, 'val')],
                               early_stopping_rounds=50, verbose_eval=False)
        
        imp = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.Series(
            {f: imp.get(f, 0) for f in self.feature_names}
        ).sort_values(ascending=False)
        
        return self
    
    def predict(self, prices):
        '''Predict volatility'''
        features = self.find_features(prices)
        valid = features['R1_l50'] != 0
        X = features.loc[valid].values
        
        predictions = np.zeros(len(prices))
        mask = ~np.isnan(X).any(axis=1)
        if mask.sum() > 0:
            dtest = xgb.DMatrix(X[mask], feature_names=self.feature_names)
            predictions[np.where(valid)[0][mask]] = self.model.predict(dtest)
        
        return predictions