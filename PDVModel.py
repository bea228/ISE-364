import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
class PDVModel:
    
    def __init__(self,alpha1,delta1,alpha2, delta2):
        self.alpha1 = alpha1
        self.delta1= delta1
        self.alpha2 = alpha2
        self.delta2= delta2
    
    def kernel(self,alpha,delta,tau):
        '''
         K(tau) = Z_alpha,delta^(-1) * (tau + delta)^(-alpha), with constraints: tau >= 0, alpha > 1, delta > 0
        z alpha delta is the normalization constant which is delta^(1-alpha) / (alpha-1)
        alpha is power-law exponent(determines memory structure so large a mean quick decay)
        tau is time elapsed since past event
        delta is the time shift which helps the kernel from blowing up if tau=0
        
        '''
        z_inverse = ((delta**(1-alpha)) / (alpha-1))**-1
        kernel_output  = z_inverse * ((tau+delta)**(-alpha))
        return kernel_output
    def compute_returns(self, prices:list):
        '''
        In the paper returns are defined as  r_t = (S_t - S_{t-1}) / S_{t-1}
        '''
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return np.concatenate([[0], returns])
    
    def get_r1(self,returns):
        '''
        returns assumed to be numpy 
        in paper R1_t = Σ K1(t - t_i) * r_{t_i}
        '''
        r1= np.zeros(len(returns))
        for t in range(len(returns)):
            if t==0:
                r1[t]= 0.0
                continue
            past_return = returns[:t]
            taus = t- np.arange(0,t)
            kernels = self.kernel(self.alpha1,self.delta1,taus)
            r1[t]= np.sum(kernels*past_return)
        return r1
    def get_r2(self,returns):
        '''
        returns assumed to be numpy 
        R2_t = Σ K2(t - t_i) * r_{t_i}^2
        '''
        r2= np.zeros(len(returns))
        for t in range(len(returns)):
            if t==0:
                r2[t]= 0.0
                continue
            past_return = returns[:t]
            taus = t- np.arange(0,t)
            kernels = self.kernel(self.alpha2,self.delta2,taus)
            r2[t]= np.sum(kernels* past_return**2)
        return r2
    
    
    def get_sigma_t(self,returns):
        return np.sqrt(self.get_r2(returns))
    
    
    def find_features(self,prices):
        #find r1 and Σ_t then put it in df
        returns = self.compute_returns(prices)
        
        features= pd.DataFrame({
            'R1': self.get_r1(returns),
            'Sigma_t': self.get_sigma_t(returns),
            'returns': returns
        })
        return features
    def fit(self, prices, volatility):
        '''
        PDV model is fit with training data
        '''
        feats = self.find_features(prices)
        x = feats[['R1','Sigma_t']].values
        y = np.array(volatility)
        
        regression = LinearRegression(fit_intercept=True)
        regression.fit(x,y)
        self.beta0 = regression.intercept_
        self.beta1= regression.coef_[0]
        self.beta2= regression.coef_[1]
        return self
    def predict(self, prices):
        features = self.find_features(prices)
        predictions = (self.beta0 + self.beta1 * features['R1'] + self.beta2 * features['Sigma_t'])
        return predictions.values