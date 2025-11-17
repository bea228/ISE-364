class PDVModel:
    
    def __init__(self,alpha1,delta1,alpha2, delta2, tau):
        self.alpha1 = alpha1
        self.delta1= delta1
        self.alpha2 = alpha2
        self.delta2= delta2
        self.tau  = tau
    
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
    