#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt

class nodeFunction(object):
    """
        Definition the node functions kernels of our network, by default and 
    because it simplifies my life, all kernels include a white noise term
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args, dtype=float)

    def __call__(self, r, t1 = None, t2=None):
        """
            r = t - t' 
            Not sure if this is a good approach since will make our life harder 
        when defining certain non-stationary kernels, e.g linear kernel.
        """
        print('call r', r.shape)
        raise NotImplementedError

    def __repr__(self):
        """
            Representation of each kernel instance
        """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))


##### Constant #################################################################
class Constant(nodeFunction):
    """
        This kernel returns its constant argument c with white noise
        Parameters:
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return self.c * np.ones_like(r)


##### White Noise ##############################################################
class WhiteNoise(nodeFunction):
    """
        Definition of the white noise kernel.
        Parameters
            wn = white noise amplitude
    """
    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)
        self.wn = wn
        self.type = 'stationary'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        if r[0,:].shape == r[:,0].shape:
            return self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        else:
            return np.zeros_like(r)


##### Squared exponential ######################################################
class SquaredExponential(nodeFunction):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            ell = length-scale
            wn = white noise
    """
    def __init__(self, ell):
        super(SquaredExponential, self).__init__(ell)
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return exp(-0.5 * r**2 / self.ell**2)


##### Periodic #################################################################
class Periodic(nodeFunction):
    """
        Definition of the periodic kernel.
        Parameters:
            ell = lenght scale
            P = period
            wn = white noise
    """
    def __init__(self, P, ell):
        super(Periodic, self).__init__(P, ell)
        self.P = P
        self.ell = ell
        self.type = 'non-stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(nodeFunction):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            ell_e = evolutionary time scale
            P = kernel periodicity
            ell_p = length scale of the periodic component
            wn = white noise
    """
    def __init__(self, ell_e, P, ell_p):
        super(QuasiPeriodic, self).__init__(ell_e, P, ell_p)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2))


##### Rational Quadratic #######################################################
class RationalQuadratic(nodeFunction):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
            wn = white noise amplitude
    """
    def __init__(self, alpha, ell):
        super(RationalQuadratic, self).__init__(alpha, ell)
        self.alpha = alpha
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return 1 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha


##### RQP kernel ###############################################################
class RQP(nodeFunction):
    """
        Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
        If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if its true.
        Parameters:
            ell_e and ell_p = aperiodic and periodic lenght scales
            alpha = alpha of the rational quadratic kernel
            P = periodic repetitions of the kernel
            wn = white noise amplitude
    """
    def __init__(self, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(alpha, ell_e, P, ell_p)
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        a = exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2)
        b = (1+ r**2/ (2*self.alpha*self.ell_e**2))#**self.alpha
        return a / (np.sign(b) * (np.abs(b)) ** self.alpha)


##### Cosine ###################################################################
class Cosine(nodeFunction):
    """
        Definition of the cosine kernel.
        Parameters:
            P = period
            wn = white noise amplitude
    """
    def __init__(self, P):
        super(Cosine, self).__init__(P)
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return cosine(2*pi*np.abs(r) / self.P)


##### Exponential ##############################################################
class Exponential(nodeFunction):
    """
        Definition of the exponential kernel.
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell):
        super(Exponential, self).__init__(ell)
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r): 
        return exp(- np.abs(r)/self.ell) 


##### Matern 3/2 ###############################################################
class Matern32(nodeFunction):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell):
        super(Matern32, self).__init__(ell)
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return (1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)


#### Matern 5/2 ################################################################
class Matern52(nodeFunction):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale  
            wn = white noise amplitude
    """
    def __init__(self, ell):
        super(Matern52, self).__init__(ell)
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)


#### Linear ####################################################################
class Linear(nodeFunction):
    """
        Definition of the Linear kernel.
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c):
        super(Linear, self).__init__(c)
        self.c = c
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r, t1, t2):
        return  (t1 - self.c) * (t2 - self.c)


##### Gamma-exponential ########################################################
class GammaExp(nodeFunction):
    """
        Definition of the gamma-exponential kernel
            gamma = shape parameter ( 0 < gamma <= 2)
            ell = lenght scale
            wn = white noise amplitude
    """
    def __init__(self, gamma, ell):
        super(GammaExp, self).__init__(gamma, ell)
        self.gamma = gamma
        self.ell = ell
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return exp( - (np.abs(r)/self.ell) ** self.gamma) 


##### Polinomial ###############################################################
class Polynomial(nodeFunction):
    """
        Definition of the polinomial kernel
            a = real value > 0
            b = real value >= 0
            c = integer value
            wn = white noise amplitude
    """
    def __init__(self, a, b, c):
        super(Polynomial, self).__init__(a, b, c)
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, r, t1, t2):
        return (self.a * t1 * t2 + self.b)**self.c 


##### END
