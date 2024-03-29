#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import norm,multivariate_normal
from copy import copy

from artgpn.node import Linear as nodeL
from artgpn.node import Polynomial as nodeP
from artgpn.weight import Linear as weightL
from artgpn.weight import Polynomial as weightP

class network(object):
    """ 
    Class to create our Artifical Gaussian Process Network.
    
    Parameters
    ----------
    num_nodes: int
        Number of node functions used
    time: array
        Time measurements
    *args: arrays
        The data (or components), it needs be given in order of data1, 
        data1_error, data2, data2_error, etc...
    """ 
    def  __init__(self, num_nodes, time, *args):
        #number of nodes being used
        self.q = num_nodes
        #time
        self.time = time 
        #the data, it should be given as data1, data1_error, data2, ...
        self.args = args 
        #number of components of y(x)
        self.p = int(len(self.args)/2)
        #total number of weights we will have
        self.qp =  self.q * self.p
        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time
        self.y = [] 
        self.yerr = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                self.y.append(j)
            else:
                self.yerr.append(j)
        self.y = np.array(self.y) #"extended" measurements
        self.yerr = np.array(self.yerr) #"extended" errors
        #check if the input was correct
        assert int((i+1)/2) == self.p, \
        'Given data and number of components dont match'
        
    def _kernel_matrix(self, kernel, time = None):
        """ Returns the covariance matrix created """
        #if time is None we use the initial time of our complexGP
        r = time[:, None] - time[None, :] if time.any() else self.time[:, None] - self.time[None, :]
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r)
        return K
    
    def _predict_kernel_matrix(self, kernel, tstar):
        """ To be used in predict_gp() """
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, tstar[:, None], self.time[None, :])
        else:
            r = tstar[:, None] - self.time[None, :]
            K = kernel(r)
        return K
    
    def _kernel_pars(self, kernel):
        """ Returns a given kernel hyperparameters """
        return kernel.pars
    
##### mean functions
    @property
    def mean_pars_size(self):
        return self._mean_pars_size
    
    @mean_pars_size.getter
    def mean_pars_size(self):
        self._mean_pars_size = 0
        for m in self.means:
            if m is None: self._mean_pars_size += 0
            else: self._mean_pars_size += m._parsize
        return self._mean_pars_size
    
    @property
    def mean_pars(self):
        return self._mean_pars
    
    @mean_pars.setter
    def mean_pars(self, pars):
        pars = list(pars)
        assert len(pars) == self.mean_pars_size
        self._mean_pars = copy(pars)
        for _, m in enumerate(self.means):
            if m is None: 
                continue
            j = 0
            for j in range(m._parsize):
                m.pars[j] = pars.pop(0)
                
    def _mean(self, means, time=None):
        """ Returns the values of the mean functions """
        if time is None:
            N = self.time.size
            m = np.zeros_like(self.tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    #print(meanfun(self.time))
                    m[i*N : (i+1)*N] = meanfun(self.time)
        else:
            N = time.size
            ttt = np.tile(time, self.p)
            m = np.zeros_like(ttt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(time)
        return m
    
    
    def sample(self, kernel, time):
        """
        Returns samples from the kernel
        
        Parameters
        ----------
        kernel: func
            Covariance function
        time: array
            Time array

        Returns
        -------
        norm: array
            Sample of K 
        """
        mean = np.zeros_like(time)
        norm = multivariate_normal(mean, kernel, allow_singular=True).rvs()
        return norm

    
##### marginal likelihood functions
    def _covariance_matrix(self, nodes, weights, time, 
                           position_p, add_errors = False):
        """ 
        Creates the smaller matrices that will be used in a big final matrix
        
        Parameters
        ----------
        node: array
            The node functions
        weight: array
            The weight funtion
        time: array
            Time of the measurements 
        position_p: int
            Position necessary to use the correct node and weight
        add_errors: bool
            To add the errors in the diagonal of the matrix
            
        Returns
        -------
        k_ii: array
            Block matrix in position ii
        """
        #measurement errors
        yy_err = np.concatenate(self.yerr)
        new_yyerr = np.array_split(yy_err, self.p)
        
        #block matrix starts empty
        k_ii = np.zeros((time.size, time.size))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            weightPars = self._kernel_pars(weights[i-1 + self.q*(position_p-1)])
            #node and weight functions kernel
            w = self._kernel_matrix(type(weights[i-1 + self.q*(position_p-1)])(*weightPars), time)
            f_hat = self._kernel_matrix(type(nodes[i - 1])(*nodePars),time)
            #now we add all the necessary stuff
            k_ii += (w * f_hat)
        #adding measurement errors to our covariance matrix
        if add_errors:
            k_ii +=  (new_yyerr[position_p - 1]**2) * np.identity(time.size)
        return k_ii
    
    def compute_matrix(self, nodes, weights, time, 
                       nugget = False, shift = False):
        """
        Creates the big covariance matrix K that will be used in the log 
        marginal likelihood calculation
        Parameters
        ----------
        node: array
            The node functions
        weight: array
            The weight funtion
        time: array
            Time of the measurements 
        nugget: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True if K is not positive definite, False otherwise
        
        Returns
        -------
        K: array
            Final covariance matrix 
        """
        #columns and lines size of the "final matrix"
        K_size = self.time.size * self.p
        #initially our "final matrix" K is empty
        K_start = np.zeros((K_size, K_size))
        #now we calculate the block matrices to be added to K
        for i in range(1, self.p+1):
            k = self._covariance_matrix(nodes, weights, self.time,
                                        position_p = i, add_errors = False)
            K_start[(i-1)*self.time.size : (i)*self.time.size, 
                        (i-1)*self.time.size : (i)*self.time.size] = k
        #addition of the measurement errors
        diag = np.concatenate(self.yerr) * np.identity(self.time.size * self.p)
        K = K_start + diag
        #more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
        if shift:
            shift = 0.01
            K = K + shift*np.identity(self.time.size * self.p)
        return K

    def log_likelihood(self, nodes, weights, means, jitters):
        """ 
        Calculates the marginal log likelihood of our network. See Rasmussen 
        & Williams (2006), page 113.
        
        Parameters
        ----------
        nodes: array
            The node functions
        weights: array
            The weight funtion
        means: array
            Mean function being used
        jitters: array
            Jitter value of each dataset
            
        Returns
        -------
        log_like: float
            Marginal log likelihood
        """
        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        new_y = np.array_split(yy, self.p)
        yy_err = np.concatenate(self.yerr)
        new_yyerr = np.array_split(yy_err, self.p)
        
        log_like = 0 #"initial" likelihood starts at zero to then add things
        #calculation of each log-likelihood
        for i in range(1, self.p+1):
            k_ii = np.zeros((self.time.size, self.time.size))
            for j in range(1,self.q + 1):
                nodePars = self._kernel_pars(nodes[j - 1])
                weightPars = self._kernel_pars(weights[j-1 + self.q*(i-1)])
                #node and weight functions kernel
                w = self._kernel_matrix(type(weights[j-1 + self.q*(i-1)])(*weightPars), self.time)
                f_hat = self._kernel_matrix(type(nodes[j - 1])(*nodePars), self.time)
                #now we add all the necessary stuff
                k_ii = k_ii + (w * f_hat)
            k_ii += (new_yyerr[i - 1]**2) * np.identity(self.time.size) \
                    + (jitters[i - 1]**2) * np.identity(self.time.size)
            #log marginal likelihood calculation
            try:
                L1 = cho_factor(k_ii, overwrite_a=True, lower=False)
                log_like += - 0.5*np.dot(new_y[i - 1].T, cho_solve(L1, new_y[i - 1])) \
                           - np.sum(np.log(np.diag(L1[0]))) \
                           - 0.5*new_y[i - 1].size*np.log(2*np.pi)
            except LinAlgError:
                return -np.inf
        return log_like
    
    def marginalLikelihood_2ys(self, nodes, weights, means, jitters, 
                               N=1000 , file = "saved_results.txt"):
        f = open(file, "a")
        m = self._mean(means)
        m = np.array(np.array_split(m, 2))
        err = np.array([np.sqrt(jitters[0]**2 + self.yerr[0]**2), 
                        np.sqrt(jitters[1]**2 + self.yerr[1]**2)])
        kernel = []
        for i in range(1, self.p+1):
            for j in range(1,self.q + 1):
                nodePars = self._kernel_pars(nodes[j - 1])
                weightPars = self._kernel_pars(weights[j-1 + self.q*(i-1)])
                #node and weight functions kernel
                w = self._kernel_matrix(type(weights[j-1 + self.q*(i-1)])(*weightPars), self.time)
                f_hat = self._kernel_matrix(type(nodes[j - 1])(*nodePars), self.time)
                #now we add all the necessary stuff
                k_ii = (w * f_hat)
            kernel.append(k_ii)
        n = 0
        llhoods = 0
        while n<N:
            sample1 = self.sample(kernel[0], self.time) + m[0]
            sample2 = self.sample(kernel[1], self.time) + m[1]
            normpdf1 = norm(loc=sample1, scale=err[0]).pdf(self.y[0])
            normpdf2 = norm(loc=sample2, scale=err[1]).pdf(self.y[1])
            llhood = np.array([normpdf1.prod(),normpdf2.prod()])
            llhood = llhood.prod()
            llhoods += llhood
            sigmaN = 0
            print(n, np.log(llhoods/n), sigmaN/np.sqrt(n))
            n += 1
            if n%500 ==0:
                 print(n, np.log(llhoods/n), sigmaN/np.sqrt(n), file=f)
        f.close()
        return np.log(llhood/N)
    
##### GP prediction functions
    def prediction(self, nodes = None, weights = None,
                   means = None, jitters= None, time = None, dataset = 1):
        """ 
        Conditional predictive distribution of the Gaussian process
        
        Parameters
        ----------
        nodes: array
            The node functions
        weights: array
            The weight funtion
        means: array
            Mean function being used
        jitters: array
            Jitter value of each dataset
            dataset = 1,2,3,... accordingly to the data we are using, 
                    1 represents the first y(x), 2 the second y(x), etc...
        time: array
            Time of the predictions
        Returns
        -------
        y_mean: array
            Mean vector
        y_std: array
            Standard deviation vector
        time: array
            Time of the predictions
        """
        print('Working with dataset {0}'.format(dataset))
        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        #Time
        time = time if time.any() else self.time
        new_y = np.array_split(yy, self.p)
        yy_err = np.concatenate(self.yerr)
        new_yerr = np.array_split(yy_err, self.p)
        #cov = k + diag(error) + diag(jitter)
        cov = self._covariance_matrix(nodes, weights, 
                                      self.time, dataset, add_errors = False)
        cov += (new_yerr[dataset - 1]**2) * np.identity(self.time.size) \
                    + (jitters[dataset - 1]**2) * np.identity(self.time.size)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, new_y[dataset - 1])
        tshape = time[:, None] - self.time[None, :]
        k_ii = np.zeros((tshape.shape[0],tshape.shape[1]))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            weightPars = self._kernel_pars(weights[i-1 + self.q*(dataset - 1)])
            #node and weight functions kernel
            w = self._predict_kernel_matrix(type(weights[i-1 + self.q*(dataset - 1)])(*weightPars), time)
            f_hat = self._predict_kernel_matrix(type(nodes[i - 1])(*nodePars), time)
            #now we add all the necessary stuff
            k_ii = k_ii + (w * f_hat)
        Kstar = k_ii
        Kstarstar = self._covariance_matrix(nodes, weights,
                                            time, dataset, add_errors = False)
        Kstarstar += (jitters[dataset - 1]**2) * np.identity(time.size)
        new_mean = np.array_split(self._mean(means, time), self.p)
        y_mean = np.dot(Kstar, sol) + new_mean[dataset-1] #mean
        kstarT_k_kstar = []
        for i, _ in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, time


##### END
