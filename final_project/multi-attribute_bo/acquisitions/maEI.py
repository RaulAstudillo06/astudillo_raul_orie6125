# Copyright (c) 2018, Raul Astudillo

import numpy as np
from ...GPyOpt.acquisitions.base import AcquisitionBase
from ...GPyOpt.core.task.cost import constant_cost_withGradients
from scipy.stats import norm

class maEI(AcquisitionBase):
    """
    Multi-attribute expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param cost_withGradients: function
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        super(maEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients


    def _compute_acq(self, X):
        """
        Computes the multi-attibute expected improvement at points X.
        """
        full_support = True
        X =np.atleast_2d(X)
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
         
        marginal_acqX = self._marginal_acq(X, utility_params_samples)           
        if full_support:
             acqX = np.matmul(marginal_acqX, utility_dist)
        acqX = np.reshape(acqX, (X.shape[0],1))
        return acqX


    def _compute_acq_withGradients(self, X):
        """
        Computes the multi-attibute expected improvement and its gradient at points X.
        """ 
        full_support = True # If true, the utility's distribution will be integrated when computing the value of information; otherwise
                            # a Monte Carlo estimate will be used
        X =np.atleast_2d(X)
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
         
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_params_samples)           
        if full_support:
             acqX = np.matmul(marginal_acqX, utility_dist)
             dacq_dX = np.tensordot(marginal_dacq_dX, utility_dist, 1)
        acqX = np.reshape(acqX,(X.shape[0], 1))
        return acqX, dacq_dX
    
    
    def _marginal_acq(self, X, utility_params_samples):
        """
        Computes the expected improvement for a set of realizations of the utility's function distribution (given by utility_params_samples) at points X.
        """
        L = len(utility_params_samples)
        marginal_acqX = np.zeros((X.shape[0],L))
        n_h = 5 # Number of GP hyperparameters samples.
        gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)
        for h in gp_hyperparameters_samples:
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            marginal_best_so_far = self._marginal_best_so_far(utility_params_samples)
            for l in range(L):
                current_best = marginal_best_so_far[l]
                for i in range(X.shape[0]):
                    mu = np.dot(utility_params_samples[l], meanX[:,i])
                    sigma = np.sqrt(np.dot(np.square(utility_params_samples[l]), varX[:,i]))
                    marginal_acqX[i,l] += (mu-current_best)*norm.cdf(mu, loc=current_best, scale=sigma) + sigma*norm.pdf(mu, loc=current_best, scale=sigma)
                    
        marginal_acqX = marginal_acqX/n_h
        return marginal_acqX
    
    
    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        Computes the expected improvement for a set of realizations of the utility function (given by utility_params_samples) at points X.
        """ 
        L = len(utility_params_samples)
        marginal_acqX = np.zeros((X.shape[0],L))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], L))
        n_h = 5 # Number of GP hyperparameters samples.
        gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)
        for h in gp_hyperparameters_samples:
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            dmean_dX = self.model.posterior_mean_gradient(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            marginal_best_so_far = self._marginal_best_so_far(utility_params_samples)
            for l in range(L):
                best = marginal_best_so_far[l]
                for i in range(X.shape[0]):
                    mu = np.dot(utility_params_samples[l], meanX[:,i])
                    sigma = np.sqrt(np.dot(np.square(utility_params_samples[l]), varX[:,i]))
                    phi = norm.pdf((mu-best)/sigma)
                    Phi = norm.cdf((mu-best)/sigma)
                    marginal_acqX[i,l] += (mu-best)*Phi + sigma*phi
                    dmu_dX = np.matmul(utility_params_samples[l], dmean_dX[:,i,:])
                    dsigma_dX = np.matmul(np.square(utility_params_samples[l]), dvar_dX[:,i,:])/sigma
                    marginal_dacq_dX[i, :, l] += dmu_dX*Phi + phi*dsigma_dX 
                    
        marginal_acqX = marginal_acqX/n_h          
        marginal_dacq_dX = marginal_dacq_dX/n_h
        return marginal_acqX, marginal_dacq_dX
    
    
    def _marginal_best_so_far(self, utility_params_samples):
        """
        Compues the current optimal value for each of the realizations of the utility function (given by utility_params_samples).
        """ 
        L = len(utility_params_samples)
        marginal_best = np.empty(L)
        muX_eval = self.model.posterior_mean_at_evaluated_points()
        for l in range(L):
            marginal_best[l] = min(np.matmul(utility_params_samples[l], muX_eval))
            
        return marginal_best
                    
            
    def _get_utility_parameters_samples(self, n_samples=None):
        """
        Returns n_samples samples of the utiliy function's parameters.
        """
        if n_samples == None:
            samples = self.utility.parameter_dist.support
        else:
            samples = self.utility.parameter_dist.sample(n_samples)      
        return samples       
        
