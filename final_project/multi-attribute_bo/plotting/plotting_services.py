# Copyright (c) 2016, Raul Astudillo

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from matplotlib import interactive
from pylab import savefig
import pylab


def plot(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in 1d
    if input_dim ==1:
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)


        model.plot_density(bounds[0], alpha=.5)

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')


        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim ==2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'k.', markersize=10)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        if filename!=None:
            savefig(filename)
        else:
            plt.show()
            
def integrated_plot(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, attribute=0, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''
    # Plots in 1d
    if input_dim ==1:
        
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.01)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        mean, var = model.predict(x_grid)
        output_dim = len(Ydata)
        
        for j in range(output_dim):
            m = mean[j,:]
            v = var[j,:]
            plt.figure(j)
            plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
            plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
            plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)
    
            plt.plot(Xdata, Ydata[j], 'r.', markersize=10)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))
    
            plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            plt.legend(loc='upper left')
            plt.show()

        if filename!=None:
            savefig(filename)
            input()
            
            
def integrated_plot2(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, attribute=0, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.01)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        mean, var = model.predict(x_grid)
        
        m = mean[attribute,:]
        v = var[attribute,:]

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata[attribute], 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')
            
        if filename!=None:
            savefig(filename)
        #else:
            plt.show()

def plot_acquisition(bounds, input_dim, acquisition_function, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        plt.plot(x_grid, acqu_normalized, 'r-',lw=2,label ='Acquisition function')
        plt.xlabel('x')
        plt.ylabel('a(x)')
        plt.legend(loc='upper left')
        if filename!=None:
            savefig(filename)
        else:
            plt.show()


def plot_convergence(historic_optimal_values, filename = None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = len(historic_optimal_values)
    plt.plot(list(range(n)), historic_optimal_values,'-o')
    plt.title('Expected value of historical optimal points given the true attributes')
    plt.xlabel('Iteration')
    plt.ylabel('value')
    grid(True)

    if filename!=None:
        savefig(filename)
    else:
        plt.show()
