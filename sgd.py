from math import e
from array import array

import numpy as np
import matplotlib.pyplot as plt

def h(x,theta):
    """
    logistic regression hypothesis function
    """
    return e**(np.dot(x,theta))/(1.+e**(np.dot(x,theta)))


def log_reg_sgd(x,y,a,max_iter=100,debug=True):
    """
    Stochastic gradient descent for two class (0,1) logistic regression
    with static learning rate
    """
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
    theta = np.random.random(n)
    z = np.arange(m)
    for t in xrange(max_iter):
        #shuffle indices prior to searching
        z = np.random.permutation(z)
        #for each training example
        for i in z:
            #update weights
            theta = theta + a*(y[i]-h(x[i],theta))*x[i]
            #compute the error and test for convergence
            if debug:err.append(sum([(y[i]-h(x[i],theta))**2 for i in range(m)]))
    if debug: return theta,err
    return theta


def log_reg_regularized_sgd(x,y,a,l=0.1,max_iter=100,debug=True):
    """
    Stochastic gradient descent for logistic regression with regularization
    and static learning rate
    """
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
    theta = np.random.random(n)
    z = np.arange(m)
    for t in xrange(max_iter):
        #shuffle indices prior to searching
        z = np.random.permutation(z)
        #for each training example
        for i in z:
            #update weights
            theta = theta + a*(y[i]-h(x[i],theta))*x[i] - l*2.*a*theta
            #compute the error and test for convergence
            if debug:err.append(sum([(y[i]-h(x[i],theta))**2 for i in range(m)]))
    if debug: return theta,err
    return theta