from math import e
from array import array

import numpy as np
import matplotlib.pyplot as plt

def h(x,theta):
    """
    logistic regression hypothesis function
    """
    return e**(np.dot(x,theta))/(1.+e**(np.dot(x,theta)))

def log_reg_sgd(x,y,a,max_iter=1000,debug=True):
    """
    Stochastic gradient descent for logistic regression, static learning rate
    """
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
    theta = np.random.random(n)
    z = np.arange(m)
    for t in xrange(max_iter):
        print t
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


def log_reg_regularized_sgd(x,y,a,l=0.1,ep=0.00001,max_iter=100000,debug=True):
    """
    Stochastic gradient descent for logistic regression with regularization
    """
    iters=0
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
    p = 1.-a*l/m
    cvg = False
    #theta = np.random.random(n)
    theta = np.zeros(n)
    z = np.arange(m)
    pe = sum([(y[i]-h(x[i],theta))**2 for i in range(m)])
    ### update rule ###
    while not cvg:
        #shuffle indices prior to searching
        z = np.random.permutation(z)
        for i in z:
            #update weights
            theta = theta*p - a*(h(x[i],theta)-y[i])*x[i]
            #compute the error and test for convergence
            e = sum([(y[i]-h(x[i],theta))**2 for i in range(m)])
            if debug: err.append(e)
            if abs(pe-e) <= ep:
                cvg = True
                break
            pe = e
        iters+=1
        if iters == max_iter:
            print '*** Max interactions exceeded ***'
            break
    if debug:
        return theta,err
    return theta