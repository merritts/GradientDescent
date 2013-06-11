from math import e
from array import array

import numpy as np


def h(x,theta):
    return 1./(1.+e**(np.dot(-x,theta)))

def log_reg_sgd(x,y,a,ep=0.0000001,max_iter=100000,debug=True):
    """
    Stochastic gradient descent for logistic regression
    """
    iters=0
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
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
            theta = theta + a*(y[i]-h(x[i],theta))*x[i]
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


def log_reg_regularized_sgd(x,y,a,ep=0.0000001,max_iter=100000,debug=True):
    iters=0
    if debug: err = array('f',[])
    ### initialize algorithm state ###
    m,n = x.shape
    cvg = False
    theta = np.random.random(n)
    z = np.arange(m)
    pe = sum([(y[i]-h(x[i],theta))**2 for i in range(m)])
    ### update rule ###
    while not cvg:
        #shuffle indices prior to searching
        z = np.random.permutation(z)
        for i in z:
            #update weights
            theta = theta + a*(y[i]-h(x[i],theta))*x[i]+l/m*theta
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