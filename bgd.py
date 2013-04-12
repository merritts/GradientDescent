import numpy as np
from array import array

def lr_bgd(a, x, y, ep=0.001, max_iter=10000, debug=True):
    #if we are debugging, we'll collect the error at each iteration
    if debug: err = array('f',[])
    
    ### Initialize the algorithm state ###
    cvg = False
    m = 0
    n = x.shape[0]
    
    #intialize the parameter/weight vector
    t = np.zeros(x.shape[1])
    
    #for each training example, compute the gradient
    z = [(y[i]-np.dot(t,x[i]))*x[i] for i in range(n)]

    #update the parameters
    t = t + a * sum(z)
    
    #compute the total error
    pe = sum([(y[i]-np.dot(t,x[i]))**2 for i in range(n)])
    
    ### Iterate until convergence or max iterations ###
    while not cvg:
        z = [(y[i]-np.dot(t,x[i]))*x[i] for i in range(n)]
        t = t + a*sum(z)
        e = sum([(y[i]-np.dot(t,x[i]))**2 for i in range(n)])
        if debug: err.append(e)
        if abs(pe-e) <= ep:
            print '*** Converged, iterations: ', m, ' ***'
            cvg = True
        pe = e
        m+=1
        if m == max_iter:
            print '*** Max interactions exceeded ***'
            break
    if debug:
        return (t,err)
    else:
        return (t)