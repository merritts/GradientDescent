import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.preprocessing
from scipy.stats import linregress

from bgd import lr_bgd
from sgd import lr_sgd

def plot(x,y):
    #for each feature
    for i in range(x.shape[1]):
        plt.plot(x[:,i],y, 'ro')
        plt.title('feature: '+str(i))
        plt.show()

def test_sgd():
    #fit the model using sgd
    theta2 = lr_sgd(a, train, ytrain)
    print 'SGD: ', theta2
    
    vals = [(i,j) for i,j in zip(ytest, [np.dot(theta2,i) for i in test])]
    plt.plot(map(lambda i: i[1], vals), 'g^', label='SGD')
    plt.xlabel('Test example, i', fontsize=24)
    plt.ylabel('P(i)', fontsize=24)
    plt.legend(loc=0)
    plt.show()
    
      
def test_bgd():
    #learning rate
    a = 0.0001
    
    #load data
    data = sklearn.datasets.load_boston()
    
    #scale it, split into training and test sets
    features = sklearn.preprocessing.scale(data['data'][:,12])
    y = sklearn.preprocessing.scale(data['target'])
    train = features[:features.shape[0]/2]
    ytrain = y[:y.shape[0]/2]
    slope, intercept, r_value, p_value, std_err = linregress(train, ytrain)
    
    print 'Scipy linear regression ', intercept, slope
    
    features = np.array([np.append(1,i) for i in features])
    train = features[:features.shape[0]/2]
    test = features[features.shape[0]/2:]
    ytrain = y[:y.shape[0]/2]
    ytest = y[y.shape[0]/2:]
    
    #fit the model using bgd
    theta1, err = lr_bgd(a, train, ytrain)
    print 'BGD: ', theta1
    
    
    #measure the total error, print and plot results
    results = sum([(ytest[i]-np.dot(theta1,test[i]))**2 for i in range(test.shape[0])])
    print 'Total BGD error: ', results
    
    #plot the error
    plt.plot(err, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=24)
    plt.ylabel(r'J($\theta$)', fontsize=24)
    plt.show()
    
    #plot test results
    x = [i[1] for i in test]
    plt.plot(x, ytest, 'ro', label='Data')
    
    #plot predictions
    plt.plot(x, [np.dot(theta1,i) for i in test], 'b-', label='Model')
    plt.show()
    
if __name__=="__main__":
    test_bgd()