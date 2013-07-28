import sys
from math import e

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.preprocessing
from sklearn import metrics, datasets
from scipy.stats import linregress

from bgd import lr_bgd
from sgd import log_reg_sgd, h, log_reg_regularized_sgd

def plot(x,y):
    #for each feature
    for i in range(x.shape[1]):
        plt.plot(x[:,i],y, 'ro')
        plt.title('feature: '+str(i))
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
    plt.plot(x, [np.dot(theta1,i) for i in test], 'b-', linewidth=2, label='Model')
    plt.legend(loc=0)
    plt.xlabel('Scaled lower status of the population', fontsize=24)
    plt.ylabel('Scaled home price', fontsize=24)
    plt.show()


def test_sgd():
    #learning rate
    a = 0.001
    
    #create a synthetic data set
    x,y = datasets.make_classification(1000)
    
    #train on half the data
    theta,err = log_reg_sgd(x[:500],y[:500],a,max_iter=50)
    
    #plot the error
    plt.plot(err, linewidth=2)
    plt.xlabel('Training example', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.show()
    
    #predict the test set
    pred=[h(x[i],theta) for i in xrange(500,1000)]
    
    #plot the error as a function of training examples
    fpr, tpr, thresholds = metrics.roc_curve(y[500:], pred)
    
    #plot the ROC curve
    plt.plot(fpr,tpr, linewidth=2)
    plt.xlabel('False positive rate', fontsize=20)
    plt.ylabel('True positive rate', fontsize=20)
    plt.show()
    
    #measure the performance using ROC and AUC
    auc = metrics.auc(fpr, tpr)
    
    print 'AUC of classifier: ', auc
    

if __name__=="__main__":
    test_bgd()
    test_sgd()
    
    
    