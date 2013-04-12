import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.preprocessing

from bgd import lr_bgd

def test_bgd():
    #learning rate
    a = 0.0001
    
    #load data
    data = sklearn.datasets.load_boston()
    
    #scale it, split into training and test sets
    features = sklearn.preprocessing.scale(data['data'])
    train = features[:features.shape[0]/2]
    test = features[features.shape[0]/2:]
    y = sklearn.preprocessing.scale(data['target'])
    ytrain = y[:y.shape[0]/2]
    ytest = y[y.shape[0]/2:]
    
    #fit the model using bgd
    theta1, err = lr_bgd(a, train, ytrain)
    
    #measure the total error
    results = sum([(ytest[i]-np.dot(theta1,test[i]))**2 for i in range(test.shape[0])])
    
    #Print and plot results
    print 'Total BGD error: ', results
    plt.plot(err, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=24)
    plt.ylabel(r'J($\theta$)', fontsize=24)
    plt.show()
    
    #plot test results
    vals = sorted([(i,j) for i,j in zip(ytest, [np.dot(theta1,i) for i in test])], key=lambda k:k[0])
    plt.plot(map(lambda i: i[0], vals), 'ro', label='Data')
    plt.plot(map(lambda i: i[1], vals), 'b*', label='BGD')
    plt.xlabel('Test example, i', fontsize=24)
    plt.ylabel('P(i)', fontsize=24)
    plt.legend(loc=0)
    plt.show()
    
if __name__=="__main__":
    test_bgd()