import sys
from math import e
from multiprocessing import Pool
import timeit

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, datasets
from scipy.stats import linregress

from bgd import lr_bgd
from sgd import log_reg_sgd, h, log_reg_regularized_sgd


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
    
    #number of examples
    EX = 500
    
    #learning rate
    a = 0.001
    
    #create a synthetic data set
    x,y = datasets.make_classification(EX)
    
    #append a 1 column at index 0 in x
    x = np.hstack((np.ones((x.shape[0],1)),x))
    
    #train on half the data
    theta,err = log_reg_sgd(x[:EX/2],y[:EX/2],a,max_iter=100)
    
    #plot the error
    plt.plot(err, linewidth=2)
    plt.xlabel('Training example', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.show()
    
    #predict the test set
    pred = [h(x[i],theta) for i in xrange(EX/2,EX)]
    
    #plot the error as a function of training examples
    fpr, tpr, thresholds = metrics.roc_curve(y[EX/2:], pred)
    
    #plot the ROC curve
    plt.plot(fpr,tpr, linewidth=2)
    plt.xlabel('False positive rate', fontsize=20)
    plt.ylabel('True positive rate', fontsize=20)
    plt.show()
    
    #measure the performance using ROC and AUC
    auc = metrics.auc(fpr, tpr)
    
    print 'AUC of classifier: ', auc


def train(input):
    x = input['x']
    y = input['y']
    a = input['learning_rate']
    iters = input['iters']
    results = log_reg_sgd(x,y,a,max_iter=iters,debug=False)
    print 'Done, ', results
    return results


def test_parallel_sgd():
    #learning rate
    a = 0.001
    
    #create a synthetic data set, default features, 1500 examples, 2 classes
    x,y = datasets.make_classification(1500)
    
    #append a 1 column at index 0 in x
    x = np.hstack((np.ones((x.shape[0],1)),x))
    
    ### PARALLEL VERSION ###
    #worker pool
    pool = Pool(4)
    
    input = [{'x':x[:250],'y':y[:250],'learning_rate':a,'iters':500},
             {'x':x[250:500],'y':y[250:500],'learning_rate':a,'iters':500},
             {'x':x[500:750],'y':y[500:750],'learning_rate':a,'iters':500},
             {'x':x[750:1000],'y':y[750:1000],'learning_rate':a,'iters':500}]
    
    thetas = pool.map(train, input)
    
    #compute the average
    theta = np.mean(thetas)
    
    #we take the average prediction
    b_pred = [h(x[i],theta) for i in xrange(1000,1500)]
    
    #plot the error as a function of training examples
    b_fpr, b_tpr, thresholds = metrics.roc_curve(y[1000:], b_pred)
    
    #plot the ROC curve
    plt.plot(b_fpr,b_tpr, 'r-', label='Bagged', linewidth=2)
    
    
    #measure the performance using ROC and AUC
    b_auc = metrics.auc(b_fpr, b_tpr)
    
    print 'AUC of parallel classifier: ', b_auc
    ###
        
    ### SEQUENTIAL VERSION ###
    #train on half the data
    theta = log_reg_sgd(x[:1000],y[:1000],a,max_iter=500,debug=False)
    
    #predict the test set
    pred = [h(x[i],theta) for i in xrange(1000,1500)]
    
    #plot the error as a function of training examples
    fpr, tpr, thresholds = metrics.roc_curve(y[1000:], pred)
    
    #plot the ROC curve
    plt.plot(fpr,tpr, 'b-', label='Non-bagged', linewidth=2)
    plt.xlabel('False positive rate', fontsize=20)
    plt.ylabel('True positive rate', fontsize=20)
    plt.legend(loc=0)
    plt.show()
    
    #measure the performance using ROC and AUC
    auc = metrics.auc(fpr, tpr)
    
    print 'AUC of sequential classifier: ', auc
    
    #write results to file for later
    with open('bag_results.tsv','w') as out:
        for i,j in zip(b_fpr,b_tpr):
            out.write("\t".join((str(i),str(j)))+"\n")
    
    
    #write results to file for later
    with open('results.tsv','w') as out:
        for i,j in zip(fpr,tpr):
            out.write("\t".join((str(i),str(j)))+"\n")


if __name__=="__main__":
    test_bgd()
    test_sgd()
    test_parallel_sgd()
    