from random import *
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from drum_environment import *

def read_data():
    # Read the prepared data, shuffle it, combine data with and without onset,
    # and split the data up into training data, test data, and validation data
    
    print("Reading datapoints")
    seed(65922)
    
    onset_filename=root_folder + "data/onset.txt"
    no_onset_filename=root_folder + "data/no_onset.txt"

    with open(onset_filename) as f:
        onset_data = np.array([[np.float64(i) for i in line.split(",")] for line in f])
    with open(no_onset_filename) as f:
        no_onset_data = np.array([[np.float64(i) for i in line.split(",")] for line in f])

    onset_data = onset_data
    no_onset_data = no_onset_data
    ratio=6/9
    ratio2=8/9
    indices=np.random.permutation(len(onset_data))
    train_indices=indices[:int(len(onset_data)*ratio)]
    test_indices=indices[int(len(onset_data)*ratio):int(len(onset_data)*ratio2)]
    validation_indices=indices[int(len(onset_data)*ratio2):]

    train_data=sum(zip(onset_data[train_indices],no_onset_data[train_indices]),())
    test_data=sum(zip(onset_data[test_indices],no_onset_data[test_indices]),())
    validation_data=sum(zip(onset_data[validation_indices],no_onset_data[validation_indices]),())

    y_train=[1 if i%2==0 else 0 for i in range(len(train_data))]
    y_test=[1 if (i+int(len(onset_data)*ratio))%2==0 else 0 for i in range(len(test_data))]
    y_validation=[1 if (i+int(len(onset_data)*ratio2))%2==0 else 0 for i in range(len(validation_data))]

    return (train_data,y_train,test_data,y_test,validation_data,y_validation)

def train_and_test(X,y,X_test,y_test,kernel="linear"):
    # Train an SVM and test its performance.
    
    score=0
    model=None
    C=0.01
    gamma=5
    print("Training with C="+str(C))
    #svm_clf = Pipeline([ ("scaler", StandardScaler()), ("svm_clf", SVC(kernel="linear",gamma=5, C=0.001,verbose=1,class_weight={0:1.5,1:1})), ],verbose=True)
    svm_clf = Pipeline([ ("scaler", StandardScaler()), ("svm_clf", SVC(kernel=kernel, gamma=gamma, C=C,verbose=1,class_weight={0:1.5,1:1})), ],verbose=True)
    svm_clf.fit(X, y)

    score=svm_clf.score(X_test,y_test)
    print("Score: "+str(score))

    return (svm_clf,score)
