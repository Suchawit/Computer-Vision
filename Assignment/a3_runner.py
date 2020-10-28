#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template

    
#Refer to code for defaults

#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pprint
import pickle
import os

pp = pprint.PrettyPrinter(indent=4)

#Classifier models for you to use
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


#Helper functions provided for you
from cifar10 import cifar10
from data import plot_confusion_matrix, Timer 
#Configuration file
import config

#TODO: your implementation of HOG
from hog import hog

#Prevents python3 from failing
TODO = None

#Epsilon 
EPS=1e-6

def evaluate(X, y, model, verbose=False):
    """For the model, calculate the:
        - Recall (per class)
        - Precision (per class)
        - Average Recall, 
        - Average Precision,
        - Average F1 score
        - Average Accuracy 

    Parameters
    ----------
    X : ndarray NxD array where N is the number of instances and D and the number of features 
    y : narray Nx1 array with a label for each instance 
    model : sklearn class model 

    Returns
        results : dictionary 
    -------
    
    """
    results = {"accuracy" : 0,
               "recall" : 0,
               "precision" : 0,
               "avg_recall" : 0,
               "avg_precision" : 0,
               "fscore" : 0}

    print("Use our trained model to predict h: X -> [0,10]");
    #Predict the class by getting the class label (ie index) with the max probability 
    pred = model.predict(X)

    print("Creating confusion matrix and calculating evaluation metrics");
    #Calculate the confusion matrix, and normalize it between 0-1
    cm = confusion_matrix(y,pred).astype(np.float32)

    #From the confusion matrix, calculate precision/recall/f1-measure
    results['recall'] = np.diag(cm) / (np.sum(cm, axis=1) + EPS)
    results['avg_recall'] = np.mean(results['recall'])

    results['precision'] = np.diag(cm) / (np.sum(cm, axis=0) + EPS)
    results['avg_precision'] = np.mean(results['precision'])

    results["fscore"] = 2 * ( results['avg_precision'] * results['avg_recall'] ) / (results['avg_precision'] + results['avg_recall'] + EPS)

    results["accuracy"] = np.mean(pred == y)
    
    if config.verbose:
        plot_confusion_matrix(cm, save=True)


    return results

def get_features(train=True):
    if train:
        # Load cifar10 train data and labels
        print("Reading training data...")
        x_data, y_data = cifar10(config.path_to_cifar, "train")
    else:
        print("Reading testing data...")
        x_data, y_data = cifar10(config.path_to_cifar, "test")

    N_data = len(x_data)

    assert x_data.shape[0] == len(y_data) and "Both data and labels should be the same set size"
    
    print("Num of training samples: ", N_data)
    x_data = np.array([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in x_data], dtype=np.float32)
    x_data /= 255.

    # Extract features
    print("Extracting HOG Features, go grab a coffee...", end=" ")
    with Timer(verbose=False) as t:
        x_data  = hog(x_data)
    print("HoG extraction for train set took {} mins".format(t.mins))
    
    if config.min_max_norm:
        #Normalize the HOG feature vectors by rescaling the value range to [-1, 1] 
        print("Normalize the HOG features by rescaling the value range to [-1, 1]")
        scaler = MinMaxScaler(feature_range=(-1,1)).fit(x_data)
        x_data = scaler.fit_transform(x_data)
    elif config.unit_vector_norm:
        #Normalize the HOG feature vectors by converting them to unit vectors (vector has length of 1)
        print("Normalize the HOG features by converting them to unit vectors")
        x_data = x_data / np.linalg.norm(x_data)

    if train:
        #Randomly shuffle the data if training. 
        rand_idx = np.arange(N_data)
        np.random.shuffle(rand_idx)
        x_data = x_data[rand_idx]
        y_data = y_data[rand_idx]
    
    return x_data, y_data


def train_model(x_data, y_data):
    print("Training the SVM classifier ... this may take a while, go grab another coffee.")
    with Timer(verbose=False) as t:
        model = SVC(kernel='linear') 
        model.fit(x_data, y_data)
    
    print(f"Saving the model as {config.saved_model}")
    pickle.dump(model, open(config.saved_model, 'wb'))
    
    print("Training took {} mins".format(t.mins))
    return model

def main():
    """
    Check if we have already trained a model
    If yes, check how good our model is at infering the correct label for our test dataset
    If no, then train a model.
    """
    if not os.path.isfile(config.saved_model):
        print("Training a new model")
        X, Y =  get_features(train=True) 
        model = train_model(X,Y) 
    else:
        print(f"Loading previously saved model {config.saved_model}")
        model = pickle.load(open(config.saved_model, 'rb'))
        X, Y =  get_features(train=False) 
    
    print("Evaluating the results")
    res = evaluate(X, Y, model, verbose=config.verbose)
    pp.pprint(res)

    

if __name__ == "__main__":
    main()


#
# a3_runner.py ends here
