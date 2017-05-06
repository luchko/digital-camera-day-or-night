#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:54:20 2017

@author: ivan

The purpose of this module is testing and lopping through different models 
('gray', RGB, color hist) for Day/Night image classification based on the 
data previously extracted in 'colorHistExtraction.py' and 
'RGBhistExtraction.py' modules and stored in '/jobs' folder in

    'job--[RGB].pkl'
    'job--[gray].pkl' files
    'job--[Nclr=2_Npx=10000].pkl'
          *
          *
          
pickle files

SVC parameters for each model were estimated using grid search 
with cross-validation. The performance of the selected hyper-parameters 
and trained model is then measured on a dedicated evaluation set 
that was not used during the model selection step.

Module contains the definition of the following functions:

    def extractFeatures(xDay,xNight,pcaBool=False,n_cut=10000):
    def get_bestSVC(X_train,y_train,score = 'recall'):
    def testClf(xDay,xNight,boolPCA,n_cut,N_test=100,testSize=0.15):
    def saveModel(jobFile,boolPCA=False,n_cut=0):
    def getScore_gray(N_test=100,testSize=0.15):
    def getScore_RGB(N_test=100,testSize=0.15):
    def getScore_colorHist_demo(N_test=100,testSize=0.15):
    def selectBest_colorHist_model(N_test=100,testSize=0.15):
        
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
from time import time
from colorHistExtraction import maxDiff_sort
from colorHistExtraction import diffHistDisplay
import sys


def extractFeatures(xDay,xNight,pcaBool=False,n_cut=10000):
    '''
    extract features from images color histogram
    
    quasi PCA application is possible
    '''
    yDay = np.zeros(xDay.shape[0])
    yNight = np.ones(xNight.shape[0])
    
    if pcaBool:
        idsPCA,xDay,xNight = maxDiff_sort(xDay,xNight)    
        X = np.vstack((xDay[:,:n_cut],xNight[:,:n_cut]))
        idsPCA = idsPCA[:n_cut]
    else:
        idsPCA = np.arange(xDay.shape[1])
        X = np.vstack((xDay,xNight))
        
    y = np.hstack((yDay,yNight))
    
    #center data to the mean value and scale it by their standard deviation
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    
    return X,y,idsPCA,scaler
    
        
def get_bestSVC(X_train,y_train,score = 'recall'):
    '''
    Grid search of the hyper-parameters via cross-validation
    
    other metrics (scores): ['precision', 'recall', 'f1']
    
    return: SVC
    '''    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        
#    print("\n# Tuning hyper-parameters for {0}, #features = {1}\n".\
#          format(score,X_train.shape[1]))
    
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)
        
    return clf


def testClf(xDay,xNight,boolPCA,n_cut,N_test=100,testSize=0.15):
    '''
    estimates the cross-validation test score for given model
    
    the larger number of tests 'N_test' the smaller 'test_size' can be chosen

    returns: score, std
    '''
    t0 = time()
    
    n_colors = xDay.shape[1]
            
    print("\n# Model cross-validation for n_colors = {0}, PCA = {1}, n_cut = {2}\n".\
          format(n_colors,boolPCA,n_cut))

    (X,y) = extractFeatures(xDay,xNight,boolPCA,n_cut)[0:2]
    
    # cross-validation
    score_list = []
    
 
    for i in range(N_test):

        timeLeft = 0 if i==0 else (time() - t0)*(N_test-i)/i
        sys.stdout.write('\rCross-validation test: {0}/{1}, left:{2:.0f}s, test dataset size: {3}'.\
              format(i+1,N_test,timeLeft,testSize))
        sys.stdout.flush()
        
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testSize)       
        if testSize == 0: #test on the train data
            X_test, y_test = X_train, y_train
        
        clf = get_bestSVC(X_train,y_train,'recall')
    
        y_true, y_pred = y_test, clf.predict(X_test)
        score_list.append(accuracy_score(y_true, y_pred))
    
    score = np.mean(score_list)
    std = np.std(score_list)

    print('\n\nAccuracy classification score: {0:.3f}+-{1:.3f}'.format(score,std))
    print("done in %0.3fs." % (time() - t0))
    
    return score, std


def saveModel(jobFile,boolPCA=False,n_cut=0):
    '''
    saves model defined by data stored in jobFile and additional parameters
    boolPCA and n_cut
    
    model is saved into "models/model--[Nclr={0},PCA={1},Nct={2}.pkl" file
    '''
    
    print("=================================================")        
    t0 = time()

    # loaf model data
    (kmeans,xDay,xNight,histLegend) = joblib.load(jobFile)
    n_colors = xDay.shape[1]

    print("# Saving model: n_colors = {0}, PCA = {1}, n_cut = {2}".\
          format(n_colors,boolPCA,n_cut))

    (X,y,idsPCA,scaler) = extractFeatures(xDay,xNight,boolPCA,n_cut)

    clf = get_bestSVC(X,y,'recall')

    histLegend = np.reshape(kmeans.cluster_centers_,(n_colors,3))[idsPCA]

    joblib.dump((kmeans,clf,scaler,idsPCA,histLegend),
                "models/model--[Nclr={0},PCA={1},Nct={2}.pkl".\
                         format(n_colors,boolPCA,n_cut))   

    print("\ndone in %0.3fs." % (time() - t0))
    print("=================================================\n")


def getScore_gray(N_test=100,testSize=0.15):
    '''
    testing 'gray' classification model

    '''
    print("=================================================\n")
    print("Testing 'gray' classification model")    
    (xDay,xNight) = joblib.load('jobs/job--[gray].pkl')
    score, std = testClf(xDay,xNight,False,1,N_test,testSize)
    print("=================================================\n")


def getScore_RGB(N_test=100,testSize=0.15):
    '''
    testing RGB classification model

    '''
    print("=================================================\n")
    print("Testing RGB classification model")    
    (xDay,xNight,RGB_Legend) = joblib.load('jobs/job--[RGB].pkl')
    score, std = testClf(xDay,xNight,False,3,N_test,testSize)
    print("=================================================\n")


def getScore_colorHist_demo(N_test=100,testSize=0.15):
    '''
    demo of the testing colorHist classification model

    '''
    print("=================================================\n")
    print("Testing colorHist classification model")    
    # specify model parameters
    n_colors = 70;    num_pix=5000
    jobFile = 'jobs/job--[Nclr={0}_Npx={1}].pkl'.\
                                       format(n_colors,num_pix)         
    boolPCA = False; n_cut = n_colors
    # loaf model data
    (kmeans,xDay,xNight,histLegend) = joblib.load(jobFile)
    # display diff histogram 
    ids_sort,xDay_,xNight_,histLegend_ = maxDiff_sort(xDay,xNight,histLegend)
    diffHistDisplay(xDay_,xNight_,histLegend_)                   
     
    score, std = testClf(xDay,xNight,boolPCA,n_cut,N_test,testSize)
    print("=================================================\n")


def selectBest_colorHist_model(N_test=100,testSize=0.15):
    '''
    best colorHist classification model selection
    
    Models are stored in "/jobs" folder
    
    CAUTION: execution for large N_test might take some time

    '''    
    print("=================================================\n")
    print("Best colorHist classification model selection")    

    # pick colorHist model plickle files 
    jobsFile_list = []
    jobs_list = []
    jobs_list = jobs_list + [(n_colors,10000) for n_colors in [2,5,10,20,30]]
    jobs_list = jobs_list + [(n_colors,5000) for n_colors in [40,50,70]]
    jobs_list = jobs_list + [(n_colors,2500) for n_colors in [100,150,200]] 
    for (n_colors,num_pix) in jobs_list:        
        jobsFile_list.append('jobs/job--[Nclr={0}_Npx={1}].pkl'.\
                            format(n_colors,num_pix))         

    best_score = 0
    score_list=[]
    for jobFile in jobsFile_list:
        
        try:    
            (kmeans,xDay,xNight,histLegend) = joblib.load(jobFile)
    
            print('\n====================\n')
            # display diff histogram 
            ids_sort,xDay_,xNight_,histLegend_ = maxDiff_sort(xDay,xNight,histLegend)
            diffHistDisplay(xDay_,xNight_,histLegend_)                   
    
            n_colors = xDay.shape[1]
            # in order to apply PCA change the following values 
            # n_cut might be looping for best model selection
            boolPCA = False; n_cut = n_colors
     
            score, std = testClf(xDay,xNight,boolPCA,n_cut,N_test,testSize)
            score_list.append((n_colors,n_cut,score,std))
            
            if score > best_score:        
                best_score = score
                best = (n_colors,n_cut)
            
            print('\n====================\n')            
    
        except FileNotFoundError:
            pass

    #printing report on best model   
    print('Score table for different colotHist models:\n')
    for (n_colors,n_cut,score,std) in score_list:
        print('  n_colors = {0}, n_cut = {0}, score = {2:.3f}+-{3:.3f}'.\
              format(n_colors,n_cut,score,std))         
    
    (n_colors_best,n_cut_best) = best
    print('\nBest model: n_colors = {0}, PCA = {1}, n_cat = {2}'.\
          format(n_colors_best,boolPCA, n_cut_best))

    print("=================================================\n")



if __name__ == '__main__':    

    #HINT: for real test set larger N_test value (e.g. 100)
    #CAUTION: execution for large N_test might take some time !!!
             
    #uncoment line below for testing 'gray' classification model
    getScore_gray(N_test=10,testSize=0.15)

    #uncoment line below for testing RGB classification model
    getScore_RGB(N_test=10,testSize=0.15)

    #uncoment line below for testing RGB classification model 
    getScore_colorHist_demo(N_test=10,testSize=0.15)

#    # uncoment for the best colorHist classification model selection
#    # Models are stored in "/jobs" folder
#    selectBest_colorHist_model(N_test=2,testSize=0.15)

#    # save selected model
#    bestJobFile = 'jobs/job--[Nclr={0}_Npx={1}].pkl'.format(70,5000)
#    saveModel(bestJobFile,boolPCA=False,n_cut=70)            
