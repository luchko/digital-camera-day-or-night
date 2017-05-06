#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:33:11 2017

@author: ivan

The purpose of this module is testing pretrained model stored in 
"model--[Nclr={0},PCA={1},Nct={2}.pkl" pickle file on user own dataset.

By default module classify images stroed in "images/test" folder 
printing images with corresponding class label.

Module contains the definition of the following functions

    def predictClass(img,model)

"""

import matplotlib.pylab as plt
from sklearn.externals import joblib
from colorHistExtraction import buildColorHist
from colorHistExtraction import imgDisplay
import os

def predict_class(img, model):
    '''
    returns img class label according to pretrained model
    
    input:
        model = (kmeans,clf,scaler,idsPCA,histLegend)
    '''
    (kmeans,clf,scaler,idsPCA,histLegend) = model    
    hist = buildColorHist(img,kmeans)
    X = hist[idsPCA].reshape((1,idsPCA.shape[0]))
    X = scaler.transform(X)
    y = clf.predict(X)
    label = 'Day image' if y==0 else 'Night image'
    
    return label

     
if __name__ == '__main__':    
    
    #load model
    model = joblib.load("models/model--[Nclr={0},PCA={1},Nct={2}.pkl".\
                                 format(70,False,70))   
 
    imagesFolder = os.getcwd()+'/images/test'
    imgFileNamesList = os.listdir(imagesFolder)    

    for imgFileName in imgFileNamesList:
        # Load Image and transform to a 2D numpy array.
        img = plt.imread(imagesFolder+'/' + imgFileName)
 
        label = predict_class(img,model)
        imgDisplay(img,label)
