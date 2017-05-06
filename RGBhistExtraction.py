#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:32:39 2017

@author: ivan

The purpose of this module is extraction of RGB color histograms from images 
and saving the data for the RGB and 'gray' models futher training
Data is saved to 'job--[RGB].pkl' and 'job--[gray].pkl' files

Module contains the definition of the following functions:
    
    def extractRGB_features_folder(imagesFolder):
    def extractRGB_features(pklFileName = 'jobs/job--[RGB].pkl'):
    def extractGray_features(pklFileName = 'jobs/job--[gray].pkl'):
    def RGB_hist_demo(pklFileName='jobs/job--[RGB].pkl'):


"""
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.externals import joblib
from colorHistExtraction import histDisplay

def extractRGB_features_folder(imagesFolder):
    '''
    Extract RGB  color histograms from images stored in 'imagesFolder' folder
    
    '''
    print("RGB features extraction from images stored in '/{0}' folder".\
          format(os.path.basename(imagesFolder)))
    t0 = time()
    imgFileNames = os.listdir(imagesFolder)    
    num_files = len(imgFileNames)
    n_colors = 3
    #alocate memory for imgs_sample
    featuresMatrix = np.zeros((num_files,n_colors))

    for k in range(num_files):
        # Load Image and transform to a 2D numpy array.
        img = plt.imread(imagesFolder+'/' + imgFileNames[k])
        if img.shape[2] == 3:
            featuresMatrix[k,:] = np.mean(img,axis=(0,1))
    
    print("done in %0.3fs.\n" % (time() - t0))
 
    return featuresMatrix/255

def extractRGB_features(pklFileName='jobs/job--[RGB].pkl'):
    '''
    RGB features extraction
    
    '''
    print("\n=================================================\n")

    RGB_Legend = np.array([[255,0,0],[0,255,0],[0,0,255]])
 
    #calculate RGB histograms for images dataset
    xDay = extractRGB_features_folder(os.getcwd()+'/images/day')
    xNight = extractRGB_features_folder(os.getcwd()+'/images/night')    
    print("Saving results into '{}' pickle file\n".format(pklFileName))
    joblib.dump((xDay,xNight,RGB_Legend),pklFileName)
    
    print('Mean RGB color histogram for Day images [0-255]:\n {0}'.\
          format(255*xDay.mean(axis=0)))
    
    print('\nMean RGB color histogram for Night images [0-255]:\n {0}'.\
          format(255*xNight.mean(axis=0)))

    print("\n=================================================\n")

    return (xDay,xNight,RGB_Legend)


def extractGray_features(pklFileName='jobs/job--[gray].pkl'):
    '''
    Gray features extraction
    
    '''
    print("\n=================================================\n")
    print('Gray features extraction')
    try:       
        (xDay,xNight,RGB_Legend) = joblib.load('jobs/job--[RGB].pkl')   
    
    except FileNotFoundError:
        (xDay,xNight,RGB_Legend) = extractRGB_features()

    xDay=np.reshape(xDay.mean(axis=1),(len(xDay),1))
    xNight=np.reshape(xNight.mean(axis=1),(len(xNight),1))
    print("\nSaving results into '{}' pickle file\n".format(pklFileName))
    joblib.dump((xDay,xNight),pklFileName)
    meanDay = xDay.mean()
    meanNight = xNight.mean()
    
    print('Mean gray color value for Day images [0-255]: {0:.1f}'.\
          format(255*meanDay))
    
    print('Mean gray color value for Night images [0-255]: {0:.1f}'.\
          format(255*meanNight))
    print("\n=================================================\n")

    return (meanDay, meanNight)


def RGB_hist_demo(pklFileName='jobs/job--[RGB].pkl'):
    '''
    Display images dataset mean RGB histograms  
    
    '''
    print('RGB histogram demo:\n')
    try:       
        (xDay,xNight,RGB_Legend) = joblib.load(pklFileName)   
    
    except FileNotFoundError:
        (xDay,xNight,RGB_Legend) = extractRGB_features(pklFileName)
          
    print('Mean RGB color histogram for Day images [0-255]:\n {0}'.\
          format(255*xDay.mean(axis=0)))
    
    print('\nMean RGB color histogram for Night images [0-255]:\n {0}'.\
          format(255*xNight.mean(axis=0)))

    histDisplay(xDay.mean(axis=0),
                       RGB_Legend,
                       'RGB feature hist of the Day images')
    
    histDisplay(xNight.mean(axis=0),
                       RGB_Legend,
                       'RGB feature hist of the Night images')

    
if __name__ == '__main__':    

    #uncoment for RGB histograms demo
    RGB_hist_demo(pklFileName = 'jobs/job--[RGB].pkl')

 
#    #uncoment for RGB feature extraction
#    #CAUTION: if images dataset folder is empty it might damage 
#    #         previously extracted data
#    extractRGB_features(pklFileName = 'jobs/job--[RGB].pkl')
              

    # uncoment this section for gray color model calculatoin
    extractGray_features(pklFileName = 'jobs/job--[gray].pkl')
    
    
    