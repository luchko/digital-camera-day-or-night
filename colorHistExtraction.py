#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:54:20 2017

@author: ivan

The purpose of this module is creation color basis (color quantization) 
using kmeans method and extraction of the color histograms from images 
according to this basis. Further this data is saved for color histogram 
model training

Module contains the definition of the following functions:
    
    def sampleImages(imagesFolder,num_pix=1000):
    def get_kmeans(folders_list,n_colors=10,num_pix=1000):
    def imgDisplay(img, title = ''):
    def recreate_image(codebook, labels, w, h):
    def imgQuantDemo(kmeans, img):
    def buildColorHist(img,kmeans):
    def histDisplay(hist,histLegend=['b'],title='',ylabel=''):
    def diffHistDisplay(xDay,xNight,histLegend):
    def maxDiff_sort(xDay,xNight,histLegend=None):
    def extractColorHist(imagesFolder,kmeans):
    def histDemo(xDay,xNight,histLegend):
    def runJob(folders_list,n_colors=10,num_pix=1000,pklFileName='temp.pkl'):

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time
from sklearn.externals import joblib
from sklearn.utils import shuffle
import gc
    

def sampleImages(imagesFolder,num_pix=1000):
    '''
    samples pixels from images stored in 'imagesFolder' folder
    
    num_pix: number of pixel per image
    '''
    print("Sampling {0} pixels per images from '/{1}' folder".\
          format(num_pix,os.path.basename(imagesFolder)))
    t0 = time()

    imgFileNames = os.listdir(imagesFolder)    
    num_files = len(imgFileNames)
    #alocate memory for imgs_sample
    imgs_sample = np.zeros((num_pix*num_files,3),dtype = np.uint8)

    for k in range(num_files):
        # Load Image and transform to a 2D numpy array.
        img = plt.imread(imagesFolder+'/' + imgFileNames[k])
        w, h, d = tuple(img.shape)
        if d == 3:
            img_arr = np.reshape(img, (w * h, d))
            #randomly pick num_pix puxels in image
            rand_ids = shuffle(range(w*h))[:num_pix]
            imgs_sample[k:k+num_pix,:] = img_arr[rand_ids,:]
    
    print("done in %0.3fs.\n" % (time() - t0))
    return imgs_sample


def get_kmeans(folders_list,n_colors=10,num_pix=1000):
    '''
    samples images data and performs K-means color quantization

    for given num_pix, imgs_sample is sampled and saved in 
    'px_samples/px_sample-[Npx=num_pix].pkl' pickle file only for the first time 
    if sample file already exist, it is loaded during the repeated queries
    
    images dataset is stoder in folders defined by 'folders_list'
       : 
    returns: k-means image compressor (color quantizator)
    '''
    
    assert folders_list != []

    #load sample of images pixels from file or create the sample
    smpFile = 'px_samples/px_sample-[Npx={}].pkl'.format(num_pix)
 
    try:
        imgs_sample = joblib.load(smpFile)
        print("Load sample of images pixels from '{}' file\n".format(smpFile))
    except FileNotFoundError:
        imgs_sample = np.zeros((1,3),dtype = np.uint8)
        for folder in folders_list:
            imgs_sample = np.vstack((imgs_sample,sampleImages(folder, num_pix)))
        imgs_sample = imgs_sample[1:,:]
        joblib.dump(imgs_sample,smpFile)
    
    print("K-means color quantization of sampled data")
    print("Sample size: {0} pixels".format(len(imgs_sample)))
    t0 = time()
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(imgs_sample)
    print("done in %0.3fs.\n" % (time() - t0))
    #free memory
    del imgs_sample; gc.collect()
    
    return kmeans


def imgDisplay(img, title = ''):
    '''Displays image img '''
    img = np.array(img, dtype=np.float64) / 255
    plt.figure()
    plt.clf()
    if title != '':
        plt.title(title)
    plt.axis('off')
    plt.imshow(img,interpolation='nearest');
    plt.show()
             
    
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def imgQuantDemo(kmeans, img):
    '''
    demo of the image kmeans color quantization
    '''
    # Transform Image to a 2D numpy array.    
    w, h, d = tuple(img.shape)
    assert d == 3
    img_arr = np.reshape(img, (w * h, d))
        
    # Get labels for all pixels
    print("Predicting color indices on the full images")
    t0 = time()
    labels = kmeans.predict(img_arr)
    print("done in %0.3fs." % (time() - t0))

    img_quant = recreate_image(kmeans.cluster_centers_,labels, w, h)
    
    n_colors = len(kmeans.cluster_centers_)
    histLegend = np.reshape(kmeans.cluster_centers_,(1,n_colors,3))
 
    imgDisplay(img,r'Original image ($256^3$ colors)')
    imgDisplay(img_quant,'Compressed image ({} colors)'.format(n_colors))
    imgDisplay(histLegend,"Color basis")
         
    
def buildColorHist(img,kmeans):
    '''
    returns: normalized histogram of colors appearence in image 
             according to basis defined in kmeans 
    '''
    n_colors = len(kmeans.cluster_centers_)
    # Transform Image to a 2D numpy array.    
    w, h, d = tuple(img.shape)
    if d != 3:
        raise Exception('not RGB image')
    img_arr = np.reshape(img, (w * h, d))
    img_arr = np.array(img_arr, dtype=np.float64)
    labels = kmeans.predict(img_arr)
    hist = np.histogram(labels, bins=n_colors)[0]
    hist = hist / sum(hist)
    
    return hist


def histDisplay(hist,histLegend=['b'],title='',ylabel=''):
    '''
    displays descriptors vector:histogram of colors appearence in image 
             according to basis defined in kmeans 
    input: 
        histLegend - 2D array containig legend colors
    '''
    histLegend = np.array(histLegend, dtype=np.float64) / 255
    plt.figure()
    if title != '':
        plt.title(title)
    plt.bar(range(len(hist)),hist,color=histLegend)
    plt.xlabel('histogram vector')
    if ylabel != '':
        plt.ylabel(ylabel)
    plt.show()


def diffHistDisplay(xDay,xNight,histLegend):
    '''
    display the normalized difference of the color appearing frequency 
    in Day and Night images
    '''
    histDay = np.mean(xDay,axis = 0)
    histNight = np.mean(xNight,axis = 0)
    histDisplay((histDay-histNight)/(histDay+histNight),
                histLegend,
                'Sorted diff/mean color histogram',
                'NIGHT        |        DAY')


def maxDiff_sort(xDay,xNight,histLegend=None):
    '''
    sort hish matrices maximizing the difference of the color appearing  
    in Day and Night images
    '''
    histDay = np.mean(xDay,axis = 0)
    histNight = np.mean(xNight,axis = 0)
    histDiff = (histDay-histNight)/(histDay+histNight) 
    ids_sort = np.argsort(np.abs(histDiff))[::-1] 
    xDay = xDay[:,ids_sort] 
    xNight = xNight[:,ids_sort]

    if histLegend is None:
        return ids_sort,xDay,xNight 
    else:
        histLegend = histLegend[ids_sort,:] 
        return ids_sort,xDay,xNight,histLegend 


def extractColorHist(imagesFolder,kmeans):
    '''
    Extract color histograms from images stored in 'imagesFolder' folder
    according to pretrained kmeans model
    '''
    print("Color hist extraction from images stored in '/{0}' folder".\
          format(os.path.basename(imagesFolder)))
    t0 = time()
    imgFileNames = os.listdir(imagesFolder)    
    num_files = len(imgFileNames)
    n_colors = len(kmeans.cluster_centers_)
    #alocate memory for imgs_sample
    histMatrix = np.zeros((num_files,n_colors))

    for k in range(num_files):
        # Load Image and transform to a 2D numpy array.
        img = plt.imread(imagesFolder+'/' + imgFileNames[k])
        if img.shape[2] == 3:
            histMatrix[k,:] = buildColorHist(img,kmeans)
    
    print("done in %0.3fs.\n" % (time() - t0))
 
    return histMatrix

def histDemo(xDay,xNight,histLegend):
    '''
    Color quantization demo.Displays color histogram for 
    All, Day, Night images and diffHist
    '''
    
    histDay = np.mean(xDay,axis = 0)
    histNight = np.mean(xNight,axis = 0)
    histAll = (histDay+histNight)/2 
    
    histDisplay(histAll,
                histLegend,
                'Mean color histogram for All images')
    
    histDisplay(histDay,
                histLegend,
                'Mean color histogram for Day images')

    histDisplay(histNight,
                histLegend,
                'Mean color histogram for Night images')

    #sorting according to max color appearence frequency difference
    ids_sort,xDay,xNight,histLegend = maxDiff_sort(xDay,xNight,histLegend)
    diffHistDisplay(xDay,xNight,histLegend)


def runJob(n_colors=10,num_pix=1000,pklFileName='temp.pkl'):
    '''
    Training kmeans classifier and color historgram extraction
    
    '''
    print("=================================================")
    print("Starting job with {} color clases\n".format(n_colors))
    t0 = time()
    
    #training classifier (image color quantizer)
    folders_list = [os.getcwd()+'/images/day',
                    os.getcwd()+'/images/night']
    kmeans = get_kmeans(folders_list,n_colors,num_pix)
    histLegend = np.reshape(kmeans.cluster_centers_,(n_colors,3) )
    
    #color histograms extraction               
    xDay = extractColorHist(os.getcwd()+'/images/day',kmeans)
    xNight = extractColorHist(os.getcwd()+'/images/night',kmeans)
                        
    print("Saving job results into '{}' pickle file\n".format(pklFileName))
    joblib.dump((kmeans,xDay,xNight,histLegend),pklFileName)   

    print("Display short results report\n")
    # sorting according to max color appearence frequency difference
    ids_sort,xDay,xNight,histLegend = maxDiff_sort(xDay,xNight,histLegend)
    diffHistDisplay(xDay,xNight,histLegend)

    print("\nJob is done in %0.3fs." % (time() - t0))
    print("=================================================\n")



if __name__ == '__main__':    

    
#    # uncoment the following section for color histogram extraction from images
#    # for real training choose larger sample size (e.g.num_pix=5000)  
#    # k-meeans might take a long time large img_sample and n_colors
#    # make sure that there is images dataset in "/images" folder
#    num_pix=500 
#    # n_colors_list = [50,70,100,150,200,250,300,350,400]
#    n_colors_list = [50, 70]
#    
#    for n_colors in n_colors_list:     
#        pklFileName = 'jobs/job--[Nclr={0}_Npx={1}].pkl'.\
#                                       format(n_colors,num_pix)
#        runJob(n_colors, num_pix,pklFileName)
#        gc.collect()
#    ######### 
    

    #the following section demomonstrates the color quantization
    num_pix=5000
    n_colors = 70
    pklFileName = 'jobs/job--[Nclr={0}_Npx={1}].pkl'.\
                                       format(n_colors,num_pix)        
    (kmeans,xDay,xNight,histLegend) = joblib.load(pklFileName) 

    img = plt.imread(os.getcwd()+'/images/day/testimage.jpg')
    imgQuantDemo(kmeans, img)
    histDemo(xDay,xNight,histLegend)                  
    ########


    