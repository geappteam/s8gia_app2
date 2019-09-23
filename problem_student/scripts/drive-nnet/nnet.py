#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:41:58 2019

@author: Anthony Parris
"""
###############################################
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.decomposition.pca import PCA
from mpl_toolkits.mplot3d import Axes3D
###############################################

###############################################
import sys
import os
import logging
import matplotlib.pyplot as plt

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
###############################################

###############################################
# Define helper functions here
###############################################
def loadDataArrayDictionary(pklzFileName):
    recordingFilename = os.path.join(CDIR, 'recordings', pklzFileName)
    episode = EpisodeRecorder.restore(recordingFilename)
    return episode.states

def scaleDataArrayDictionary(states):
    #Recognize all the keys just in the first dictionary
    #of the array
    keys = list()
    for key in states[0]:
        keys.append(key)
            
    #Find both min and max value for all given keys 
    #initializing with first dictionary
    minValues = states[0]
    maxValues = states[0]
    
    for state in states:
        for key in keys:
            if state[key] < minValues[key]:
                minValues[key] = state[key]
            if state[key] > maxValues[key]:
                maxValues[key] = state[key]
    
    #Massage all data values
    for state in states:
        for key in keys:
            state[key] = (state[key]-minValues[key])/(maxValues[key]-minValues[key])
    
    return states, keys

###############################################
# Define code logic here
###############################################

def main():

    # Create neural network
    model = Sequential()
    model.add(Dense(units=31, activation='sigmoid',
                    input_shape=(data.shape[-1],), name='input_layer'))
    model.add(Dense(units=5, activation='sigmoid'), name='hidden_layer')
    model.add(Dense(units=4, activation='sigmoid'), name='output_layer')
    print(model.summary())

    # Define training parameters
    # TODO : Tune the training parameters
    model.compile(optimizer=SGD(lr=0.1, momentum=0.9),
                  loss='mse')

    # Perform training
    # TODO : Tune the maximum number of iterations and desired error
    model.fit(data, target, batch_size=len(data),
              epochs=1000, shuffle=True, verbose=1)

    # Save trained model to disk
    model.save('nnet.h5')

    # Test model (loading from disk)
    model = load_model('nnet.h5')
    targetPred = model.predict(data)

    # Print the number of classification errors from the training data
    nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(target, axis=-1))
    accuracy = (len(data) - nbErrors) / len(data)
    print('Classification accuracy: %0.3f' % (accuracy))


if __name__ == "__main__":
    main()
