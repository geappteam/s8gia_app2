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

def selectDataArrayDictionary(states):
    #Selection is hardcoded for now. It would be better to specify as arguments
    # Selected 14 data (also known as number of inputs chosen) :
    # - angle
    # - distRaced
    # - gear
    # - rpm
    # - speedX
    # - speedY
    # - track[0] #First value (at -pi/2)
    # - track[9] #Middle value (at 0)
    # - track[18] # Last value (at +pi/2)
    # - trackPos
    # - wheelSpinVel[0]
    # - wheelSpinVel[1]
    # - wheelSpinVel[2]
    # - wheelSpinVel[3]
    
    selectedStates = list()
    
    
    for state in states:
        selectedState = {
                    'angle': state['angle'][0],
#                    'curLapTime': state['curLapTime'][0],
#                    'damage': state['damage'][0],
#                    'distFromStart': state['distFromStart'][0],
                    'distRaced': state['distRaced'][0],
#                    'fuel': state['fuel'][0],
                    'gear': state['gear'][0],
                    'rpm': state['rpm'][0],
                    'speedX': state['speed'][0],
                    'speedY': state['speed'][1],
                    'track0': state['track'][0],
#                    'track1': state['track'][1],
#                    'track2': state['track'][2],
#                    'track3': state['track'][3],
#                    'track4': state['track'][4],
#                    'track5': state['track'][5],
#                    'track6': state['track'][6],
#                    'track7': state['track'][7],
#                    'track8': state['track'][8],
                    'track9': state['track'][9],
#                    'track10': state['track'][10],
#                    'track11': state['track'][11],
#                    'track12': state['track'][12],
#                    'track13': state['track'][13],
#                    'track14': state['track'][14],
#                    'track15': state['track'][15],
#                    'track16': state['track'][16],
#                    'track17': state['track'][17],
                    'track18': state['track'][18],
                    'trackPos': state['trackPos'][0], 
                    'wheelSpinVel0': state['wheelSpinVel'][0],
                    'wheelSpinVel1': state['wheelSpinVel'][1],
                    'wheelSpinVel2': state['wheelSpinVel'][2],
                    'wheelSpinVel3': state['wheelSpinVel'][3]
#                    ,
#                    'accelCmd': state['accelCmd'][0],
#                    'brakeCmd': state['brakeCmd'][0], 
#                    'steerCmd': state['steerCmd'][0], 
#                    'gearCmd': state['gearCmd'][0]
                }
        selectedStates.append(selectedState)
    return selectedStates

#TODO *Important*: Needs to buffer target datasets for fitting, 
#      to be continued...
#def selectTargetArrayDictionary(states):

#TODO Optional: Could be interesting to filter out noisy data(ones in the extremes), 
#      to be continued...
#def filterDataArrayDictionary(states):

#TODO Optional: Could be interesting to shuffle every data, changing the training order, 
#      to be continued...
#def shuffleDataArrayDictionary(states):
    
#TODO *Important*: Needs to separate datasets for one training dataset and one testing dataset, 
#      to be continued...
def seperateTrainingTestingDatasets(states, trainPercent, testPercent):

    return trainSet, testSet

def scaleDataArrayDictionary(states):
    #TODO : Put this function out (rather in the main())
    #       and modify it so it's not hardcoded
    selectedStates = selectDataArrayDictionary(states)

    #Recognize all the keys just in the first dictionary
    #of the array
    keys = list()
    for key in selectedStates[0]:
        keys.append(key)
        
    #Find both min and max value for all given keys 
    #initializing with first and second dictionary
    minValues = selectedStates[0]
    maxValues = selectedStates[1]
    
    for state in selectedStates:
        for key in keys:
            if state[key] < minValues[key]:
                minValues[key] = state[key]
                #print('minValues[key]: %s' % (minValues[key]))
            if state[key] > maxValues[key]:
                maxValues[key] = state[key]
    
    #Normalize all data values
    normalizedStates = list()
    for state in selectedStates:
        normalizedState = {}
        for key in keys:
            normalizedState[key] = (state[key]-minValues[key])/(maxValues[key]-minValues[key])
        normalizedStates.append(normalizedState)
    
    return normalizedStates, keys

###############################################
# Define code logic here
###############################################

def main():
    #Prepare data
    states = loadDataArrayDictionary('track.pklz')
    states, keys = scaleDataArrayDictionary(states)
    
    # Create neural network
    model = Sequential()
    model.add(Dense(units=10, activation='sigmoid',
                    input_shape=(14,), name='input_layer'))
    model.add(Dense(units=5, activation='sigmoid', name='hidden_layer'))
    model.add(Dense(units=4, activation='sigmoid', name='output_layer'))
    print(model.summary())

    # Define training parameters
    # TODO : Tune the training parameters
    model.compile(optimizer=SGD(lr=0.1, momentum=0.9),
                  loss='mse')

#    # Perform training
#    # TODO : Tune the maximum number of iterations and desired error
#    model.fit(data, target, batch_size=len(data),
#              epochs=1000, shuffle=True, verbose=1)
#
#    # Save trained model to disk
#    model.save('nnet.h5')
#
#    # Test model (loading from disk)
#    model = load_model('nnet.h5')
#    targetPred = model.predict(data)
#
#    # Print the number of classification errors from the training data
#    nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(target, axis=-1))
#    accuracy = (len(data) - nbErrors) / len(data)
#    print('Classification accuracy: %0.3f' % (accuracy))


if __name__ == "__main__":
    main()
