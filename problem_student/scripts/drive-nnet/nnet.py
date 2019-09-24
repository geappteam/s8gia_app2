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
def loadEpisodeArrayDictionary(pklzFileName):
    recordingFilename = os.path.join(CDIR, 'recordings', pklzFileName)
    episode = EpisodeRecorder.restore(recordingFilename)
    return episode.states

def selectStatesVariablesArrayDictionary(states):
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
    
    # Selected 4 targets (also known as number of outputs chosen) :
    # - accelCmd
    # - brakeCmd
    # - steerCmd
    # - gearCmd
    
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
                    ,
                    'accelCmd': state['accelCmd'][0],
                    'brakeCmd': state['brakeCmd'][0], 
                    'steerCmd': state['steerCmd'][0], 
                    'gearCmd': state['gearCmd'][0]
                }
        selectedStates.append(selectedState)
    return selectedStates

def normalizeStatesArrayDictionary(states):
    #Recognize all the keys just in the first dictionary
    #of the array
    keys = list()
    for key in states[0]:
        keys.append(key)
        
    #Find both min and max value for all given keys 
    #initializing with first and second dictionary
    minValues = states[0]
    maxValues = states[1]
    
    for state in states:
        for key in keys:
            if state[key] < minValues[key]:
                minValues[key] = state[key]
            if state[key] > maxValues[key]:
                maxValues[key] = state[key]
    
    #Normalize all data values
    normalizedStates = list()
    for state in states:
        normalizedState = {}
        for key in keys:
            normalizedState[key] = (state[key]-minValues[key])/(maxValues[key]-minValues[key])
        normalizedStates.append(normalizedState)
    
    return normalizedStates, keys

def seperateTrainTestArrayDictionary(states, trainPercent):
    #Must be in the range of 0 to 100
    if trainPercent > 100 or trainPercent < 0:
        print('[ERROR] : Training and testing percentage are disproportionate')
        return
    #We'll want to assume train percentage is always greater than the test percentage
    elif trainPercent < 50:
        print('[WARNING] : Training percentage is lower than testing percentage, may leading to a bad neural model design')
    
    statesLength = len(states)
    
    trainLenght = int(statesLength * trainPercent / 100)
    
    #Sets separation
    trainStates = states[0:trainLenght-1]
    testStates = states[trainLenght:statesLength-1]
    
    return trainStates, testStates

def selectTargetArrayDictionary(states):
    #Selection is hardcoded for now. It would be better to specify as arguments 
    # Selected 4 targets (also known as number of outputs chosen) :
    # - accelCmd
    # - brakeCmd
    # - steerCmd
    # - gearCmd
    
    selectedStates = list()
      
    for state in states:
        selectedState = {
                            'accelCmd': state['accelCmd'],
                            'brakeCmd': state['brakeCmd'], 
                            'steerCmd': state['steerCmd'], 
                            'gearCmd': state['gearCmd']
                        }
        selectedStates.append(selectedState)
    return selectedStates
    
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
                    'angle': state['angle'],
#                    'curLapTime': state['curLapTime'],
#                    'damage': state['damage'],
#                    'distFromStart': state['distFromStart'],
                    'distRaced': state['distRaced'],
#                    'fuel': state['fuel'],
                    'gear': state['gear'],
                    'rpm': state['rpm'],
                    'speedX': state['speedX'],
                    'speedY': state['speedY'],
                    'track0': state['track0'],
#                    'track1': state['track1'],
#                    'track2': state['track2'],
#                    'track3': state['track3'],
#                    'track4': state['track4'],
#                    'track5': state['track5'],
#                    'track6': state['track6'],
#                    'track7': state['track7'],
#                    'track8': state['track8'],
                    'track9': state['track9'],
#                    'track10': state['track10'],
#                    'track11': state['track11'],
#                    'track12': state['track12'],
#                    'track13': state['track13'],
#                    'track14': state['track14'],
#                    'track15': state['track15'],
#                    'track16': state['track16'],
#                    'track17': state['track17'],
                    'track18': state['track18'],
                    'trackPos': state['trackPos'], 
                    'wheelSpinVel0': state['wheelSpinVel0'],
                    'wheelSpinVel1': state['wheelSpinVel1'],
                    'wheelSpinVel2': state['wheelSpinVel2'],
                    'wheelSpinVel3': state['wheelSpinVel3']
                }
        selectedStates.append(selectedState)
    return selectedStates

def arrayDictionaryToArrayArrayFloats(arrayDictionaryStates):
    arrayArrayStates = list() 
    
    for arrayDictionaryState in arrayDictionaryStates:
        arrayArrayStates.append(list(arrayDictionaryState.values()))
    
    arrayArrayStates = np.array(arrayArrayStates, dtype=np.float32)
    
    return arrayArrayStates
        

#TODO Optional: Could be interesting to filter out noisy data(ones in the extremes), 
#      to be continued...
#def filterDataArrayDictionary(states):

#TODO Optional: Could be interesting to shuffle every data, changing the training order, 
#      to be continued...
#def shuffleDataArrayDictionary(states):

###############################################
# Define code logic here
###############################################

def main():
    #Load data
    states = loadEpisodeArrayDictionary('track.pklz')
    #Selecting and buffering chosen state variables
    states = selectStatesVariablesArrayDictionary(states)
    #Normalize all state variables in a range of [0,1]
    states, keys = normalizeStatesArrayDictionary(states)
    #Create a portion of states for training (second argument in the range of [0,100]) 
    #and anthoner one for testing
    trainStates, testStates = seperateTrainTestArrayDictionary(states, 75)
    
    #Train sets expected values
    trainTargetStates = selectTargetArrayDictionary(trainStates)
    #Train sets input values
    trainDataStates = selectDataArrayDictionary(trainStates)
    
    #Test sets expected values
    testTargetStates = selectTargetArrayDictionary(testStates)
    #Test sets input values
    testDataStates = selectDataArrayDictionary(testStates)    
    
    #Fitting function accepts array of arrays
    trainTargetStates = arrayDictionaryToArrayArrayFloats(trainTargetStates)
    trainDataStates = arrayDictionaryToArrayArrayFloats(trainDataStates)
    testTargetStates = arrayDictionaryToArrayArrayFloats(testTargetStates)
    testDataStates = arrayDictionaryToArrayArrayFloats(testDataStates) 
    
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

    # Perform training
    # TODO : Tune the maximum number of iterations and desired error
    model.fit(trainDataStates, trainTargetStates, batch_size=len(trainDataStates),
              epochs=1000, shuffle=True, verbose=1)

    # Save trained model to disk
    model.save('nnet.h5')

    # Test model (loading from disk)
    model = load_model('nnet.h5')
    targetPred = model.predict(testDataStates)

    # Print the number of classification errors from the training data
    nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(testTargetStates, axis=-1))
    accuracy = (len(testDataStates) - nbErrors) / len(testDataStates)
    print('Classification accuracy: %0.3f' % (accuracy))


if __name__ == "__main__":
    main()
