#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:41:58 2019

@author: Anthony Parris
"""

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
import sys
import os
import logging

sys.path.append('../..')
from torcs.control.core import EpisodeRecorder
CDIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)

###############################################
# Define global variables here
###############################################
### DATASETS PARAMETERS ###
TRAIN_DATASETS_PERCENTAGE = 98
EPISODE_PATHS = ['track.pklz']

# Selected input data :
CHOSEN_INPUT_KEYS = {
                        'angle': ['angle', 0],
#                        'curLapTime': ['curLapTime', 0],
#                        'damage': ['damage', 0],
#                        'distFromStart': ['distFromStart', 0],
#                        'distRaced': ['distRaced', 0],
#                        'fuel': ['fuel', 0],
                        'gear': ['gear', 0],
                        'rpm': ['rpm', 0],
                        'speedX': ['speed', 0],
                        'speedY': ['speed', 1],
                        'track0': ['track', 0],
#                        'track1': ['track', 1],
#                        'track2': ['track', 2],
#                        'track3': ['track', 3],
#                        'track4': ['track', 4],
#                        'track5': ['track', 5],
#                        'track6': ['track', 6],
#                        'track7': ['track', 7],
                        'track8': ['track', 8],
#                        'track9': ['track', 9],
#                        'track10': ['track', 10],
#                        'track11': ['track', 11],
#                        'track12': ['track', 12],
#                        'track13': ['track', 13],
#                        'track14': ['track', 14],
#                        'track15': ['track', 15],
#                        'track16': ['track', 16],
#                        'track17': ['track', 17],
                        'track18': ['track', 18],
                        'trackPos': ['trackPos', 0], 
                        'wheelSpinVel0': ['wheelSpinVel', 0],
                        'wheelSpinVel1': ['wheelSpinVel', 1],
                        'wheelSpinVel2': ['wheelSpinVel', 2],
                        'wheelSpinVel3': ['wheelSpinVel', 3]
                    }

### MODEL CONFIGS ###
MODEL_NAME = 'nnet.h5'

#INPUT LAYER CONFIG
INPUT_UNITS = 10
INPUT_ACTIVATION = 'sigmoid'
INPUT_DATA_SHAPE = -1

#OTHER LAYERS CONFIG
MODEL_LAYERS_CONFIG =   [
                            Dense(units=15, activation='sigmoid', name='hidden_layer'),
                            Dense(units=4, activation='sigmoid', name='output_layer')
                        ]

#COMPILATION CONFIG
OPTIMIZER = SGD(lr=0.1, momentum=0.9)
LOSS = 'mse'

#FITTING CONFIG
EPOCHS = 1000
SHUFFLE = True
VERBOSE = 1

###############################################
# Define constant variables here
###############################################
### DATASETS PARAMETERS ###
RECORDING_FOLDER_PATH = 'recordings'

# Output data :
OUTPUT_KEYS =   {
                    'accelCmd': ['accelCmd', 0],
                    'brakeCmd': ['brakeCmd', 0], 
                    'steerCmd': ['steerCmd', 0], 
                    'gearCmd': ['gearCmd', 0]
                }

###############################################
# Define helper functions here
###############################################
def loadEpisodeArrayDictionary(episodePath):
    recordingFilename = os.path.join(CDIR, RECORDING_FOLDER_PATH, episodePath)
    episode = EpisodeRecorder.restore(recordingFilename)
    return episode.states

def loadEpisodesArrayDictionary(episodesPaths):
    concatenedStates = list()
    for episodePath in episodesPaths :
        states = loadEpisodeArrayDictionary(episodePath)
        concatenedStates = concatenedStates + states
    return concatenedStates

def selectStatesVariablesArrayDictionary(states, selectedVariablesKeysIndexes):
    selectedStates = list()
      
    for state in states:
        selectedState = list()
        for keyIndex in selectedVariablesKeysIndexes:
            selectedState.append(state[selectedVariablesKeysIndexes[keyIndex][0]][selectedVariablesKeysIndexes[keyIndex][1]])
            
        selectedStates.append(selectedState)
        
    return selectedStates

def normalizeStatesArrayArray(states):      
    #Find both min and max value for all given state variables 
    #initializing with first and second array
    minValues = states[0]
    maxValues = states[1]
    
    for state in states:
        for variable in state:
            if variable < minValues[state.index(variable)]:
                minValues[state.index(variable)] = variable
            if variable > maxValues[state.index(variable)]:
                maxValues[state.index(variable)] = variable
    
    #Normalize all data values
    normalizedStates = list()
    for state in states:
        normalizedState = list()
        for variable in state:
            #Ensuring maximum and minimum value clipping
            if variable <= minValues[state.index(variable)] :
                normalizedValue = minValues[state.index(variable)]
            elif variable >= maxValues[state.index(variable)]:
                normalizedValue = maxValues[state.index(variable)]
            else: 
                normalizedValue =   (variable - minValues[state.index(variable)])                           \
                                    /                                                                       \
                                    (maxValues[state.index(variable)] - minValues[state.index(variable)])
            normalizedState.append(normalizedValue)                                                         
        normalizedStates.append(normalizedState)
    
    return normalizedStates

def seperateTrainTestArrayArray(states, trainPercent):
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

def separateTargetAndDataArrayArray(states):
    targetStates = list()
    dataStates = list()
    stateLength = len(states[0])
    for state in states:
        targetStates.append(state[stateLength-5:stateLength-1])
        dataStates.append(state[0:stateLength-6])
    return targetStates, dataStates

def arrayArrayToNumpyArrayArrayFloats(states):  
    numpyArrayArrayStates = np.array(states, dtype=np.float32)
    
    return numpyArrayArrayStates
        
#TODO Optional: Could be interesting to filter out noisy data(ones in the extremes), 
#      to be continued...
#def filterDataArrayDictionary(states):

###############################################
# Define code logic here
###############################################

def main():
    #Load data
    states = loadEpisodesArrayDictionary(EPISODE_PATHS)
    #Selecting and buffering chosen state variables
    states = selectStatesVariablesArrayDictionary(states, {**CHOSEN_INPUT_KEYS, **OUTPUT_KEYS})

    #Create a portion of states for training (second argument in the range of [0,100]) 
    #and anthoner one for testing
    trainStates, testStates = seperateTrainTestArrayArray(states, TRAIN_DATASETS_PERCENTAGE)
    
    #Train sets expected and input values
    trainTargetStates, trainDataStates = separateTargetAndDataArrayArray(trainStates)
    
    #Test sets expected and input values
    testTargetStates, testDataStates = separateTargetAndDataArrayArray(testStates)  

    #Normalize all state variables in a range of [0,1]
    trainDataStates = normalizeStatesArrayArray(trainDataStates)
    testDataStates = normalizeStatesArrayArray(testDataStates)

    #Fitting function accepts array of arrays of floats
    trainTargetStates = arrayArrayToNumpyArrayArrayFloats(trainTargetStates)
    trainDataStates = arrayArrayToNumpyArrayArrayFloats(trainDataStates)
    testTargetStates = arrayArrayToNumpyArrayArrayFloats(testTargetStates)
    testDataStates = arrayArrayToNumpyArrayArrayFloats(testDataStates) 
    
    # Create neural network
    model = Sequential()
    
    #Setting up first input layer according to the input_shape
    if INPUT_DATA_SHAPE == -1:
        MODEL_LAYERS_CONFIG.insert(0,Dense(units=INPUT_UNITS, activation=INPUT_ACTIVATION, input_shape=(trainDataStates.shape[-1],), name='input_layer'))
    else:
        MODEL_LAYERS_CONFIG.insert(0,Dense(units=INPUT_UNITS, activation=INPUT_ACTIVATION, input_shape=(INPUT_DATA_SHAPE,), name='input_layer'))
    
    for layer in MODEL_LAYERS_CONFIG:
        model.add(layer)
    print(model.summary())

    # Define training parameters
    model.compile(OPTIMIZER, LOSS)

    # Perform training
    model.fit(trainDataStates, trainTargetStates, batch_size=len(trainDataStates),
              epochs=EPOCHS, shuffle=SHUFFLE, verbose=VERBOSE)

    # Save trained model to disk
    model.save(MODEL_NAME)

    # Test model (loading from disk)
    model = load_model(MODEL_NAME)
    targetPred = model.predict(testDataStates)

    # Print the number of classification errors from the training data
    nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(testTargetStates, axis=-1))
    accuracy = (len(testDataStates) - nbErrors) / len(testDataStates)
    print('Classification accuracy: %0.3f' % (accuracy))

if __name__ == "__main__":
    main()
