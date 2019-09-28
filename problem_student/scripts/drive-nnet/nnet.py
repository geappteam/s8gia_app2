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

sys.path.append('../..')
from torcs.control.core import EpisodeRecorder
CDIR = os.path.dirname(os.path.realpath(__file__))

###############################################
# Define global variables here
###############################################
### DATASETS PARAMETERS ###
TRAIN_DATASETS_PERCENTAGE = 65
EPISODE_PATHS =     [
                        'track-aalborg.pklz',
                        'track-alpine-1.pklz',
                        'track-alpine-2.pklz',
                        'track-brondehach.pklz',
                        'track-corkscrew.pklz',
                        'track-e-track-1.pklz',
                        'track-e-track-2.pklz',
                        'track-e-track-3.pklz',
                        'track-e-track-4.pklz',
                        'track-e-track-6.pklz',
                        'track-eroad.pklz',
                        'track-forza.pklz',
                        'track-g-track-1.pklz',
                        'track-g-track-2.pklz',
                        'track-g-track-3.pklz',
                        'track-ole-road-1.pklz',
                        'track-ruudskogen.pklz', 
                        'track-spring.pklz',
                        'track-street-1.pklz',
                        'track-wheel-1.pklz',
                        'track-wheel-2.pklz'
                    ]

# Selected input data :
CHOSEN_INPUT_KEYS = {
                        'angle': ['angle', 0],
                        'curLapTime': ['curLapTime', 0],
                        'damage': ['damage', 0],
                        'distFromStart': ['distFromStart', 0],
                        'distRaced': ['distRaced', 0],
                        'fuel': ['fuel', 0],
                        'gear': ['gear', 0],
                        'rpm': ['rpm', 0],
                        'speedX': ['speed', 0],
                        'speedY': ['speed', 1],
                        'track0': ['track', 0],
                        'track1': ['track', 1],
                        'track2': ['track', 2],
                        'track3': ['track', 3],
                        'track4': ['track', 4],
                        'track5': ['track', 5],
                        'track6': ['track', 6],
                        'track7': ['track', 7],
                        'track8': ['track', 8],
                        'track9': ['track', 9],
                        'track10': ['track', 10],
                        'track11': ['track', 11],
                        'track12': ['track', 12],
                        'track13': ['track', 13],
                        'track14': ['track', 14],
                        'track15': ['track', 15],
                        'track16': ['track', 16],
                        'track17': ['track', 17],
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
INPUT_UNITS = 15
INPUT_ACTIVATION = 'sigmoid'
INPUT_DATA_SHAPE = -1

#OTHER LAYERS CONFIG
MODEL_LAYERS_CONFIG =   [
                            Dense(units=20, activation='relu', name='hidden_layer'),
                            Dense(units=4, activation='sigmoid', name='output_layer')
                        ]

#COMPILATION CONFIG
OPTIMIZER = SGD(lr=0.8, momentum=0.8)
LOSS = 'mse'

#FITTING CONFIG
EPOCHS = 2000
SHUFFLE = True
VERBOSE = 1

###############################################
# Define constant variables here
###############################################
### DATASETS PARAMETERS ###
RECORDING_FOLDER_PATH = 'data'

# Output data :
OUTPUT_KEYS =   {
                    'accelCmd': ['accelCmd', 0],
                    'brakeCmd': ['brakeCmd', 0], 
                    'steerCmd': ['steerCmd', 0], 
                    'gearCmd': ['gearCmd', 0]
                }

MIN_VALUES = list()
MAX_VALUES = list()

TRAINING_CONFIG_FILE = "training.config"

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

def getMinMaxValuesStatesArrayArray(states):     
    #Find both min and max value for all given state variables 
    minValues = list()
    maxValues = list()
    
    for state in states:
        if not minValues:
            minValues = states[0]
        if not maxValues:
            maxValues = states[-1]
        index = 0
        for variable in state:
            if variable < minValues[index]:
                minValues[index] = variable
            elif variable > maxValues[index]:
                maxValues[index] = variable
            index = index + 1
        
    return minValues, maxValues

def normalizeStatesArrayArray(states, minValues, maxValues):      
    #Normalize all data values
    normalizedStates = list()
    for state in states:
        normalizedState = list()
        for variable in state:
            #Ensuring maximum and minimum value clipping
            if variable <= minValues[state.index(variable)] :
                normalizedValue = 0
            elif variable >= maxValues[state.index(variable)]:
                normalizedValue = 1
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
    for state in states:
        targetStates.append(state[-4:])
        dataStates.append(state[:-4])
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
    #Selecting and buffering chosen state variablesl
    states = selectStatesVariablesArrayDictionary(states, {**CHOSEN_INPUT_KEYS, **OUTPUT_KEYS})

    #Create a portion of states for training (second argument in the range of [0,100]) 
    #and anthoner one for testing
    trainStates, testStates = seperateTrainTestArrayArray(states, TRAIN_DATASETS_PERCENTAGE)
    
    #Train sets expected and input values
    trainTargetStates, trainDataStates = separateTargetAndDataArrayArray(trainStates)
    
    #Test sets expected and input values
    testTargetStates, testDataStates = separateTargetAndDataArrayArray(testStates)  

    #Get minimum and maximum values from datasets
    global MIN_VALUES, MAX_VALUES
    MIN_VALUES, MAX_VALUES = getMinMaxValuesStatesArrayArray(states)

    #Normalize all state variables in a range of [0,1]
    trainDataStates = normalizeStatesArrayArray(trainDataStates, MIN_VALUES[:-4], MAX_VALUES[:-4])
    testDataStates = normalizeStatesArrayArray(testDataStates, MIN_VALUES[:-4], MAX_VALUES[:-4])
    
    trainTargetStates = normalizeStatesArrayArray(trainTargetStates, MIN_VALUES[-4:], MAX_VALUES[-4:])
    testTargetStates = normalizeStatesArrayArray(testTargetStates, MIN_VALUES[-4:], MAX_VALUES[-4:])

    #Fitting function accepts numpy array of arrays of floats
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
    
    # Save minimum and mximum values in config file
    f = open(TRAINING_CONFIG_FILE, "w")
    f.write(" ".join(str(m) for m in MIN_VALUES))
    f.write("\n")
    f.write(" ".join(str(m) for m in MAX_VALUES))
    f.write("\n")
    f.close()

    # Test model (loading from disk)
    model = load_model(MODEL_NAME)
    targetPred = model.predict(testDataStates)

    # Print the number of classification errors from the training data
    nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(testTargetStates, axis=-1))
    accuracy = (len(testDataStates) - nbErrors) / len(testDataStates)
    print('Accuracy: %0.3f' % (accuracy))

if __name__ == "__main__":
    main()
