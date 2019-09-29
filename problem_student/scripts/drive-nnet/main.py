# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import numpy as np
import os
import sys
import time
import logging
import nnet
from keras.models import load_model

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

from sklearn.preprocessing import MinMaxScaler

###############################################
# Define constant variables here
###############################################
MIN_VALUES = list()
MAX_VALUES = list()

################################
# Define helper functions here
################################
class NNetController(object):
    def __init__(self):
        logger.info('-----------------------------------------------------------')
        logger.info('----------------LOADING TRAINED NNET MODEL-----------------')
        logger.info('-----------------------------------------------------------')
        
        self.model = load_model('nnet.h5')
        
        global MIN_VALUES, MAX_VALUES
        minMaxValuesStr = [line.rstrip('\n') for line in open(nnet.TRAINING_CONFIG_FILE)]
        MIN_VALUES = [float(element) for element in minMaxValuesStr[0].rstrip("").split(" ")]
        MAX_VALUES = [float(element) for element in minMaxValuesStr[1].rstrip("").split(" ")]
        
        self.scalerIn = MinMaxScaler()
        self.scalerIn.fit([MIN_VALUES[:-len(nnet.OUTPUT_KEYS)],MAX_VALUES[:-len(nnet.OUTPUT_KEYS)]])

        self.scalerOut = MinMaxScaler()
        self.scalerOut.fit([MIN_VALUES[-len(nnet.OUTPUT_KEYS):],MAX_VALUES[-len(nnet.OUTPUT_KEYS):]])
        
    def scale(self, data, minValue, maxValue):
        scaled = list()
        i = 0
        for x in data:
            scaled.append((x - minValue[i]) / (maxValue[i] - minValue[i]))
            i = i + 1
        return scaled
    
    def unscale(self, data, minValue, maxValue):       
        unscaled = list()
        i = 0
        for x in data:
            unscaled.append((x * (maxValue[i] - minValue[i])) + minValue[i])
            i = i + 1
        return unscaled
            
    
    def drive(self, state):      
        data = nnet.selectStatesVariablesArrayDictionary([state], nnet.CHOSEN_INPUT_KEYS)
        
        data = self.scalerIn.transform(data)
#        self.scale(data, MIN_VALUES[:-len(nnet.OUTPUT_KEYS)],MAX_VALUES[:-len(nnet.OUTPUT_KEYS)])
        prediction = self.model.predict(data)
#        self.unscale(prediction, MIN_VALUES[-len(nnet.OUTPUT_KEYS):],MAX_VALUES[-len(nnet.OUTPUT_KEYS):])
        prediction = self.scalerOut.inverse_transform(prediction)      
        
        
        
        accel = np.clip(prediction[0][0], 0.8, 1.0)     #Clips at [0,1]
        brake = np.clip(prediction[0][1], 0.0, 0.1)     #Clips at [0,1]
        gear = np.clip(prediction[0][2], 1.0, 6.0)      #Clips at [-1,6]
        steer = np.clip(prediction[0][3], -1.0, 1.0)    #Clips at [-1,1]
        
        action = {'accel': np.array([accel], dtype=np.float32),
                  'brake': np.array([brake], dtype=np.float32),
                  'gear': np.array([gear], dtype=np.int32),
                  'steer': np.array([steer], dtype=np.float32)}
        
        return action

def main():

    recordingsPath = os.path.join(CDIR, 'recordings')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    try:
        with TorcsControlEnv(render=True) as env:
            controller = NNetController()
            
            nbTracks = len(TorcsControlEnv.availableTracks)
            nbSuccessfulEpisodes = 0
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        #Select the next action based on the observation
                        action = controller.drive(observation)
                        recorder.save(observation, action)
    
                        # Execute the action
                        observation, reward, done, _ = env.step(action)
                        curNbSteps += 1
    
                        if observation and curNbSteps % nbStepsShowStats == 0:
                            curLapTime = observation['curLapTime'][0]
                            distRaced = observation['distRaced'][0]
                            logger.info('Current lap time = %4.1f sec (distance raced = %0.1f m)' % (curLapTime, distRaced))
    
                        if done:
                            if reward > 0.0:
                                logger.info('Episode was successful.')
                                nbSuccessfulEpisodes += 1
                            else:
                                logger.info('Episode was a failure.')
    
                            elapsedTime = time.time() - startTime
                            logger.info('Episode completed in %0.1f sec (computation time).' % (elapsedTime))

            logger.info('-----------------------------------------------------------')
            logger.info('Total number of successful tracks: %d (out of %d)' % (nbSuccessfulEpisodes, nbTracks))
            logger.info('-----------------------------------------------------------')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
