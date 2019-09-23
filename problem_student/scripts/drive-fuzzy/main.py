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

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

class FuzzyController(object):

    def __init__(self, absEnabled=True):
        self.absEnabled = absEnabled

    # usage: GEAR = calculateGear(STATE)
    #
    # Calculate the gear of the transmission for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - GEAR, the selected gear. -1 is reverse, 0 is neutral and the forward gear can range from 1 to 6.
    #

    def _calculateGear(self, state):

        # Generer l'univers de discours (universe variables)
        x_rpm = np.arange(0, 10001, 1)
        x_speed = np.arange(0, 251, 1)
        x_gear = np.arange(-1, 7, 1)

        # Generer les fonctions d'appartenance (membership functions)
        rpmL = fuzzy.trapmf(x_rpm, [0,0, 6500, 8000])
        rpmM = fuzzy.trimf(x_rpm, [6500, 8000, 9500])
        rpmH = fuzzy.trapmf(x_rpm, [8000, 9500, 10000, 10000])

        speedL = fuzzy.trimf(x_speed, [0., 0., 62.5])
        speedML = fuzzy.trimf(x_speed, [0., 62.5, 125])
        speedM = fuzzy.trimf(x_speed, [62.5, 125., 187.5])
        speedMH = fuzzy.trimf(x_speed, [125., 187.5, 250])
        speedH = fuzzy.trimf(x_speed, [187.5, 250, 250])

        gearR = fuzzy.trimf(x_gear, [-1, -1, 0])
        gearN = fuzzy.trimf(x_gear, [-1, 0, 1])
        gear1 = fuzzy.trimf(x_gear, [0, 1, 2])
        gear2 = fuzzy.trimf(x_gear, [1, 2, 3])
        gear3 = fuzzy.trimf(x_gear, [2, 3, 4])
        gear4 = fuzzy.trimf(x_gear, [3, 4, 5])
        gear5 = fuzzy.trimf(x_gear, [4, 5, 6])
        gear6 = fuzzy.trimf(x_gear, [5, 6, 6])

#        # Membership functions Graph
#        fig, (rpmMF, speedMF, gearMF) = plt.subplots(nrows=3, figsize = (10, 10))
#
#        rpmMF.plot(x_rpm, rpmL, 'b', linewidth = 1.5, label = 'Low RPM')
#        rpmMF.plot(x_rpm, rpmM, 'g', linewidth = 1.5, label = 'Medium RPM')
#        rpmMF.plot(x_rpm, rpmH, 'r', linewidth = 1.5, label = 'Hgh RPM')
#        rpmMF.set_title('RPM')
#        rpmMF.legend()
#
#        speedMF.plot(x_speed, speedL, 'b', linewidth = 1.5, label = 'Low speed')
#        speedMF.plot(x_speed, speedML, 'y', linewidth = 1.5, label = 'Medium Low speed')
#        speedMF.plot(x_speed, speedM, 'g', linewidth = 1.5, label = 'Medium speed')
#        speedMF.plot(x_speed, speedMH, 'c', linewidth = 1.5, label = 'Medium High speed')
#        speedMF.plot(x_speed, speedH, 'r', linewidth = 1.5, label = 'Hgh speed')
#        speedMF.set_title('Speed')
#        speedMF.legend()
#
#        gearMF.plot(x_gear, gearR, 'b', linewidth = 1.5, label = 'Reverse')
#        gearMF.plot(x_gear, gearN, 'y', linewidth = 1.5, label = 'Neutral')
#        gearMF.plot(x_gear, gear1, 'g', linewidth = 1.5, label = '1st')
#        gearMF.plot(x_gear, gear2, 'm', linewidth = 1.5, label = '2nd')
#        gearMF.plot(x_gear, gear3, 'r', linewidth = 1.5, label = '3rd')
#        gearMF.plot(x_gear, gear4, 'g', linewidth = 1.5, label = '4th')
#        gearMF.plot(x_gear, gear5, 'k', linewidth = 1.5, label = '5th')
#        gearMF.plot(x_gear, gear6, 'r', linewidth = 1.5, label = '6th')
#        gearMF.set_title('Gear')
#        gearMF.legend()
#
#        for graph in (rpmMF, speedMF, gearMF):
#            graph.spines['top'].set_visible(False)
#            graph.spines['right'].set_visible(False)
#            graph.get_xaxis().tick_bottom()
#            graph.get_yaxis().tick_left()
#
#        plt.tight_layout()

        # Get current crisp values
        curGear = state['gear'][0]
        curRpm = state['rpm'][0]
        curSpeed = np.sqrt(np.sum(np.power(state['speed'], 2))) # Total speed is calculated using pythagoras

        # Determin the membership value of current input values for each linguistic value
        rpmLevelL = fuzzy.interp_membership(x_rpm, rpmL, curRpm)
        rpmLevelM = fuzzy.interp_membership(x_rpm, rpmM, curRpm)
        rpmLevelH = fuzzy.interp_membership(x_rpm, rpmH, curRpm)

        speedLevelL = fuzzy.interp_membership(x_speed, speedL, curSpeed)
        speedLevelML = fuzzy.interp_membership(x_speed, speedML, curSpeed)
        speedLevelM = fuzzy.interp_membership(x_speed, speedM, curSpeed)
        speedLevelMH = fuzzy.interp_membership(x_speed, speedMH, curSpeed)
        speedLevelH = fuzzy.interp_membership(x_speed, speedH, curSpeed)

        # Rules
        #If speed is L and rpm is H, then gear is: 1
        #If speed is ML and rpm is H, then gear is: 2
        #If speed is M and rpm is H, then gear is: 3
        #If speed is MH and rpm is H, then gear is: 4
        #If speed is H and rpm is H, then gear is: 5
        activationRule1 = np.fmin(speedLevelL, rpmLevelH)
        gearActivation1 = np.fmin(activationRule1, gear1)

        activationRule2 = np.fmin(speedLevelML, rpmLevelH)
        gearActivation2 = np.fmin(activationRule2, gear2)

        activationRule3 = np.fmin(speedLevelM, rpmLevelH)
        gearActivation3 = np.fmin(activationRule3, gear3)

        activationRule4 = np.fmin(speedLevelMH, rpmLevelH)
        gearActivation4 = np.fmin(activationRule4, gear4)

        activationRule5 = np.fmin(speedLevelH, rpmLevelH)
        gearActivation5 = np.fmin(activationRule5, gear5)


#        fig, (rpmMF) = plt.subplots(figsize=(8,3))
#
#        gear0 = np.zeros_like(x_gear)
#
#        rpmMF.fill_between(x_gear, gear0, gearActivation1, facecolor = 'b', alpha = 0.7)
#        rpmMF.plot(x_gear, gear1, 'b', linewidth = 0.5, linestyle = '--')
#        rpmMF.fill_between(x_gear, gear0, gearActivation2, facecolor = 'r', alpha = 0.7)
#        rpmMF.plot(x_gear, gear2, 'r', linewidth = 0.5, linestyle = '--')
#        rpmMF.fill_between(x_gear, gear0, gearActivation3, facecolor = 'k', alpha = 0.7)
#        rpmMF.plot(x_gear, gear3, 'k', linewidth = 0.5, linestyle = '--')
#        rpmMF.fill_between(x_gear, gear0, gearActivation4, facecolor = 'g', alpha = 0.7)
#        rpmMF.plot(x_gear, gear4, 'g', linewidth = 0.5, linestyle = '--')
#        rpmMF.fill_between(x_gear, gear0, gearActivation5, facecolor = 'y', alpha = 0.7)
#        rpmMF.plot(x_gear, gear5, 'y', linewidth = 0.5, linestyle = '--')

        try:
            aggregation = np.fmax(gearActivation1, np.fmax(gearActivation3, np.fmax(gearActivation4, gearActivation5)))
            nextGear = fuzzy.defuzz(x_gear, aggregation, 'centroid')
        except:
            nextGear = 1

#        nextGearActivation = fuzzy.interp_membership(x_gear, aggregation, nextGear)

        return nextGear

    # usage: STEERING = calculateSteering(STATE)
    #
    # Calculate the steering value for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - STEERING, the steering value. -1 and +1 means respectively full left and right, that corresponds to an angle of 0.785398 rad.
    #
    def _calculateSteering(self, state):
        # Steering constants
        steerLock = 0.785398
        steerSensitivityOffset = 80.0
        wheelSensitivityCoeff = 1.0

        curAngle = state['angle'][0]
        curTrackPos = state['trackPos'][0]
        curSpeedX = state['speed'][0]

        # Steering angle is computed by correcting the actual car angle w.r.t. to track
        # axis and to adjust car position w.r.t to middle of track
        targetAngle = curAngle - curTrackPos * 2.0

        # At high speed, reduce the steering command to avoid loosing control
        if curSpeedX > steerSensitivityOffset:
            steering = targetAngle / (steerLock * (curSpeedX - steerSensitivityOffset) * wheelSensitivityCoeff)
        else:
            steering = targetAngle / steerLock

        # Normalize steering
        steering = np.clip(steering, -1.0, 1.0)

        return steering

    # usage: ACCELERATION = calculateAcceleration(STATE)
    #
    # Calculate the accelerator (gas pedal) value for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - ACCELERATION, the virtual gas pedal (0 means no gas, 1 full gas), in the range [0,1].
    #
    def _calculateAcceleration(self, state):

        # Accel and Brake Constants
        maxSpeedDist = 95.0
        maxSpeed = 100.0
        sin10 = 0.17365
        cos10 = 0.98481
        angleSensitivity = 2.0

        curSpeedX = state['speed'][0]
        curTrackPos = state['trackPos'][0]

        # checks if car is out of track
        if (curTrackPos < 1 and curTrackPos > -1):

            # Reading of sensor at +10 degree w.r.t. car axis
            rxSensor = state['track'][8]
            # Reading of sensor parallel to car axis
            cSensor = state['track'][9]
            # Reading of sensor at -5 degree w.r.t. car axis
            sxSensor = state['track'][10]

            # Track is straight and enough far from a turn so goes to max speed
            if cSensor > maxSpeedDist or (cSensor >= rxSensor and cSensor >= sxSensor):
                targetSpeed = maxSpeed
            else:
                # Approaching a turn on right
                if rxSensor > sxSensor:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = rxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))

                # Approaching a turn on left
                else:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = sxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))

                # Estimate the target speed depending on turn and on how close it is
                targetSpeed = maxSpeed * (cSensor * np.sin(angle) / maxSpeedDist) * angleSensitivity
                targetSpeed = np.clip(targetSpeed, 0.0, maxSpeed)

            # Accel/brake command is exponentially scaled w.r.t. the difference
            # between target speed and current one
            accel = (2.0 / (1.0 + np.exp(curSpeedX - targetSpeed)) - 1.0)

        else:
            # when out of track returns a moderate acceleration command
            accel = 0.3

        if accel > 0:
            accel = accel
            brake = 0.0
        else:
            brake = -accel
            accel = 0.0

            if self.absEnabled:
                # apply ABS to brake
                brake = self._filterABS(state, brake)

        brake = np.clip(brake, 0.0, 1.0)
        accel = np.clip(accel, 0.0, 1.0)

        return accel, brake

    def _filterABS(self, state, brake):

        wheelRadius = [0.3179, 0.3179, 0.3276, 0.3276]
        absSlip = 2.0
        absRange = 3.0
        absMinSpeed = 3.0

        curSpeedX = state['speed'][0]

        # convert speed to m/s
        speed = curSpeedX / 3.6

        # when speed lower than min speed for abs do nothing
        if speed >= absMinSpeed:
            # compute the speed of wheels in m/s
            slip = np.dot(state['wheelSpinVel'], wheelRadius)
            # slip is the difference between actual speed of car and average speed of wheels
            slip = speed - slip / 4.0
            # when slip too high apply ABS
            if slip > absSlip:
                brake = brake - (slip - absSlip) / absRange

            # check brake is not negative, otherwise set it to zero
            brake = np.clip(brake, 0.0, 1.0)

        return brake

    # usage: ACTION = drive(STATE)
    #
    # Calculate the accelerator, brake, gear and steering values based on the current car state.
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - ACTION, the structure describing the action to execute (see function 'applyAction').
    #
    def drive(self, state):
        accel, brake = self._calculateAcceleration(state)
        gear = self._calculateGear(state)
        steer = self._calculateSteering(state)

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
            controller = FuzzyController()
            
#            nbTracks = len(TorcsControlEnv.availableTracks)
            nbTracks = 1
            nbSuccessfulEpisodes = 0
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
                doIt = True
                
                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        # Select the next action based on the observation
                        
                        if(doIt):
                            print("Observation: ", observation)
                        action = controller.drive(observation)
                        if(doIt):
                            print("Action: ", action)
                            doIt = False
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
