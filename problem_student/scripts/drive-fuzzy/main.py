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

    printFuzzyLogic = True
    printGear = True
    printAccel = True
    printSteering = True
    printInterval = 0

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

    def _recursiveAggregation(self, activationResults):

        if not isinstance(activationResults, list):
            raise ValueError("activationResults should be a list of lists")
        else:
            if len(activationResults) == 2:
                return np.fmax(activationResults[0], activationResults[1])
            else:
                _tmpRes = activationResults[0]
                del activationResults[0]
                return np.fmax(_tmpRes, self._recursiveAggregation(activationResults))

    def _calculateGear(self, state):

        # Generer l'univers de discours (universe variables)
        x_rpm = np.arange(0, 10001, 1)
        x_speed = np.arange(0, 251, 1)
        x_gear = np.arange(-1, 7, 1)

        # Generer les fonctions d'appartenance (membership functions)
        rpmL = fuzzy.trapmf(x_rpm, [0, 0, 6500, 8000])
        rpmM = fuzzy.trimf(x_rpm, [6500, 8000, 9500])
        rpmH = fuzzy.trapmf(x_rpm, [8000, 9500, 10000, 10000])

        speedL = fuzzy.trimf(x_speed, [0., 0., 62.5])
        speedML = fuzzy.trimf(x_speed, [0., 62.5, 125.])
        speedM = fuzzy.trimf(x_speed, [62.5, 125., 187.5])
        speedMH = fuzzy.trimf(x_speed, [125., 187.5, 250.])
        speedH = fuzzy.trimf(x_speed, [187.5, 250., 250.])

        gear1 = fuzzy.trimf(x_gear, [0, 1, 2])
        gear2 = fuzzy.trimf(x_gear, [1, 2, 3])
        gear3 = fuzzy.trimf(x_gear, [2, 3, 4])
        gear4 = fuzzy.trimf(x_gear, [3, 4, 5])
        gear5 = fuzzy.trimf(x_gear, [4, 5, 6])

        # Get current crisp values
#        curGear = state['gear'][0]
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
        activationRule1 = np.fmin(speedLevelL, rpmLevelL) # If SpeedLevel is Low and rpm Level is Low, ... (1/2)
        activationRule2 = np.fmin(speedLevelL, rpmLevelM)
        activationRule3 = np.fmin(speedLevelL, rpmLevelH)
        activationRule4 = np.fmin(speedLevelML, rpmLevelL)
        activationRule5 = np.fmin(speedLevelML, rpmLevelM)
        activationRule6 = np.fmin(speedLevelML, rpmLevelH)
        activationRule7 = np.fmin(speedLevelM, rpmLevelL)
        activationRule8 = np.fmin(speedLevelM, rpmLevelM)
        activationRule9 = np.fmin(speedLevelM, rpmLevelH)
        activationRule10 = np.fmin(speedLevelMH, rpmLevelL)
        activationRule11 = np.fmin(speedLevelMH, rpmLevelM)
        activationRule12 = np.fmin(speedLevelMH, rpmLevelH)
        activationRule13 = np.fmin(speedLevelH, rpmLevelL)
        activationRule14 = np.fmin(speedLevelH, rpmLevelM)
        activationRule15 = np.fmin(speedLevelH, rpmLevelH)

        gearActivation1 = np.fmin(activationRule1, gear1) # ... then gear is 1 (2/2)
        gearActivation2 = np.fmin(activationRule2, gear1)
        gearActivation3 = np.fmin(activationRule3, gear2)
        gearActivation4 = np.fmin(activationRule4, gear2)
        gearActivation5 = np.fmin(activationRule5, gear2)
        gearActivation6 = np.fmin(activationRule6, gear3)
        gearActivation7 = np.fmin(activationRule7, gear3)
        gearActivation8 = np.fmin(activationRule8, gear3)
        gearActivation9 = np.fmin(activationRule9, gear4)
        gearActivation10 = np.fmin(activationRule10, gear4)
        gearActivation11 = np.fmin(activationRule11, gear4)
        gearActivation12 = np.fmin(activationRule12, gear5)
        gearActivation13 = np.fmin(activationRule13, gear5)
        gearActivation14 = np.fmin(activationRule14, gear5)
        gearActivation15 = np.fmin(activationRule15, gear5)

        aggregationVector = list()

        aggregationVector.append(gearActivation1)
        aggregationVector.append(gearActivation2)
        aggregationVector.append(gearActivation3)
        aggregationVector.append(gearActivation4)
        aggregationVector.append(gearActivation5)
        aggregationVector.append(gearActivation6)
        aggregationVector.append(gearActivation7)
        aggregationVector.append(gearActivation8)
        aggregationVector.append(gearActivation9)
        aggregationVector.append(gearActivation10)
        aggregationVector.append(gearActivation11)
        aggregationVector.append(gearActivation12)
        aggregationVector.append(gearActivation13)
        aggregationVector.append(gearActivation14)
        aggregationVector.append(gearActivation15)

        aggregation = self._recursiveAggregation(aggregationVector)

        nextGear = fuzzy.defuzz(x_gear, aggregation, 'centroid')
        nextGearActivation = fuzzy.interp_membership(x_gear, aggregation, nextGear) # for plot

        if(self.printFuzzyLogic == True and self.printGear == True):

            # Membership functions Graph
            figFuzzyMemberships, (rpmMF, speedMF, gearMF, rpmActivation, gearAggregationFig) = plt.subplots(nrows=5, figsize = (10, 10))

            rpmMF.plot(x_rpm, rpmL, 'b', linewidth = 1.5, label = 'Low RPM')
            rpmMF.plot(x_rpm, rpmM, 'g', linewidth = 1.5, label = 'Medium RPM')
            rpmMF.plot(x_rpm, rpmH, 'r', linewidth = 1.5, label = 'Hgh RPM')
            rpmMF.set_title('RPM')
            rpmMF.legend()

            speedMF.plot(x_speed, speedL, 'b', linewidth = 1.5, label = 'Low speed')
            speedMF.plot(x_speed, speedML, 'y', linewidth = 1.5, label = 'Medium Low speed')
            speedMF.plot(x_speed, speedM, 'g', linewidth = 1.5, label = 'Medium speed')
            speedMF.plot(x_speed, speedMH, 'c', linewidth = 1.5, label = 'Medium High speed')
            speedMF.plot(x_speed, speedH, 'r', linewidth = 1.5, label = 'Hgh speed')
            speedMF.set_title('Speed')
            speedMF.legend()

#            gearMF.plot(x_gear, gearR, 'b', linewidth = 1.5, label = 'Reverse')
#            gearMF.plot(x_gear, gearN, 'y', linewidth = 1.5, label = 'Neutral')
            gearMF.plot(x_gear, gear1, 'g', linewidth = 1.5, label = '1st')
            gearMF.plot(x_gear, gear2, 'm', linewidth = 1.5, label = '2nd')
            gearMF.plot(x_gear, gear3, 'r', linewidth = 1.5, label = '3rd')
            gearMF.plot(x_gear, gear4, 'g', linewidth = 1.5, label = '4th')
            gearMF.plot(x_gear, gear5, 'k', linewidth = 1.5, label = '5th')
#            gearMF.plot(x_gear, gear6, 'r', linewidth = 1.5, label = '6th')
            gearMF.set_title('Gear')
            gearMF.legend()
    
            for graph in (rpmMF, speedMF, gearMF):
                graph.spines['top'].set_visible(False)
                graph.spines['right'].set_visible(False)
                graph.get_xaxis().tick_bottom()
                graph.get_yaxis().tick_left()

            gear0 = np.zeros_like(x_gear)

            rpmActivation.fill_between(x_gear, gear0, gearActivation1, facecolor = 'b', alpha = 0.7)
            rpmActivation.plot(x_gear, gear1, 'b', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation2, facecolor = 'r', alpha = 0.7)
            rpmActivation.plot(x_gear, gear1, 'r', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation3, facecolor = 'k', alpha = 0.7)
            rpmActivation.plot(x_gear, gear2, 'k', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation4, facecolor = 'g', alpha = 0.7)
            rpmActivation.plot(x_gear, gear2, 'g', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation5, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear2, 'b', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation6, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear3, 'k', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation7, facecolor = 'g', alpha = 0.7)
            rpmActivation.plot(x_gear, gear3, 'r', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation8, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear3, 'y', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation9, facecolor = 'g', alpha = 0.7)
            rpmActivation.plot(x_gear, gear4, 'g', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation10, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear4, 'k', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation11, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear4, 'b', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation12, facecolor = 'g', alpha = 0.7)
            rpmActivation.plot(x_gear, gear5, 'g', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation13, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear5, 'y', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation14, facecolor = 'g', alpha = 0.7)
            rpmActivation.plot(x_gear, gear5, 'r', linewidth = 0.5, linestyle = '--')
            rpmActivation.fill_between(x_gear, gear0, gearActivation15, facecolor = 'y', alpha = 0.7)
            rpmActivation.plot(x_gear, gear5, 'y', linewidth = 0.5, linestyle = '--')
            rpmActivation.set_title('Gear membership')

            gearAggregationFig.plot(x_gear, gear1, 'b', linewidth = 0.5, linestyle = '--')
            gearAggregationFig.plot(x_gear, gear2, 'g', linewidth = 0.5, linestyle = '--')
            gearAggregationFig.plot(x_gear, gear3, 'r', linewidth = 0.5, linestyle = '--')
            gearAggregationFig.plot(x_gear, gear4, 'y', linewidth = 0.5, linestyle = '--')
            gearAggregationFig.plot(x_gear, gear5, 'k', linewidth = 0.5, linestyle = '--')

            gearAggregationFig.fill_between(x_gear, gear0, aggregation, facecolor = 'Orange', alpha = 0.7)
            gearAggregationFig.plot([nextGear, nextGear], [0, nextGearActivation], 'k', linewidth = 1.5, alpha = 0.9)
            gearAggregationFig.set_title('Aggregated gear membership')

            for fig in (rpmMF, speedMF, gearMF, rpmMF, gearAggregationFig):
                fig.spines['top'].set_visible(False)
                fig.spines['right'].set_visible(False)
                fig.get_xaxis().tick_bottom()
                fig.get_yaxis().tick_left()

            plt.tight_layout()

        if(nextGear == 0):
            nextGear+=1

        return (int)(np.ceil(nextGear))

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

        # Universe variables
        x_angle = np.arange(-np.pi/2, np.pi/2, np.pi/180)
        x_position = np.arange(-2., 2., 0.01)
        x_steering = np.arange(-1., 1., 0.01)

        angleL = fuzzy.trapmf(x_angle, [-np.pi/2, -np.pi/2, -np.pi/9, 0])
        angleC = fuzzy.trimf(x_angle, [-np.pi/9, 0., np.pi/9])
        angleR = fuzzy.trapmf(x_angle, [0., np.pi/9, np.pi/2, np.pi/2])

        positionL = fuzzy.trapmf(x_position, [-2., -2., -0.50, 0.])
        positionC = fuzzy.trimf(x_position, [ -0.50, 0., 0.50])
        positionR = fuzzy.trapmf(x_position, [ 0., .50, 2., 2.])

        steeringL = fuzzy.trimf(x_steering, [ -1., -1., -0.3])
        steeringML = fuzzy.trimf(x_steering, [ -0.6, -0.05, 0.])
        steeringC = fuzzy.trimf(x_steering, [-0.05, 0., 0.05])
        steeringMR = fuzzy.trimf(x_steering, [0., 0.05, 0.6])
        steeringR = fuzzy.trimf(x_steering, [0.3, 1., 1.])

        # Hedges
        angleL = np.power(angleL, 2)
        angleC = np.sqrt(angleC)
        angleR = np.power(angleR, 2)

#        positionC = np.sqrt(positionC)

        steeringL = np.power(steeringL, 2)
        steeringML = np.power(steeringML, 2)
        steeringC = np.power(steeringC, 2)
        steeringMR = np.power(steeringMR, 2)
        steeringR = np.power(steeringR, 2)

        curAngle = state['angle'][0]
        curTrackPos = -state['trackPos'][0] # track position has to be inverted to match fuzzy set

        angleLevelL = fuzzy.interp_membership(x_angle, angleL, curAngle)
        angleLevelC = fuzzy.interp_membership(x_angle, angleC, curAngle)
        angleLevelR = fuzzy.interp_membership(x_angle, angleR, curAngle)

        positionLevelL = fuzzy.interp_membership(x_position, positionL, curTrackPos)
        positionLevelC = fuzzy.interp_membership(x_position, positionC, curTrackPos)
        positionLevelR = fuzzy.interp_membership(x_position, positionR, curTrackPos)

        activationRule1 =  np.fmin( positionLevelL, angleLevelL)
        activationRule2 =  np.fmin( positionLevelL, angleLevelC)
        activationRule3 =  np.fmin( positionLevelL, angleLevelR)

        activationRule4 =  np.fmin( positionLevelC, angleLevelL)
        activationRule5 =  np.fmin( positionLevelC, angleLevelC)
        activationRule6 =  np.fmin( positionLevelC, angleLevelR)

        activationRule7 =  np.fmin( positionLevelR, angleLevelL)
        activationRule8 =  np.fmin( positionLevelR, angleLevelC)
        activationRule9 =  np.fmin( positionLevelR, angleLevelR)

        steeringActivation1 = np.fmin(activationRule1, steeringR)
        steeringActivation2 = np.fmin(activationRule2, steeringR)
        steeringActivation3 = np.fmin(activationRule3, steeringC)

        steeringActivation4 = np.fmin(activationRule4, steeringMR)
        steeringActivation5 = np.fmin(activationRule5, steeringC)
        steeringActivation6 = np.fmin(activationRule6, steeringML)

        steeringActivation7 = np.fmin(activationRule7, steeringR)
        steeringActivation8 = np.fmin(activationRule8, steeringL)
        steeringActivation9 = np.fmin(activationRule9, steeringL)

        aggregationVector = list()

        aggregationVector.append(steeringActivation1)
        aggregationVector.append(steeringActivation2)
        aggregationVector.append(steeringActivation3)
        aggregationVector.append(steeringActivation4)
        aggregationVector.append(steeringActivation5)
        aggregationVector.append(steeringActivation6)
        aggregationVector.append(steeringActivation7)
        aggregationVector.append(steeringActivation8)
        aggregationVector.append(steeringActivation9)

        aggregation = self._recursiveAggregation(aggregationVector)

        if(not all(v == 0 for v in aggregation)):
            nextSteering = fuzzy.defuzz(x_steering, aggregation, 'centroid')
        else:
            nextSteering = 0

        nextSteeringActivation = fuzzy.interp_membership(x_steering, aggregation, nextSteering) # for plot

        if(self.printFuzzyLogic == True and self.printSteering):

            # Membership functions Graph
            figFuzzyMemberships, (angleFig, positionFig, steeringFig, accelActivationFig, steeringAggregationFig) = plt.subplots(nrows = 5, figsize = (10, 10))

            angleFig.plot(x_angle, angleL, 'b', linewidth = 1.5, label = 'Left angle')
            angleFig.plot(x_angle, angleC, 'g', linewidth = 1.5, label = 'Center angle')
            angleFig.plot(x_angle, angleR, 'r', linewidth = 1.5, label = 'Right angle')
            angleFig.set_title('Angle with respect to road axis')
            angleFig.legend()

            positionFig.plot(x_position, positionL, 'b', linewidth = 1.5, label = 'Left position')
            positionFig.plot(x_position, positionC, 'g', linewidth = 1.5, label = 'Center position')
            positionFig.plot(x_position, positionR, 'r', linewidth = 1.5, label = 'Right position')
            positionFig.set_title('Track Position')
            positionFig.legend()

            steeringFig.plot(x_steering, steeringL, 'b', linewidth = 1.5, label = 'Left steering')
            steeringFig.plot(x_steering, steeringML, 'k', linewidth = 1.5, label = 'Medium Left steering')
            steeringFig.plot(x_steering, steeringC, 'y', linewidth = 1.5, label = 'Center steering')
            steeringFig.plot(x_steering, steeringMR, 'R', linewidth = 1.5, label = 'Medium Right steering')
            steeringFig.plot(x_steering, steeringR, 'g', linewidth = 1.5, label = 'Right steering')
            steeringFig.set_title('Steering')
            steeringFig.legend()

            steering0 = np.zeros_like(x_steering)

            accelActivationFig.fill_between(x_steering, steering0, steeringActivation1, facecolor = 'b', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringMR, 'b', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation2, facecolor = 'r', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringMR, 'r', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation3, facecolor = 'k', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringC, 'k', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation4, facecolor = 'g', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringR, 'g', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation5, facecolor = 'y', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringC, 'b', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation6, facecolor = 'y', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringL, 'k', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation7, facecolor = 'g', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringR, 'r', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation8, facecolor = 'y', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringML, 'y', linewidth = 0.5, linestyle = '--')
            accelActivationFig.fill_between(x_steering, steering0, steeringActivation9, facecolor = 'g', alpha = 0.7)
            accelActivationFig.plot(x_steering, steeringML, 'g', linewidth = 0.5, linestyle = '--')

            accelActivationFig.set_title('Steering membership')

            steeringAggregationFig.plot(x_steering, steeringL, 'b', linewidth = 0.5, linestyle = '--')
            steeringAggregationFig.plot(x_steering, steeringML, 'k', linewidth = 0.5, linestyle = '--')
            steeringAggregationFig.plot(x_steering, steeringC, 'g', linewidth = 0.5, linestyle = '--')
            steeringAggregationFig.plot(x_steering, steeringMR, 'y', linewidth = 0.5, linestyle = '--')
            steeringAggregationFig.plot(x_steering, steeringR, 'r', linewidth = 0.5, linestyle = '--')

            steeringAggregationFig.fill_between(x_steering, steering0, aggregation, facecolor = 'Orange', alpha = 0.7)
            steeringAggregationFig.plot([nextSteering, nextSteering], [0, nextSteeringActivation], 'k', linewidth = 1.5, alpha = 0.9)
            steeringAggregationFig.set_title('Aggregated steering membership')

            for fig in (angleFig, positionFig, steeringFig, accelActivationFig, steeringAggregationFig):
                fig.spines['top'].set_visible(False)
                fig.spines['right'].set_visible(False)
                fig.get_xaxis().tick_bottom()
                fig.get_yaxis().tick_left()

            plt.tight_layout()

        # Normalize steering
        steering = np.clip(nextSteering, -1.0, 1.0)
        
        return -steering

    # usage: ACCELERATION = calculateAcceleration(STATE)
    #
    # Calculate the accelerator (gas pedal) value for the current car state.
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - ACCELERATION, the virtual gas pedal (0 means no gas, 1 full gas), in the range [0,1].
    #
    def _calculateAcceleration(self, state):

        curSpeedX = state['speed'][0]

        # Reading of sensor at +10 degree w.r.t. car axis
#            rxSensor = state['track'][8]
        # Reading of sensor parallel to car axis
        cSensor = np.average([state['track'][9], np.sqrt(np.sum(np.power([state['track'][8], state['track'][10]], 1.5)))])
        # Reading of sensor at -5 degree w.r.t. car axis
#            sxSensor = state['track'][10]

        # Universe of discours variables
        x_straightDistance = np.arange(0,101,1)
        x_speed = np.arange(0, 251, 1)
        x_accel = np.arange(-1., 1, 0.01)
#        x_brake = np.arange(0, 1.01, 0.01)

        # Fuzzy membership functions
        straightDistanceC = fuzzy.trimf(x_straightDistance, [0, 0, 30])
        straightDistanceN = fuzzy.trimf(x_straightDistance, [25, 50, 75])
        straightDistanceF = fuzzy.trimf(x_straightDistance, [50, 100, 100])

        speedL = fuzzy.trimf(x_speed, [0., 0., 25])
        speedML = fuzzy.trimf(x_speed, [0., 25, 50.])
        speedM = fuzzy.trimf(x_speed, [25, 50., 187.5])
        speedMH = fuzzy.trimf(x_speed, [50., 75., 250.])
        speedH = fuzzy.trimf(x_speed, [75., 250., 250.])

        accelNH = fuzzy.trimf(x_accel, [-1., -1., -0.5])
        accelNM = fuzzy.trimf(x_accel, [-1., -0.5, 0.])
        accelNULL = fuzzy.trimf(x_accel, [-0.5, 0., 0.5])
        accelPM = fuzzy.trimf(x_accel, [0., 0.5, 1.])
        accelPH = fuzzy.trimf(x_accel, [0.5, 1., 1.])

        # Hedges
        accelNH = np.power(accelNH, 4)
        accelNM = np.power(accelNM, 4)
        accelNULL = np.power(accelNULL, 4)
        accelPM = np.power(accelPM, 4)
        accelPH = np.power(accelPH, 4)

        straightDistanceC = np.power(straightDistanceC, 2)
        straightDistanceN = np.power(straightDistanceN, 2)
        straightDistanceF = np.power(straightDistanceF, 2)

        speedLevelL = fuzzy.interp_membership(x_speed, speedL, curSpeedX)
        speedLevelML = fuzzy.interp_membership(x_speed, speedML, curSpeedX)
        speedLevelM = fuzzy.interp_membership(x_speed, speedM, curSpeedX)
        speedLevelMH = fuzzy.interp_membership(x_speed, speedMH, curSpeedX)
        speedLevelH = fuzzy.interp_membership(x_speed, speedH, curSpeedX)

        straightDistanceLevelC = fuzzy.interp_membership(x_straightDistance, straightDistanceC, cSensor)
        straightDistanceLevelN = fuzzy.interp_membership(x_straightDistance, straightDistanceN, cSensor)
        straightDistanceLevelF = fuzzy.interp_membership(x_straightDistance, straightDistanceF, cSensor)

        # Rules accel
        activationRule1 = np.fmin(speedLevelL, straightDistanceLevelC)
        activationRule2 = np.fmin(speedLevelL, straightDistanceLevelN)
        activationRule3 = np.fmin(speedLevelL, straightDistanceLevelF)
        
        activationRule4 = np.fmin(speedLevelML, straightDistanceLevelC)
        activationRule5 = np.fmin(speedLevelML, straightDistanceLevelN)
        activationRule6 = np.fmin(speedLevelML, straightDistanceLevelF)
        
        activationRule7 = np.fmin(speedLevelM, straightDistanceLevelC)
        activationRule8 = np.fmin(speedLevelM, straightDistanceLevelN)
        activationRule9 = np.fmin(speedLevelM, straightDistanceLevelF)
        
        activationRule10 = np.fmin(speedLevelMH, straightDistanceLevelC)
        activationRule11 = np.fmin(speedLevelMH, straightDistanceLevelN)
        activationRule12 = np.fmin(speedLevelMH, straightDistanceLevelF)
        
        activationRule13 = np.fmin(speedLevelH, straightDistanceLevelC)
        activationRule14 = np.fmin(speedLevelH, straightDistanceLevelN)
        activationRule15 = np.fmin(speedLevelH, straightDistanceLevelF)

        accelActivation1 = np.fmin(activationRule1, accelNULL)
        accelActivation2 = np.fmin(activationRule2, accelPM)
        accelActivation3 = np.fmin(activationRule3, accelPH)
        
        accelActivation4 = np.fmin(activationRule4, accelNULL)
        accelActivation5 = np.fmin(activationRule5, accelPM)
        accelActivation6 = np.fmin(activationRule6, accelPH)
        
        accelActivation7 = np.fmin(activationRule7, accelNULL)
        accelActivation8 = np.fmin(activationRule8, accelPH)
        accelActivation9 = np.fmin(activationRule9, accelPH)
        
        accelActivation10 = np.fmin(activationRule10, accelNH)
        accelActivation11 = np.fmin(activationRule11, accelPM)
        accelActivation12 = np.fmin(activationRule12, accelPH)
        
        accelActivation13 = np.fmin(activationRule13, accelNH)
        accelActivation14 = np.fmin(activationRule14, accelNM)
        accelActivation15 = np.fmin(activationRule15, accelPH)

        aggregationVector = list()

        aggregationVector.append(accelActivation1)
        aggregationVector.append(accelActivation2)
        aggregationVector.append(accelActivation3)
        aggregationVector.append(accelActivation4)
        aggregationVector.append(accelActivation5)
        aggregationVector.append(accelActivation6)
        aggregationVector.append(accelActivation7)
        aggregationVector.append(accelActivation8)
        aggregationVector.append(accelActivation9)
        aggregationVector.append(accelActivation10)
        aggregationVector.append(accelActivation11)
        aggregationVector.append(accelActivation12)
        aggregationVector.append(accelActivation13)
        aggregationVector.append(accelActivation14)
        aggregationVector.append(accelActivation15)

        aggregation = self._recursiveAggregation(aggregationVector)
        if(not all(v == 0 for v in aggregation)):
            accel = fuzzy.defuzz(x_accel, aggregation, 'centroid')
        else:
            accel = 0

        nextAccelActivation = fuzzy.interp_membership(x_accel, aggregation, accel)

        if(accel<0):
            brake = np.abs(accel)
            accel = 0
        else:
            brake = 0

        if(self.printFuzzyLogic == True and self.printAccel == True):

            figMembershipFctAccel, ((straightDistanceMF, speedMF, AccelMF), (accelFig, accelMembershipFig, accelAggregationFig)) = plt.subplots(nrows=2, ncols=3, figsize = (10, 10))

            straightDistanceMF.plot(x_straightDistance, straightDistanceC, 'b', linewidth = 1.5, label = 'Close')
            straightDistanceMF.plot(x_straightDistance, straightDistanceN, 'g', linewidth = 1.5, label = 'Near')
            straightDistanceMF.plot(x_straightDistance, straightDistanceF, 'r', linewidth = 1.5, label = 'Far')
            straightDistanceMF.set_title('Straight Distance')
            straightDistanceMF.legend()

            speedMF.plot(x_speed, speedL, 'b', linewidth = 1.5, label = 'Low speed')
            speedMF.plot(x_speed, speedML, 'y', linewidth = 1.5, label = 'Medium Low speed')
            speedMF.plot(x_speed, speedM, 'g', linewidth = 1.5, label = 'Medium speed')
            speedMF.plot(x_speed, speedMH, 'c', linewidth = 1.5, label = 'Medium High speed')
            speedMF.plot(x_speed, speedH, 'r', linewidth = 1.5, label = 'Hgh speed')
            speedMF.set_title('Speed')
            speedMF.legend()

            AccelMF.plot(x_accel, accelNH, 'b', linewidth = 1.5, label = 'Low')
            AccelMF.plot(x_accel, accelNM, 'g', linewidth = 1.5, label = 'Medium')
            AccelMF.plot(x_accel, accelNULL, 'r', linewidth = 1.5, label = 'High')
            AccelMF.plot(x_accel, accelPM, 'g', linewidth = 1.5, label = 'Medium')
            AccelMF.plot(x_accel, accelPH, 'r', linewidth = 1.5, label = 'High')
            AccelMF.set_title('Accel.')
            AccelMF.legend()

            accel0 = np.zeros_like(x_accel)

            accelFig.fill_between(x_accel, accel0, accelActivation1, facecolor = 'b', alpha = 0.7)
            accelFig.plot(x_accel, accelNULL, 'b',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation2, facecolor = 'r', alpha = 0.7)
            accelFig.plot(x_accel, accelPM, 'r',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation3, facecolor = 'y', alpha = 0.7)
            accelFig.plot(x_accel, accelPH, 'y',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation4, facecolor = 'g', alpha = 0.7)
            accelFig.plot(x_accel, accelNM, 'g',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation5, facecolor = 'w', alpha = 0.7)
            accelFig.plot(x_accel, accelNULL, 'w',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation6, facecolor = 'k', alpha = 0.7)
            accelFig.plot(x_accel, accelPH, 'k',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation7, facecolor = 'm', alpha = 0.7)
            accelFig.plot(x_accel, accelNH, 'm',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation8, facecolor = 'c', alpha = 0.7)
            accelFig.plot(x_accel, accelPM, 'c',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation9, facecolor = 'w', alpha = 0.7)
            accelFig.plot(x_accel, accelPH, 'w',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation10, facecolor = 'b', alpha = 0.7)
            accelFig.plot(x_accel, accelNH, 'b',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation11, facecolor = 'r', alpha = 0.7)
            accelFig.plot(x_accel, accelNULL, 'r',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation12, facecolor = 'y', alpha = 0.7)
            accelFig.plot(x_accel, accelPH, 'y',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation13, facecolor = 'g', alpha = 0.7)
            accelFig.plot(x_accel, accelNH, 'g',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation14, facecolor = 'r', alpha = 0.7)
            accelFig.plot(x_accel, accelNM, 'r',  linewidth = 0.5, linestyle = '--')
            accelFig.fill_between(x_accel, accel0, accelActivation15, facecolor = 'k', alpha = 0.7)
            accelFig.plot(x_accel, accelPH, 'k',  linewidth = 0.5, linestyle = '--')
            accelFig.set_title('Accel. membership activation')

            # Accel.
            accelMembershipFig.fill_between(x_accel, accel0, accelActivation1, facecolor = 'b', alpha = 0.7)
            accelMembershipFig.plot(x_accel, accelNH, 'b', linewidth = 0.5, linestyle = '--')
            accelMembershipFig.fill_between(x_accel, accel0, accelActivation2, facecolor = 'r', alpha = 0.7)
            accelMembershipFig.plot(x_accel, accelNM, 'r', linewidth = 0.5, linestyle = '--')
            accelMembershipFig.fill_between(x_accel, accel0, accelActivation3, facecolor = 'k', alpha = 0.7)
            accelMembershipFig.plot(x_accel, accelNULL, 'k', linewidth = 0.5, linestyle = '--')
            accelMembershipFig.fill_between(x_accel, accel0, accelActivation2, facecolor = 'r', alpha = 0.7)
            accelMembershipFig.plot(x_accel, accelPM, 'r', linewidth = 0.5, linestyle = '--')
            accelMembershipFig.fill_between(x_accel, accel0, accelActivation3, facecolor = 'k', alpha = 0.7)
            accelMembershipFig.plot(x_accel, accelPH, 'k', linewidth = 0.5, linestyle = '--')
            accelMembershipFig.set_title('Acceleration membership')

            accelAggregationFig.plot(x_accel, accelNH, 'b', linewidth = 0.5, linestyle = '--')
            accelAggregationFig.plot(x_accel, accelNM, 'g', linewidth = 0.5, linestyle = '--')
            accelAggregationFig.plot(x_accel, accelNULL, 'r', linewidth = 0.5, linestyle = '--')
            accelAggregationFig.plot(x_accel, accelPM, 'y', linewidth = 0.5, linestyle = '--')
            accelAggregationFig.plot(x_accel, accelPH, 'k', linewidth = 0.5, linestyle = '--')

            accelAggregationFig.fill_between(x_accel, accel0, aggregation, facecolor = 'Orange', alpha = 0.7)
            accelAggregationFig.plot([accel, accel], [0, nextAccelActivation], 'k', linewidth = 1.5, alpha = 0.9)
            accelAggregationFig.set_title('Aggregated acceleration membership')

            for fig in (straightDistanceMF, speedMF, AccelMF, accelFig, accelMembershipFig, accelAggregationFig):
                fig.spines['top'].set_visible(False)
                fig.spines['right'].set_visible(False)
                fig.get_xaxis().tick_bottom()
                fig.get_yaxis().tick_left()

            plt.tight_layout()

        brake = self._filterABS(state, brake)

        brake = np.clip(brake, 0.0, 1.0)
        accel = np.clip(accel, 0.0, 1.0)
        print('accel: %s, brake: %s'%(accel, brake))
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

        if(self.printFuzzyLogic == True):
            self.printFuzzyLogic = False

#        if(self.printInterval == 100):
#            self.printFuzzyLogic = True
#            self.printInterval = 0
#        self.printInterval += 1

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
            nbTracks = 3
            nbSuccessfulEpisodes = 0
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
#                doIt = True

                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        # Select the next action based on the observation
                        
#                        if(doIt):
                        print("Observation: ", observation)
                        action = controller.drive(observation)
#                        if(doIt):
                        print("Action: ", action)
#                            doIt = False
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