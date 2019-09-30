#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:20:56 2019

@author: user
"""
#%%
rpm = ['L', 'M', 'H']
speed = ['L', 'ML', 'M', 'MH', 'H']
gear = ['R', 'N', 1, 2, 3, 4, 5, 6]

for s in speed:
    for r in rpm:
        printGear = False
        gear = 5
        print('If speed is %s and rpm is %s, then gear is: %s' % (s, r, gear))


#%%
incr = 1
rpm = ['L', 'M', 'H']
speed = ['L', 'ML', 'M', 'MH', 'H']

#        activationRule1 = np.fmin(speedLevelL, rpmLevelH)
#        gearActivation1 = np.fmin(activationRule1, gear1)
#
#        activationRule2 = np.fmin(speedLevelML, rpmLevelH)
#        gearActivation2 = np.fmin(activationRule2, gear2)
#
#        activationRule3 = np.fmin(speedLevelM, rpmLevelH)
#        gearActivation3 = np.fmin(activationRule3, gear3)
#
#        activationRule4 = np.fmin(speedLevelMH, rpmLevelH)
#        gearActivation4 = np.fmin(activationRule4, gear4)
#
#        activationRule5 = np.fmin(speedLevelH, rpmLevelH)
#        gearActivation5 = np.fmin(activationRule5, gear5)
for s in speed:
    for r in rpm:
        gear = 5

        print('activationRule%i = np.fmin(speedLevel%s, rpmLevel%s)' %(incr, s, r))
        print('gearActivition%i = np.fmin(activitationRule%i, gear%s)\n\r ' %(incr,incr, gear))
        incr += 1

#%%
incr = 1
Track19 = ['C', 'N', 'F'] # Close, Medium Close, Medium Near, Near, Far, Very Far
speed = ['L', 'ML', 'M', 'MH', 'H']
        
        # If speed levl is Low and rpm level is high, then gear is 1 translates to:
#        activationRule1 = np.fmin(speedLevelL, rpmLevelH)
#        gearActivation1 = np.fmin(activationRule1, gear1)

for s in speed:
    for t in Track19:
        accel = 0.5
        print('activationRule%i = np.fmin(speedLevel%s, track19Level%s)' %(incr, s, t))

        incr += 1

print('\r\n')

incr = 1
for s in speed:
    for t in Track19:
        accel = 0.5
        print('accelActivation%i = np.fmin(activationRule%i, accel%s)' %(incr,incr, accel))

        incr += 1

#%%
incr = 1

angle = ['L', 'C', 'R']
speed = ['L', 'M', 'H']
TrackPos = ['L', 'C', 'R'] # Close, Medium Close, Medium Near, Near, Far, Very Far

        # If speed levl is Low and rpm level is high, then gear is 1 translates to:
#        activationRule1 = np.fmin(speedLevelL, rpmLevelH)
#        gearActivation1 = np.fmin(activationRule1, gear1)

for s in speed:
    for t in TrackPos:
        for a in angle:
            accel = 0.5
            print('activationRule%i = np.fmin(speedLevel%s, positionLevel%s, angleLevel%s)' %(incr, s, t, a))
            incr += 1

print('\r\n')

incr = 1
for s in speed:
    for t in TrackPos:
        for a in angle:
            steer = 0.5
            print('accelActivation%i = np.fmin(activationRule%i, steering%s)' %(incr,incr, steer))
            incr += 1

#%%
def recursiveTryOut(vector):
    if(len(vector)==2):
        return vector[0] * vector[1]
    else:
        a = vector[0]
        del vector[0]
        return a * recursive(vector)

recTryOut([2, 2, 2])
