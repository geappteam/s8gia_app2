# Copyright (c) 2019, Simon Brodeur
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
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2019

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D

###############################################
# Define helper functions here
###############################################


def singletonmf(x, a):
    """
    Singleton membership function generator.
    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : constant
    Returns
    -------
    y : 1d array
        Singleton membership function.
    """
    y = np.zeros(len(x))

    if a >= np.min(x) and a <= np.max(x):
        idx = (np.abs(x - a)).argmin()
        y[idx] = 1.0

    return y


def createFuzzyController():

    # TODO: Create the fuzzy variables for inputs and outputs.
    # Defuzzification (defuzzify_method) methods for fuzzy variables:
    #    'centroid': Centroid of area
    #    'bisector': bisector of area
    #    'mom'     : mean of maximum
    #    'som'     : min of maximum
    #    'lom'     : max of maximum
    perimeter = ctrl.Antecedent(np.linspace(0, 200, 10000), 'meanperimeter')
    area = ctrl.Antecedent(np.linspace(0, 3000, 10000), 'meanarea')
    diagnosis = ctrl.Consequent(np.linspace(0, 1, 10000), 'diagnosis', defuzzify_method='centroid')

    # Accumulation (accumulation_method) methods for fuzzy variables:
    #    np.fmax
    #    np.multiply
    diagnosis.accumulation_method = np.fmax

    # TODO: Create membership functions
    perimeter['small'] = fuzz.trapmf(perimeter.universe, [0, 0, 80, 100])
    perimeter['large'] = fuzz.trapmf(perimeter.universe,[80,100,200,200])

    area['small'] = fuzz.trapmf(area.universe, [0, 0, 500, 750])
    area['large'] = fuzz.trapmf(area.universe,[500,750,3000,3000])

    diagnosis['zero'] = singletonmf(diagnosis.universe,0)
    diagnosis['one'] = singletonmf(diagnosis.universe,1)

    # TODO: Define the rules.
    rules = []
    rules.append(ctrl.Rule(antecedent=(perimeter['small'] & area['small']), consequent=diagnosis['zero']))
    rules.append(ctrl.Rule(antecedent = (perimeter['large'] & area['large']), consequent = diagnosis['one']))

    # Conjunction (and_func) and disjunction (or_func) methods for rules:
    #     np.fmin
    #     np.fmax
    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def predict(sim, data):
    outputs = []
    for perimeter, meanArea, meanSmoothness in data:
        # TODO: set the input to the fuzzy system
        sim.input['meanperimeter'] = perimeter
        sim.input['meanarea'] = meanArea

        sim.compute()

        # TODO: get the output from the fuzzy system
        pred = np.round(sim.output['diagnosis'])
        outputs.append(pred)

    outputs = np.array(outputs, dtype=np.int)
    return outputs

###############################################
# Define code logic here
###############################################


def main():

    # Load breast cancer data set from file
    # Attributes:
    # mean_radius: mean of distances from center to points on the perimeter
    # mean_area: mean area of the core tumor
    # mean_texture: standard deviation of gray-scale values
    # mean_perimeter: mean size of the core tumor
    # mean_smoothness: mean of local variation in radius lengths

    # TODO: Analyze the input data
    # Input attributes: mean_texture, mean_area, mean_smoothness
    S = np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1)
    data = np.array(S[:, [2, 3, 4]], dtype=np.float32)

    # Output:
    # The diagnosis of breast tissues (benign = 0, malignant = 1) where malignant denotes that the disease is harmful
    target = np.array(1 - S[:, -1], dtype=np.int)

    # Show the data
    colors = np.array([[0.0, 1.0, 0.0],    # Green
                       [1.0, 0.0, 0.0]])   # Red

    c = colors[target]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10.0, c=c, marker='x')
    ax.set_title('Breast cancer dataset')
    ax.set_xlabel('mean texture')
    ax.set_ylabel('mean area')
    ax.set_zlabel('mean smoothness')
    plt.show()

    # TODO : Apply any relevant transformation to the data
    # (e.g. filtering, normalization, dimensionality reduction)

    # Create fuzzy controller
    sim = createFuzzyController()

    # Display rules
    print('------------------------ RULES ------------------------')
    for rule in sim.ctrl.rules:
        print(rule)
    print('-------------------------------------------------------')

    # Display fuzzy variables
    for var in sim.ctrl.fuzzy_variables:
        var.view()
    plt.show()

    # Print the number of classification errors from the training data
    targetPred = predict(sim, data)
    nbErrors = np.count_nonzero(targetPred != target)
    accuracy = (len(data) - nbErrors) / len(data)
    print('Classification accuracy: %0.3f' % (accuracy))


if __name__ == "__main__":
    main()
