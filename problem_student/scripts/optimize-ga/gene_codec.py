
# Contains the necessary functions to encode genes and recover the elements

import numpy as np


# The information that needs to be encoded in the genes is the following:
#   - gear-N-ration : N >= 2, N <=6, float, [0.1, 5]
#   - rear-differential-ratio : float,  [1, 10]
#   - rear-spoiler-angle : float, [0, 90]
#   - front-spoiler-angle : float, [0, 90]
default_gene_format = {
    'gear-2-ratio' : {
        'range' : [0.1, 5],
        'bits' : 6
    },
    'gear-3-ratio' : {
        'range' : [0.1, 5],
        'bits' : 6
    },
    'gear-4-ratio' : {
        'range' : [0.1, 5],
        'bits' : 6
    },
    'gear-5-ratio' : {
        'range' : [0.1, 5],
        'bits' : 6
    },
    'gear-6-ratio' : {
        'range' : [0.1, 5],
        'bits' : 6
    },
    'rear-differential-ratio' : {
        'range' : [1, 10],
        'bits' : 6
    },
    'rear-spoiler-angle' : {
        'range' : [0, 90],
        'bits' : 7
    },
    'front-spoiler-angle' : {
        'range' : [0, 90],
        'bits' : 7
    }
}


# Computes the length of the chromosone needed to store the defined structure
def length(chromosone_format):
    total = 0
    for parameter, specs in chromosone_format.items():
        total += specs['bits']
    return total


# Encode function (parameters -> chromosone)
def encode(parameters, chromosone_format = default_gene_format):
    chromosone = ''
    for parameter_name, param in chromosone_format.items():
        parameter_value = parameters[parameter_name][0]
        encoded_parameter = encode_parameter(
                parameters[parameter_name][0], param['range'], param['bits'])
        chromosone += encode_parameter
    return chromosone

def encode_parameter(param, limits, precision):
    constrained_param = np.clip(param, limits[0], limits[1])
    norm_param = (constrained_param - limits[0]) / (limits[1] - limits[0])
    quantized_param = int(round(norm_param * (2**percision - 1)))
    return format(quantized_param, f'0{precision}b')


# Decode function (chromosone -> parameters)
def decode(chromosone, chromosone_format = default_gene_format):
    parameters = {}
    for parameter_name, param in chromosone_format.items():
        encoded_parameter = chromosone[:param['bits']]
        parameter_value = decode_parameter(encoded_parameter, param['range'])
        parameters[parameter_name] = np.array([parameter_value])
    return parameters

def decode_parameter(encoded_parameter, limits):
    norm_param = int(encoded_parameter, 2) / (2**len(encoded_parameter) - 1)
    return norm_param * (limits[1] - limits[0]) + limits[0]
