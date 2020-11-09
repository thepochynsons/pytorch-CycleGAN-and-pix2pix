from torch import nn

def get_activation_unit(unit_name):
    '''
    Given the name of an activation unit, return it
    '''

    if unit_name == 'relu':
        # Basic Rectified Linear Unit
        return nn.ReLU()

    elif unit_name == 'prelu':
        # Parametric Rectified Linear Unit
        # from the paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # (https://arxiv.org/pdf/1502.01852)
        return nn.PReLU()

    elif unit_name == 'elu':
        # Exponential Linear Unit
        # from the paper: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        # (https://arxiv.org/pdf/1511.07289.pdf)
        return nn.ELU()

    elif unit_name == 'lrelu':
        # Leaky ReLU
        # Better described here: https://arxiv.org/pdf/1505.00853.pdf
        return nn.LeakyReLU()

    elif unit_name == 'rrelu':
        # Randomized Leaky Rectified Linear Unit
        # from the paper: Empirical Evaluation of Rectified Activations in Convolutional Network
        # (https://arxiv.org/pdf/1505.00853.pdf)
        return nn.RReLU()

    elif unit_name == 'selu':
        # Scale Exponential Linear Unit
        # from the paper: Self-Normalizing Neural Networks 
        # (https://arxiv.org/pdf/1505.00853.pdf)
        return nn.SELU()

    else:
        raise ValueError('Activation unit {} unknown'.format(unit_name))



def get_activation_function(activation_name):
    '''
    Return the activation function to use on the outputs of a neural network
    '''

    if activation_name == 'Softmax':
        return nn.Softmax(dim=1)
    elif activation_name == 'Softmax2d':
        return nn.Softmax2d()
    else:
        raise ValueError('Unkown activation function {}'.format(activation_name))

def get_pooling_layer(pooling_name, kernel_size):
    '''
    Given the name of a pooling layer, returns its corresponding layer
    '''

    if pooling_name == 'max':
        # Max Pooling layer
        return nn.MaxPool2d(kernel_size=kernel_size)

    elif pooling_name == 'avg':
        # Avg Pooling layer
        return nn.AvgPool2d(kernel_size=kernel_size)

    else:
        raise ValueError('Pooling layer {} unknown'.format(pooling_name))