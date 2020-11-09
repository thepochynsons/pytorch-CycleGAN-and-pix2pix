
import torch
import torch.nn as nn
import numpy as np

from ..util import string_to_list, get_list_of_strings_from_string


def get_loss_function(config):
    '''
    Initialize a loss function 
    '''
    loss = []
    loss_names = None
    #if config['architecture']['model-type'] == 'multinet':
    if (',' in config['training']['loss']):
        loss_names = get_list_of_strings_from_string(config['training']['loss'])
    else:
        loss_names = [ config['training']['loss'] ]
    for current_loss_name in loss_names:
            # append current loss
            loss.append(get_loss_from_loss_name(current_loss_name, config))
    if len(loss_names)==1:
        loss = loss[0]
        loss_names = loss_names[0]
    return loss, loss_names



def get_loss_from_loss_name(loss_name, config):
    '''
    Return the loss object corresponding to the given loss name
    (TODO: eventually we might use here the configuration of the loss as a parameter)
    '''

    if loss_name == 'L1Loss':

        current_loss = nn.L1Loss()   # return the L1 loss

    elif loss_name == 'MSELoss':

        current_loss = nn.MSELoss()  # return the MSE loss 

    elif loss_name == 'CrossEntropyLoss':

        # weight = None
        weights = torch.Tensor(np.ones(len(string_to_list(config['data']['classes']))))
        if ('CrossEntropyLoss' in config):
            if 'weights' in config['CrossEntropyLoss']:
                weights = torch.Tensor(np.fromstring(config['CrossEntropyLoss']['weights'], sep=','))

        if torch.cuda.is_available():
            weights = weights.cuda()

        current_loss = nn.CrossEntropyLoss(weight=weights)  # return the cross entropy loss

    return current_loss



