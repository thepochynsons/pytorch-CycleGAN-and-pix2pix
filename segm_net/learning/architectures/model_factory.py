from .. import training as training_routines
from . import segmentation_networks as segmentation_networks

class ModelFactory():
    '''
    Initialize a model based on a given configuration file
    '''

    @staticmethod
    def get_model(config):
        '''
        Creates a model from a configuration file
        '''
         # if it's an image segmentation
        if config['experiment']['type'] == 'image-segmentation-2d':
            # pick one of the implemented networks for image segmentation
            if config['architecture']['model'] =='unet':
                return segmentation_networks.Unet(config)
        
        # unknown type of experiment
        else:
            raise ValueError('Unknown type of experiment {}. We dont have any architectures for this problem'.format(config['experiment']['type']))



    @staticmethod
    def get_training_wrapper(model, config):
        '''
        Returns a training wrapper that might handle all the training process for a given model
        '''
        if (',' in config['training']['loss']):
            return training_routines.AdversarialLossTrainingWrapper(config,model)
        else:
            return training_routines.TrainingWrapper(config, model)
        