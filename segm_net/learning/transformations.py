from PIL import Image
import numpy as np
#from . import segmentation_transforms as ST
import torchvision.transforms as transforms

from ..util import string_to_list, parse_boolean

class NormalizeWithOwnMeanAndStd(object):
    '''
    Normalize with the mean and std of the image
    '''

    def __call__(self, x):
        # normalize the image by its own mean and unit variance
        return transforms.functional.normalize(x, mean=x.view(x.shape[0], -1).mean(dim=1), std=x.view(x.shape[0], -1).std(dim=1))

def get_transformations(config, split):
    '''
    Initialize a composite of transformations that have to be applied on the
    input data
    '''
    #normalized_image = TF.normalize(normalized_image, mean=normalized_image.view(normalized_image.shape[0], -1).mean(dim=1), std=normalized_image.view(normalized_image.shape[0], -1).std(dim=1))
    # Transformation list from config file
    transformation_names = string_to_list(config['data-augmentation']['transformations'])
    # initialize an empty list
    transform_list = []
    # if augmentation is going to be applied
    if parse_boolean(config['training']['augmentation']) and (split == 'training'):
        if "ColorJitter" in transformation_names:
            transform_list += [transforms.ColorJitter(brightness=float(config['ColorJitter']['brightness']), contrast=float(config['ColorJitter']['contrast']), saturation=float(config['ColorJitter']['saturation']), hue=float(config['ColorJitter']['hue']))]
      
    # by default, after any preprocessing we need to turn the ndarray into a tensor
    transform_list += [transforms.ToTensor()]
    # and then we need to apply the normalization
    transform_list += [transforms.Normalize((0.5,), (0.5,))]
    # return the compose
    return transforms.Compose(transform_list)