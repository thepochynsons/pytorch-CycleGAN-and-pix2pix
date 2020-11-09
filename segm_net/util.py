
def parse_boolean(input_string):
    '''
    Parse a boolean
    '''
    return input_string.upper()=='TRUE'


def list_to_string(input_list):
    '''
    Turn an input list of strings into a single string with comma separated values
    '''
    return ','.join(list(input_list))


def string_to_list(input_string):
    '''
    Turn a string with comma separated values into a list
    '''
    return input_string.split(',')

def get_list_of_strings_from_string(given_string):
    '''
    Split a string with colons
    '''
    return given_string.replace(' ','').split(',')
    
from os import path

def remove_extensions(filenames):
    '''
    Remove file extensions from filenames
    '''

    # initialize an empty list of files
    new_filenames = []
    # iterate for each filename 
    for i in range(len(filenames)):
        # get the filename and the extension
        filename_without_extension, file_extension = path.splitext(filenames[i])
        # append only the filename to the list
        new_filenames.append(filename_without_extension)
    
    return new_filenames


import re

def natural_key(string_):
    '''
    To sort strings using natural ordering. See http://www.codinghorror.com/blog/archives/001018.html
    '''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


import torch

def get_available_device():
    '''
    Returns the available device to run the algorithm
    '''
    # detect if we have a GPU available
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



import numpy as np


def colors_to_classes(label):
    '''
    Given a labelling with colors, return a labelling with class numbers
    '''
    # Convierto los colores en las etiquetas de all_classes
    mask = np.zeros((label.shape[0],label.shape[1]),dtype=np.dtype('i'))
    # (pelvis - grey)
    mask[(label[:,:,0] == 151) * (label[:,:,1]==151) * (label[:,:,2]==147)]=9
    # (spleen - pink)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==0) * (label[:,:,2]==255)]=8
    # (liver - purple)
    mask[(label[:,:,0] == 100) * (label[:,:,1]==0) * (label[:,:,2]==100)]=7
    # (surrenalGland - cyan)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==255) * (label[:,:,2]==255)]=6
    # (kidney - yellow)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==255) * (label[:,:,2]==0)]=5
    # (gallbladder - green)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==255) * (label[:,:,2]==0)]=4
    # (pancreas - blue)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==0) * (label[:,:,2]==255)]=3
    # (artery - red)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==0) * (label[:,:,2]==0)]=2
    # (bones - white)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==255) * (label[:,:,2]==255)]=1

    return mask


def remove_labels_from_segmentation(labels, all_classes, class_names, new_class_nums):

    new_label = np.zeros((labels.shape[0],labels.shape[1]), dtype=np.dtype('i'))
    for i in range(len(class_names)):
        new_label[labels == all_classes[class_names[i]]] = new_class_nums[i]
    return new_label


def segmentation_to_colors(real_predictions):
    '''
    real_predictions =  np.zeros((predictions.shape[0],predictions.shape[1]),dtype=np.dtype('i'))
    for i in range(1,len(self.class_names)):
        real_predictions[predictions == i] = self.all_classes.get(self.class_names[i])
    '''

    red_channel = np.zeros(real_predictions.shape)
    green_channel = np.zeros(real_predictions.shape)
    blue_channel = np.zeros(real_predictions.shape)
    # (bones - white)
    red_channel[real_predictions==1] = 255
    green_channel[real_predictions==1] = 255
    blue_channel[real_predictions==1] = 255
    # (artery - red)
    red_channel[real_predictions==2] = 255
    green_channel[real_predictions==2] = 0
    blue_channel[real_predictions==2] = 0
    # (pancreas - blue)
    red_channel[real_predictions==3] = 0
    green_channel[real_predictions==3] = 0
    blue_channel[real_predictions==3] = 255
    # (gallbladder - green)
    red_channel[real_predictions==4] = 0
    green_channel[real_predictions==4] = 255
    blue_channel[real_predictions==4] = 0
    # (kidney - yellow)
    red_channel[real_predictions==5] = 255
    green_channel[real_predictions==5] = 255
    blue_channel[real_predictions==5] = 0
    # (surrenalGland - cyan)
    red_channel[real_predictions==6] = 0
    green_channel[real_predictions==6] = 255
    blue_channel[real_predictions==6] = 255
    # (liver - purple)
    red_channel[real_predictions==7] = 100
    green_channel[real_predictions==7] = 0
    blue_channel[real_predictions==7] = 100
    # (spleen - pink)
    red_channel[real_predictions==8] = 255
    green_channel[real_predictions==8] = 0
    blue_channel[real_predictions==8] = 255
    # (pelvis - grey)
    red_channel[real_predictions==9] = 151
    green_channel[real_predictions==9] = 151
    blue_channel[real_predictions==9] = 147

    predictions_rgb = np.stack((red_channel, green_channel, blue_channel), axis=0)

    return predictions_rgb

def recover_all_classes(labels, all_classes, class_names, new_class_nums):
    new_label = np.zeros((labels.shape[0],labels.shape[1]), dtype=np.dtype('i'))
    for i in range(len(class_names)):
        new_label[labels == new_class_nums[i]] = all_classes[class_names[i]]
    return new_label