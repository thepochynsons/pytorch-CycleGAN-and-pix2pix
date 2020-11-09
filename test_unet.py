import torch
from PIL import Image
from configparser import ConfigParser
from os import path, makedirs
import scipy.io as spio
import numpy as np
from segm_net.util import string_to_list, parse_boolean, segmentation_to_colors, remove_labels_from_segmentation, recover_all_classes
from segm_net.learning.transformations import get_transformations, get_normalization
from segm_net.learning.architectures.model_factory import ModelFactory
from imageio import imwrite
import torchvision.transforms as transforms

def test(config, evaluation_set):
    # -------------------------------------
    # -------- Loading test-set -----------
    # -------------------------------------
    
    # read the data split
    data_split = ConfigParser()
    data_split.read(config['data-split']['split-file'])
    # get the filenames of the samples
    test_data_filenames = string_to_list(data_split['split'][evaluation_set])

    # prepare the paths to the input images
    img_path = path.join(config['data']['input-folder'], 'images')        

    # make results folder
    dir_results = path.join(config['data']['results-folder'], config['experiment']['name'])
    makedirs(dir_results, exist_ok=True)
    # inside it, create a folder for the segmentations
    dir_segmentations = path.join(dir_results,'segmentations')
    makedirs(dir_segmentations, exist_ok=True)
    # and another one for the scores
    dir_scores = path.join(dir_results,'scores')
    makedirs(dir_scores, exist_ok=True)

    # retrieve the classes
    class_names = ['background', 'artery', 'gallbladder', 'kidney', 'liver', 'spleen']
    num_classes = len(class_names)
    # and get the new numbers for the comparison
    new_class_nums = np.arange(0, num_classes)
    # and all the class names
    dic_keys = ['background', 'bones', 'artery', 'pancreas', 'gallbladder', 'kidney', 'surrenalGland', 'liver', 'spleen']
    all_classes = dict(zip(dic_keys, np.arange(0, len(dic_keys))))

    # transformation list
    transform_list = []
    transform_list += [transforms.ToPILImage()]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5,), (0.5,))]
    #transform_list +=[get_normalization(config)] 
    transformation = transforms.Compose(transform_list)
    # -------------------------------------
    # -------- Loading the model ----------
    # -------------------------------------
    
    # Loading best_model
        
    model_path = path.join(config['data']['output-folder'], config['experiment']['name'], 'checkpoints')
    model = ModelFactory.get_model(config)
    loaded_checkpoint = torch.load(path.join(model_path, 'best_model.pt'))
    model.load_state_dict(loaded_checkpoint['model'])
    
    print('Model successfully loaded')

    # -------------------------------------
    # -------- pasar por la red ----------
    # -------------------------------------

    # put the model in evaluation mode (just in case)
    model.cuda()
    model.eval()
    i=0
    for filename in test_data_filenames:
        #CAMBIAR ACA!!!
        # open the image
        image = Image.open(path.join(img_path, filename + '.png'))
        # segment the image, el metodo segment_image se encarga de las transformaciones
        print('Processing image {} ({}/{})'.format(filename, i+1, len(test_data_filenames)))
        outputs = model.segment_image(image) 

        # save the scores
        spio.savemat(path.join(dir_scores,filename + '.mat'), {'scores': outputs['scores']})
        
        # map the segmentation to colors and save it
        segmentation = segmentation_to_colors(recover_all_classes(outputs['segmentation'], all_classes, class_names, new_class_nums))
        segmentation = np.transpose(segmentation, (1,2,0))
        
        imwrite(path.join(dir_segmentations,filename + '.png'), segmentation)

        i=i+1

    print('test finished')



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="full path and filename of the configuration file", type=str)
    parser.add_argument("--eval_set", help="a string indicating the set to evaluate", type=str, default='test')
    args = parser.parse_args()

    # read the configuration file
    config = ConfigParser()
    config.read(args.config)
    #print('Evaluating on' + args.set)
    test(config, args.eval_set)