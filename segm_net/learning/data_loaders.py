
import torch
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import path

from ..util import parse_boolean,string_to_list, colors_to_classes, remove_labels_from_segmentation
from .transformations import get_transformations


#Santi acomodar
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms



def initialize_data_loader(config, split, filenames):
    '''
    Initialize a data loader by:
    1. Get the normalization strategy
    2. Creating a dataset object
    3. Assigning it to a data loader
    '''

    # create an image dataset, according to the problem to solve
    if config['experiment']['type']=='image-segmentation-2d':
        dataset_object = SimecoSegmentationDataset(config, split=split, filenames=filenames)
    
    # initialize a data loader
    loader = torch.utils.data.DataLoader(dataset_object,
                                         batch_size = int(config['training']['batch-size']),
                                         num_workers = 4,
                                         shuffle = (split=='training'),
                                         drop_last = parse_boolean(config['architecture']['batch-norm'])) # drop last batch if using batch norm

    return dataset_object, loader
    
    


class ImageDataset(torch.utils.data.Dataset):
    '''
    Dataset object for image datasets
    '''
    
    def __init__(self, config, split, filenames):
        '''
        Constructor
        '''
        super(ImageDataset, self).__init__()

        # assign the configuration of the image dataset object
        self.split = split                                                          # type of split (training, validation, test)
        self.is_random = (split == 'training')                                      # training data is always random
        self.img_path = path.join(config['data']['input-folder'], 'images')         # image data folder
        self.filenames = filenames                                                  # list of filenames       
        self.input_size = int(config['architecture']['input-size'])                 # input size
        # get the transformations that will be applied on data
        self.transformations = get_transformations(config, split)
        # validate all inputs
        # - split
        assert self.split in ['training', 'validation', 'test'], "Unknown split {}. It should be training, validation or test.".format(self.split)
        # - input path
        assert path.exists(self.img_path), "Folder {} does not exist".format(self.img_path)


    def __len__(self):
        '''
        Return the number of images
        '''
        # return the length of the array of filenames
        return len(self.filenames)


    def load_image(self, filename):
        '''
        Read an image and return as PIL format
        '''
        # load the image 
        image = Image.open(path.join(self.img_path, filename + '.png'))
        image_size = image.size
        # resize the image
        if not ((self.input_size == image_size[0]) and (self.input_size == image_size[1])):
            if len(image_size) == 2:
                image = image.resize((self.input_size, self.input_size))
            else:
                image = image.resize(image, (self.input_size, self.input_size, 3))

        return image


    def __getitem__(self, index):
        '''
        Get an image
        '''
        # load the image
        image = self.load_image(self.filenames[index])
        # apply the transformations on the image
        image = self.transform(image)
        # turn it to float
        image = image.float()
        return image


    def transform(self, data):
        '''
        Apply data augmentation
        '''
        # apply a series of image transformations to the input image
        transformed_data = self.transformations(data)
        return transformed_data


    def get_sample_for_plot(self, sample_size=2):
        '''
        Return a random sample of images
        '''

        # initialize an empty list of images and their filenames
        image_list = []
        filenames_list = []
        # shuffle the names to get a random sample
        idx = list(range(0, len(self)))
        random.shuffle(idx)
        # retrieve the first sample_size images
        for i in range(sample_size):
            image_list.append(self.load_image(self.filenames[idx[i]]))
            filenames_list.append(self.filenames[idx[i]])
        # return a dictionary
        return image_list, filenames_list

class ImageSegmentation2dDataset (ImageDataset):
    '''
    Dataset object for loading an imagen and its associated multi-label image
    '''
    def __init__(self, config, split, filenames):
        '''
        Constructor
        '''
        super(ImageSegmentation2dDataset, self).__init__(config, split, filenames)
        self.apply_augmentation = parse_boolean(config['training']['augmentation'])
        # labels_path
        self.labels_path = path.join(config['data']['labels-folder'], 'labels')
        self.labelnames = filenames # acomodarlo para el caso que son iguales o distintos?

        # Transformaciones
        self.transformations = get_transformations(config, split)
        transformation_names = string_to_list(config['data-augmentation']['transformations'])
        self.HorizontalFlip = "RandomHorizontalFlip" in transformation_names
        self.p_flip = float(config['RandomHorizontalFlip']['p'])
                
    def __getitem__(self, index):
        '''
        Get an image and it's asociated label
        '''

        # load image using parent class implementation
        image = self.load_image(self.filenames[index])
        # load the label
        label = self.load_label(self.labelnames[index])
        # apply the transformations on the image and de label 
                
        image, label = self.transform(image, label)
        
        label = np.squeeze(label)
        
        # turn it to long and float
        image = image.float()
        label = label.long()

        return image, label
    
    def transform(self, image, mask):
        # No puedo hacer los flips en imagen y mascara al mismo tiempo si lo hago en la lista de transformaciones.
        # Primero convierto a escala de grises y hago el Hflip
        mask = TF.to_pil_image(mask)
        image = TF.to_grayscale(image, num_output_channels=1)
        if self.apply_augmentation:
            if self.HorizontalFlip:
                # Random horizontal flipping
                if random.random() > self.p_flip:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
        # Hago el resto de las transformaciones con la lista de transformaciones
        image = self.transformations(image)
        # Convierto la mascara a tensor tambien
        mask = TF.to_tensor(mask)
        
        return image, mask
        

    def get_sample_for_plot(self, sample_size=2):
        '''
        Return a random sample of images
        '''

        # retrieve the images and their names
        loaded_images, filenames_list = super().get_sample_for_plot(sample_size)
        # initialize loaded data list
        loaded_data = []
        # and add the data
        for i in range(sample_size):
            label = self.load_label(filenames_list[i])
            #ACORDARSE DE ESTA BASURA! SANTI
            #loaded_image = loaded_images[i][:,:,0]
            loaded_image = loaded_images[i]
            loaded_data.append([loaded_image, label])

        return loaded_data, filenames_list
    
    def load_label(self, labelname):
        '''
        Read a label
        '''
        # load the label 
        label = Image.open(path.join(self.labels_path, labelname + '.png'))
        label_size = label.size
        # resize the label
        if not ((self.input_size == label_size[0]) and (self.input_size == label_size[1])):
            if len(label_size) == 2:
                label = label.resize((self.input_size, self.input_size))
            else:
                label = label.resize(label, (self.input_size, self.input_size, 3))
        label = np.array(label)      
        return label

class SimecoSegmentationDataset(ImageSegmentation2dDataset):

    def __init__(self, config, split, filenames):
        '''
        Constructor
        '''
        super(SimecoSegmentationDataset, self).__init__(config, split, filenames)

        self.class_names= string_to_list(config['data']['classes'])
        self.all_classes = dict(zip(string_to_list(config['data']['dic-keys']), np.fromstring( config['data']['dic-values'], dtype=int, sep=',' )))
        self.new_class_nums = np.arange(0,len(self.class_names))
                
    def __getitem__(self, index):
        '''
        Get an image and it's asociated label
        '''

        # load image using parent class implementation
        image = self.load_image(self.filenames[index])
        # load the label
        label = self.load_label(self.labelnames[index])
        label = self.reformat_label(label)
        # apply the transformations on the image and de label 
        image, label = self.transform(image, label)
        label = np.squeeze(label)
        # turn it to long and float
        image = image.float()
        label = label.long()
        
        return image, label
    
    
    def reformat_label(self, label):
        new_label = remove_labels_from_segmentation(colors_to_classes(label),
                                                    self.all_classes,
                                                    self.class_names,
                                                    self.new_class_nums)
        return new_label

