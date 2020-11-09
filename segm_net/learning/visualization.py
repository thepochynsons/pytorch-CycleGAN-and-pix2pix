import shutil
import numpy as np
import matplotlib.pyplot as plt

from os import path, makedirs
from abc import ABC, abstractmethod
from ..util import string_to_list
from ..util import segmentation_to_colors as segm_to_colors

from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from scipy.stats import entropy

def get_plotter(config):
    '''
    Setup the right plotter
    '''

    # setup the visualizer according to the backend library
    if config['visualization']['library'] == 'visdom':
        if config['experiment']['type'] == 'image-segmentation-2d':
            plotter = ImageSegmentation2dVisdomPlotter(config)

    return plotter




class GenericPlotter(ABC):
    '''
    Generic abstract class to plot training statistics
    '''

    def __init__(self, config):
        '''
        Generic initializer
        '''
        super(GenericPlotter, self).__init__()


    @abstractmethod
    def plot_multiple_statistics(self, plot_name, x, y_values):
        '''
        Plot line plots in the same plot
        '''
        pass

    @abstractmethod
    def plot_scalar(self, plot_name, x, y, legend):
        '''
        Plot a line plot
        '''
        pass

    @abstractmethod
    def display_image(self, image_key, image, caption=''):
        '''
        Display given images in the plot
        '''
        pass


class GenericVisdomPlotter(GenericPlotter):
    '''
    Visdom based generic plotter implementation
    '''

    def __init__(self, config):
        '''
        Initializer
        '''
        super(GenericVisdomPlotter, self).__init__(config)

        # prepare the environment name
        self.env = 'Santi_' + config['experiment']['name']
        # default host and port
        hostname = 'http://localhost/'
        port = 8098
        # replace host and port by the one provided in the config file
        if 'hostname' in config['visualization']:
            hostname = config['visualization']['hostname']
        if 'port' in config['visualization']:
            port = int(config['visualization']['port'])

        # initialize the object for visualization
        self.viz = Visdom(server=hostname, port=port)
        # the dictionary of plots and figures
        self.figures = dict()
        self.plots = dict()
        
        # initialize the current epoch in 0
        self.current_epoch = 0
        

    def plot(self, plot_name, split_name, x, y, x_label='Epochs'):
        '''
        Plot a line plot
        '''

        # if the plot is not in the dictionary, initialize one
        if (plot_name not in self.plots):
            self.plots[plot_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, 
                                                  opts=dict(legend=[str(split_name)],
                                                            title=plot_name,
                                                            xlabel=x_label,
                                                            ylabel=plot_name))
        # if the plot is already there, update
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, 
                          update='append', win=self.plots[plot_name], name=str(split_name))


    def plot_multiple_statistics(self, plot_name, x, y_values):
        '''
        Plot multiple statistics within the same plot
        '''
        # get the split names
        split_names = y_values.keys()
        # iterate for each of them
        for split in split_names:
            # plot the values
            self.plot(plot_name, split, x, y_values[split])


    def plot_scalar(self, plot_name, x, y, legend):
        '''
        Plot a line plot
        '''
        self.plot(plot_name, legend, x, y)


    def display_image(self, image_key, image, caption=''):
        '''
        Display given image in the plot
        '''
        # if the image is already in the plot, remove it to replace it for the new one
        if image_key in self.figures:
            self.viz.close(win=self.figures[image_key], env=self.env)
            del self.figures[image_key]
        # plot the image
        self.figures[image_key] = self.viz.images(image, env=self.env, opts=dict(title=caption))



class GenericTensorboardPlotter(GenericPlotter):
    '''
    Tensorboard generic plotter implementation
    '''

    def __init__(self, config):
        '''
        Initializer
        '''
        super(GenericTensorboardPlotter, self).__init__(config)

        # set the information folder
        self.info_folder = path.join(config['data']['output-folder'], config['experiment']['name'], 'intermediate-results', 'tensorboard')
        # remove old files within
        if path.exists(self.info_folder):
            shutil.rmtree(self.info_folder, ignore_errors=True)
        # create the new one
        makedirs(self.info_folder, exist_ok=True)
        # initialize the object for visualization
        self.viz = SummaryWriter(self.info_folder)
        

    def plot_multiple_statistics(self, plot_name, x, y_values):
        '''
        Plot multiple statistics within the same plot
        '''
        self.viz.add_scalars(plot_name, y_values, x)


    def plot_scalar(self, plot_name, x, y, legend):
        '''
        Plot a line plot
        '''
        self.viz.add_scalar(plot_name, y, x)


    def display_image(self, image_key, image, caption=''):
        '''
        Display given image in the plot
        '''
        # plot the image
        self.viz.add_image(image_key, image)



import plotly.tools as tls

class ImageSegmentation2dVisdomPlotter(GenericVisdomPlotter):
    '''
    Image Segmentation2d plotter
    '''

    def __init__(self, config):
        '''
        Initializer
        '''
        super(ImageSegmentation2dVisdomPlotter, self).__init__(config)

        # retrieve the classes
        self.classes = string_to_list(config['architecture']['num-classes'])
        self.class_names= string_to_list(config['data']['classes'])
        self.all_classes = dict(zip(string_to_list(config['data']['dic-keys']), np.fromstring( config['data']['dic-values'], dtype=int, sep=',' )))


    def display_results(self, images, predictions, probabilities, true_labels, epoch):
        '''
        Plot segmentation results
        '''
        
        fig = plt.figure()
        for i in range(0, len(images)):
            #Siempre me manejo con pil image. por eso tengo que pasarlo a np y transponer.
            np_img = np.array(images[i])
            np_img = np.transpose(np_img,(2, 0, 1))
            
            ent = entropy(probabilities[i], base=probabilities[i].shape[0], axis=0)
            ent = 255 * (ent - np.min(ent.flatten()))/(np.max(ent.flatten())- np.min(ent.flatten()))          
            prediction_rgb = self.segmentation_to_colors(predictions[i])
            true_labels_rgb = np.transpose(true_labels[i], (2,0,1))

            to_plot = np.stack((np_img, prediction_rgb, true_labels_rgb), axis=0)
            #stackear las probabilidades
            for k in range(probabilities[i].shape[0]):
                these_probabilities = np.stack((probabilities[i][k,:,:], probabilities[i][k,:,:], probabilities[i][k,:,:]),axis=0) * 255
                to_plot = np.append(to_plot, np.expand_dims(these_probabilities,axis=0), axis=0)                
            to_plot = np.append(to_plot,np.expand_dims(np.stack((ent,ent,ent),axis=0), axis=0), axis=0)
            self.display_image(str(i), to_plot)
                      
            ax = fig.add_subplot(1, len(images), i+1, xticks=[], yticks=[])
            # display the image
            #plt.imshow(to_plot)            
        return fig


    def segmentation_to_colors(self, predictions):
        
        real_predictions =  np.zeros((predictions.shape[0],predictions.shape[1]),dtype=np.dtype('i'))
        for i in range(1,len(self.class_names)):
            real_predictions[predictions == i] = self.all_classes.get(self.class_names[i])
        
        return segm_to_colors(real_predictions)
