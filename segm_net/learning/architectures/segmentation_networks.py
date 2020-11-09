import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os import path

from torch.autograd import Variable
from skimage import filters
from ...util import parse_boolean

from .activations import get_activation_function, get_pooling_layer, get_activation_unit

import functools

import torchvision.transforms.functional as TF

class SegmentationNetwork(nn.Module):
    '''
    Abstract class defining some key methods for segmenting B-scans.
    '''

    def __init__(self, config):
        '''
        Constructor of the base class.
        '''
        super(SegmentationNetwork, self).__init__()

        # set the name of the model
        self.name = 'model'

        # setup default configuration
        self.num_classes = 2          # number of output classes
        self.is_deconv = False      # deconvolutions or upsampling
        self.is_batchnorm = True    # batch normalization
        self.dropout = 0.0          # dropout probability
        self.use_otsu = False       # va boolean indicating if we need to use Otsu or not

        # change configuration if available in the file
        if 'num-classes' in config['architecture']:
            self.num_classes = int(config['architecture']['num-classes'])
        if 'use-deconvolution' in config['architecture']:
            self.is_deconv = parse_boolean(config['architecture']['use-deconvolution'])
        if 'batch-norm' in config['architecture']:
            self.is_batchnorm = parse_boolean(config['architecture']['batch-norm'])
        if 'dropout' in config['training']:
            self.dropout = float(config['training']['dropout'])
        if 'use-otsu' in config['experiment']:
            self.use_otsu = parse_boolean(config['experiment']['use-otsu'])
   
    def format_loaded_data(self, loaded_data):
        '''
        Format input data to be processed by the network during training
        '''

        # reformat the data: in segmentation networks our loaded data consists of:
        # - an image
        inputs = loaded_data[0]
        # - and the labels
        targets = loaded_data[1]

        return inputs, targets

    def plot_outputs(self, plotter, input_data):
        '''
        Plot the outputs of the model for given input data
        '''
        # initialize empty lists
        images = []
        segmentations = []
        scores = []
        true_labels = []
        # iterate for each sample in the input data list
        for i in range(len(input_data)):
            # format the input data in the appropiate shape
            image, label = self.format_loaded_data(input_data[i])
            # segment image
            outputs = self.segment_image(image)
            segmentations.append(outputs['segmentation'])
            scores.append(outputs['scores'])
            # append everything
            images.append(image)
            true_labels.append(label)
            # display the results
        return plotter.display_results(images, segmentations, scores, true_labels, None)

    def save_intermediate_results(self, outputs, intermediate_results_folder, epoch):
        '''
        Save intermediate results
        '''
        # by default, we assume outputs is a matplotlib figure
        plt.savefig(path.join(intermediate_results_folder, 'intermediate_results_epoch={}.png'.format(str(epoch))), format='png')

    def get_scores(self, image_tensor):
        '''
        Do a forward pass and return the scores as a numpy array
        '''
        # in this based class, it is just a forward pass y le pasa la la capa de activacion final aca, no se hace en el foward xq la loss ya la incluye y sino se hace 2 veces)
        model_outputs = self.final_activation(self.forward(image_tensor))
        # encode the output and return it
        return self.encode_output(model_outputs)


    def encode_output(self, outputs):
        '''
        moving to numpy
        '''
           # put it in numpy shape
        if self.num_classes == 2:
            outputs = outputs.data[0][1].cpu().numpy()
        elif self.num_classes == 1:
            outputs = outputs.data[0].cpu().numpy().squeeze()
        else:
            outputs = np.squeeze(outputs.data.cpu().numpy())
            
        return outputs


    def get_segmentation_from_scores(self, scores):
        '''
        Given the scores, threshold it to get the segmentation
        '''

        if (self.num_classes <= 2):
            
            if self.use_otsu:
                # use Otsu to threshold binary scores
                if np.unique(scores.flatten()).size > 1:
                    val = filters.threshold_otsu(scores)
                else:
                    val = scores[0,0]
            else:
                # use 0.5
                val = 0.5
            segmentation = np.asarray(scores >= val, dtype=np.float32)

        else:
            # use max probability of each class to predict the segmentation      
            segmentation = np.asarray( np.argmax(scores, axis=0), dtype=np.float32 )

        return segmentation


    def get_outputs_names(self, bayesian=False, from_forward=False):
        '''
        Return the names of the outputs (same as keys in segment_image)
        '''

        if from_forward:
            # default outputs
            outputs_names = ['segmentation']

        else:

            # default outputs
            outputs_names = ['scores', 'segmentation']
            # if bayesian, add the other outputs
            if bayesian:
                outputs_names = outputs_names + ['stdev-uncertainty', 'entropy-uncertainty']
        
        return outputs_names


    def get_outputs(self, normalized_image):
        '''
        Process a normalized image to get all the outputs in the right shape
        '''
        # get the scores doing a forward pass
        scores = self.get_scores(normalized_image)
        # segment the scores
        segmentation = self.get_segmentation_from_scores(scores)
        
        return {'scores': scores, 'segmentation': segmentation}


    def segment_image(self, image, is_bayesian=False):
        '''
        Standard way to segment an image
        '''
        
        if is_bayesian:
            # use always train mode for bayesian
            self.apply(bayesian_eval_mode)
        else:
            # make sure that you're always in eval mode
            self.eval()
        
        # Aplico las transformaciones 
        normalized_image = TF.to_grayscale(image, num_output_channels=1)
        normalized_image = TF.to_tensor(normalized_image)
        normalized_image = TF.normalize(normalized_image,(0.5,), (0.5,))
        normalized_image = torch.unsqueeze(normalized_image, 0)
        with torch.no_grad():
            # turn it to cuda
            if (torch.cuda.is_available()):
                normalized_image = normalized_image.cuda()
            # get outputs
            outputs = self.get_outputs(normalized_image)

        return outputs

    def evaluate(self, targets, predicted, eval_metric):
        '''
        Evaluate the Dice index of the predicted segmentation
        '''
    
        if self.num_classes == 2:
            if predicted.size()[0] == 1:
                # get the numpy scores of the true positive class
                predicted = predicted.data[0][1].cpu().numpy()
            else:
                # retrieve the true positive class
                predicted = predicted.data.cpu().numpy()
        else:
            # retrieve the true positive class
            predicted = predicted.data.cpu().numpy()
            
        # extract ground truth from the targets
        gt = targets.data.cpu().numpy()

        # accumulate the dice values for each image in the
        # batch on the dice values list
        dice_values = []
        # expand the prediction if necessary
        if len(predicted.shape) < 4:
            for i in range(0, 4-len(predicted.shape)):
                predicted = np.expand_dims(predicted, axis=0)
        
        for j in range(0, gt.shape[0]):

            # don't compute dice if the ground truth don't have a target class
            if np.unique(gt.flatten()).size > 1:

                if self.num_classes == 2:

                    # get the segmentation
                    if predicted.shape[1]==1:
                        current_prediction = self.model.get_segmentation_from_scores(predicted[j,0,:,:])
                    else:
                        current_prediction = self.model.get_segmentation_from_scores(predicted[j,1,:,:])
                    # compute the dice value on the list
                    current_average_dice = eval_metric(gt[j,:,:], current_prediction)

                elif self.num_classes == 1:

                    # get the segmentation
                    current_prediction = self.model.get_segmentation_from_scores(predicted[j,0,:,:])
                    # compute the dice value on the list
                    current_average_dice = eval_metric(gt[j,:,:], current_prediction)

                else:

                    #TODO: Chequear el temita de mandarle NaN al dice
                    current_prediction = np.argmax(predicted, axis=1)
                    metric_per_class = np.zeros((self.num_classes-1,1), dtype=np.float)
                    #current_average_dice = 0.0

                    for k in range(1, self.num_classes):
                        # get the binary segmentation of the current class
                        k_class_gt = gt[j,:,:]==k 
                        k_class_prediction = current_prediction[j,:,:]==k 
                        m = eval_metric(k_class_gt, k_class_prediction)
                        metric_per_class[k-1] = m
                        #current_average_dice += m
                    
                    current_average_dice = np.nanmean(metric_per_class)
                    #current_average_dice = current_average_dice / (self.num_classes -1)

                dice_values.append( current_average_dice )

        return dice_values

###############################################################################################################################

class UnetConvBlock(nn.Module):
    '''
    Convolutional block of a U-Net:
    Conv2d - Batch normalization (optional) - ReLU
    Conv2D - Batch normalization (optional) - ReLU
    Basic Dropout (optional)
    '''

    def __init__(self, in_size, out_size, is_batchnorm, dropout, activation='relu'):
        '''
        Constructor of the convolutional block
        '''
        super(UnetConvBlock, self).__init__()

        # Convolutional layer with IN_SIZE --> OUT_SIZE
        conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))
        # Activation unit
        activ_unit1 = get_activation_unit(activation)
        # Add batch normalization if necessary
        if is_batchnorm:
            self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_size), activ_unit1)
        else:
            self.conv1 = nn.Sequential(conv1, activ_unit1)

        # Convolutional layer with OUT_SIZE --> OUT_SIZE
        conv2 = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))
        # Activation unit
        activ_unit2 = get_activation_unit(activation)
        # Add batch normalization if necessary
        if is_batchnorm:
            self.conv2 = nn.Sequential(conv2, nn.BatchNorm2d(out_size), activ_unit2)
        else:
            self.conv2 = nn.Sequential(conv2, activ_unit2)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None


    def forward(self, inputs):
        '''
        Do a forward pass
        '''
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        return outputs

class UnetUpsampling(nn.Module):
    '''
    Upsampling block of a U-Net:
    TransposeConvolution / Upsampling
    Convolutional block
    '''

    def __init__(self, in_size, out_size, upsample_size, is_deconv, dropout, is_batchnorm, activation='relu', upsampling_type='nearest'):
        '''
        Constructor of the upsampling block
        '''
        super(UnetUpsampling, self).__init__()

        if is_deconv:
            # first a transposed convolution
            self.up = nn.ConvTranspose2d(upsample_size, upsample_size, kernel_size=2, stride=2)
            # and then a convolution
            conv = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))   # convolution
            activ_unit1 = get_activation_unit(activation)                                                                              # activation
            if is_batchnorm:                                                                                                           # batch norm
                self.conv = nn.Sequential(conv, nn.BatchNorm2d(out_size), activ_unit1)
            else:
                self.conv = nn.Sequential(conv, activ_unit1)
        else:
            # first an upsampling operation
            self.up = nn.Upsample(scale_factor=2)
        # and then a convolutional block
        self.conv = UnetConvBlock(in_size, out_size, is_batchnorm, dropout, activation)


    def forward(self, from_skip_connection, from_lower_size):
        '''
        Do a forward pass
        '''

        # upsampling the input from the previous layer
        rescaled_input = self.up(from_lower_size)
        # verify the differences between the two tensors and apply padding
        offset = rescaled_input.size()[2] - from_skip_connection.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        from_skip_connection = F.pad(from_skip_connection, padding)
        # concatenate and apply the convolutional block on it
        return self.conv(torch.cat([from_skip_connection, rescaled_input], 1))

class Unet(SegmentationNetwork):
    '''
    CustomizedUnet U-Net architecture
    Same as in the original paper (https://arxiv.org/pdf/1505.04597.pdf)
    but you can modify:
    -- activation functions
    -- pooling layers
    -- dropout
    '''

    def __init__(self, config):
        '''
        Constructor of the standard U-Net
        '''

        super(Unet, self).__init__(config)

        # set the default configuration
        self.dropout = [0.0, 0.0, 0.0, 0.0, self.dropout, 0.0, 0.0, 0.0, 0.0]   # dropout rate of each layer
        filters_encoder = [64, 128, 256, 512, 1024]                             # number of channels of each conv block in the encoder
        filters_decoder = [64, 128, 256, 512]                                   # number of channels of each conv block in the decoder
        activation = 'relu'                                                     # activation units
        pooling = 'max'                                                         # type of pooling

        # change configuration if available in the file
        if 'filters-encoder' in config:
            filters_encoder = np.fromstring( config['architecture']['filters-encoder'], dtype=int, sep=',' )
        if 'filters-decoder' in config:
            filters_decoder = np.fromstring( config['architecture']['filters-decoder'], dtype=int, sep=',' )
        if 'dropout-list' in config:
            self.dropout = np.fromstring( config['architecture']['dropout-list'], dtype=float, sep=',' )
        if 'activation-unit' in config:
            activation = config['architecture']['activation-unit']
        if 'pooling' in config:
            pooling = config['architecture']['pooling']

        # Activation function to produce the scores
        self.final_activation = get_activation_function(config['architecture']['activation-function'])
        
        # downsampling
        self.conv1 = UnetConvBlock(1, int(filters_encoder[0]), self.is_batchnorm, self.dropout[0], activation)
        self.pool1 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv2 = UnetConvBlock(int(filters_encoder[0]), int(filters_encoder[1]), self.is_batchnorm, self.dropout[1], activation)
        self.pool2 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv3 = UnetConvBlock(int(filters_encoder[1]), int(filters_encoder[2]), self.is_batchnorm, self.dropout[2], activation)
        self.pool3 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv4 = UnetConvBlock(int(filters_encoder[2]), int(filters_encoder[3]), self.is_batchnorm, self.dropout[3], activation)
        self.pool4 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        # intermediate module
        self.conv5 = UnetConvBlock(int(filters_encoder[3]), int(filters_encoder[4]), self.is_batchnorm, self.dropout[4], activation)
        # upsampling
        self.up_concat4 = UnetUpsampling(int(filters_encoder[4]) + int(filters_encoder[3]), int(filters_decoder[3]), int(filters_encoder[4]), self.is_deconv, self.dropout[5], self.is_batchnorm, activation)
        self.up_concat3 = UnetUpsampling(int(filters_decoder[3]) + int(filters_encoder[2]), int(filters_decoder[2]), int(filters_encoder[3]), self.is_deconv, self.dropout[6], self.is_batchnorm, activation)
        self.up_concat2 = UnetUpsampling(int(filters_decoder[2]) + int(filters_encoder[1]), int(filters_decoder[1]), int(filters_encoder[2]), self.is_deconv, self.dropout[7], self.is_batchnorm, activation)
        self.up_concat1 = UnetUpsampling(int(filters_decoder[1]) + int(filters_encoder[0]), int(filters_decoder[0]), int(filters_encoder[1]), self.is_deconv, self.dropout[8], self.is_batchnorm, activation)
        # final conv (without any concat)
        self.final = nn.Conv2d(int(filters_decoder[0]), self.num_classes, 1)


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # downsampling
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool2(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool2(conv4)
        # intermediate module with dropout
        conv5 = self.conv5(pool4)
        # upsampling
        up4 = self.up_concat4(conv4, conv5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        # get the segmentation
        up1 = self.final(up1)
        # NO SE HACE LA ULTIMA CAPA DE ACTIVACION PORQUE LA LOSS YA LA TIENE INCORPORADA. SI LA HAGO APLICA 2 VECES LA CAPA) 
        #up1 = self.final_activation(up1)

        return up1


###############################################################################################################################



class UnetWithPyramidPooling(SegmentationNetwork):
    '''
    UNet with pyramid pooling in the skip connections
    '''

    def __init__(self, config):
        '''
        Constructor of the standard U-Net
        '''

        super(UnetWithPyramidPooling, self).__init__(config)


        # set the default configuration
        self.dropout = [0.0, 0.0, 0.0, 0.0, self.dropout, 0.0, 0.0, 0.0, 0.0]   # dropout rate of each layer
        filters_encoder = [64, 128, 256, 512, 1024]                             # number of channels of each conv block in the encoder
        filters_decoder = [64, 128, 256, 512]                                   # number of channels of each conv block in the decoder
        activation = 'relu'                                                     # activation units
        pooling = 'max'                                                         # type of pooling
        self.pooling_kernel_sizes = [2, 4, 8]                                   # size of the poolings for pyramid pooling

        # change configuration if available in the file
        if 'filters-encoder' in config['architecture']:
            filters_encoder = np.fromstring( config['architecture']['filters-encoder'], dtype=int, sep=',' )
        if 'filters-decoder' in config['architecture']:
            filters_decoder = np.fromstring( config['architecture']['filters-decoder'], dtype=int, sep=',' )
        if 'dropout-list' in config['architecture']:
            self.dropout = np.fromstring( config['architecture']['dropout-list'], dtype=float, sep=',' )
        if 'activation-unit' in config['architecture']:
            activation = config['architecture']['activation-unit']
        if 'pooling' in config['architecture']:
            pooling = config['architecture']['pooling']
        if 'pooling-sizes' in config['architecture']:
            self.pooling_kernel_sizes = np.fromstring(config['architecture']['pooling-sizes'],dtype=int,sep=',')

        # Activation function to produce the scores
        self.final_activation = get_activation_function(config['architecture']['activation-function'])
        
        # initialize a pooling operation
        self.pool = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        # initialize the pyramid pooling blocks
        self.spatial_pp = PyramidPoolingBlock(pooling, self.pooling_kernel_sizes)

        # convolutions
        self.conv1 = UnetConvBlock(1, int(filters_encoder[0]), self.is_batchnorm, self.dropout[0], activation)
        self.conv2 = UnetConvBlock(int(filters_encoder[0]), int(filters_encoder[1]), self.is_batchnorm, self.dropout[1], activation)
        self.conv3 = UnetConvBlock(int(filters_encoder[1]), int(filters_encoder[2]), self.is_batchnorm, self.dropout[2], activation)
        self.conv4 = UnetConvBlock(int(filters_encoder[2]), int(filters_encoder[3]), self.is_batchnorm, self.dropout[3], activation)
        self.conv5 = UnetConvBlock(int(filters_encoder[3]), int(filters_encoder[4]), self.is_batchnorm, self.dropout[4], activation)
        # upsampling
        self.up_concat4 = UnetUpsampling(int(filters_encoder[4]) + int(filters_encoder[3]) + int(filters_encoder[3]) * len(self.pooling_kernel_sizes), int(filters_decoder[3]), int(filters_encoder[4]), self.is_deconv, self.dropout[5], self.is_batchnorm, activation)
        self.up_concat3 = UnetUpsampling(int(filters_decoder[3]) + int(filters_encoder[2]) + int(filters_encoder[2]) * len(self.pooling_kernel_sizes), int(filters_decoder[2]), int(filters_encoder[3]), self.is_deconv, self.dropout[6], self.is_batchnorm, activation)
        self.up_concat2 = UnetUpsampling(int(filters_decoder[2]) + int(filters_encoder[1]) + int(filters_encoder[1]) * len(self.pooling_kernel_sizes), int(filters_decoder[1]), int(filters_encoder[2]), self.is_deconv, self.dropout[7], self.is_batchnorm, activation)
        self.up_concat1 = UnetUpsampling(int(filters_decoder[1]) + int(filters_encoder[0]) + int(filters_encoder[0]) * len(self.pooling_kernel_sizes), int(filters_decoder[0]), int(filters_encoder[1]), self.is_deconv, self.dropout[8], self.is_batchnorm, activation)
        # final conv (without any concat)
        self.final = nn.Conv2d(int(filters_decoder[0]), self.num_classes, 1)


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # downsampling and pyramid pooling
        conv1 = self.conv1(inputs)
        pyramid_pooling_1 = self.spatial_pp(conv1)
        pool1 = self.pool(conv1)
        conv2 = self.conv2(pool1)
        pyramid_pooling_2 = self.spatial_pp(conv2)
        pool2 = self.pool(conv2)
        conv3 = self.conv3(pool2)
        pyramid_pooling_3 = self.spatial_pp(conv3)
        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)
        pyramid_pooling_4 = self.spatial_pp(conv4)
        pool4 = self.pool(conv4)
        # intermediate module with dropout
        conv5 = self.conv5(pool4)
        # upsampling
        up4 = self.up_concat4(pyramid_pooling_4, conv5)
        up3 = self.up_concat3(pyramid_pooling_3, up4)
        up2 = self.up_concat2(pyramid_pooling_2, up3)
        up1 = self.up_concat1(pyramid_pooling_1, up2)
        # get the segmentation
        up1 = self.final(up1)
        # NO SE HACE LA ULTIMA CAPA DE ACTIVACION PORQUE LA LOSS YA LA TIENE INCORPORADA. SI LA HAGO APLICA 2 VECES LA CAPA) 
        #up1 = self.final_activation(up1)

        return up1



class PyramidPoolingBlock(nn.Module):
    '''
    Pyramid Pooling Block:
    It applies pyramid pooling at multiple levels to capture global details of the feature maps
    and to increase the effective receptive field of the network
    '''

    def __init__(self, pooling_type, pooling_sizes=[2, 4, 8]):
        '''
        Constructor of the convolutional block
        '''
        super(PyramidPoolingBlock, self).__init__()

        self.pooling_sizes = pooling_sizes

        self.pooling_layers = []
        for kernel in self.pooling_sizes:
            self.pooling_layers.append(get_pooling_layer(pooling_name=pooling_type, kernel_size=kernel))

    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        outputs = None
        pooling_layer_idx = 0
        for kernel_size in self.pooling_sizes:
            # apply the pooling layer on the inputs
            pooled_inputs = self.pooling_layers[pooling_layer_idx](inputs)
            # reshape to match the input size
            upsampled_pooled_inputs = F.upsample(pooled_inputs, scale_factor=kernel_size)
            # concatenate with the previous outputs
            if outputs is None:
                outputs = upsampled_pooled_inputs
            else:
                outputs = torch.cat([outputs, upsampled_pooled_inputs], 1)
            # advance the idx
            pooling_layer_idx = pooling_layer_idx + 1

        outputs = torch.cat([outputs, inputs], 1)

        return outputs

####################################################################################################################

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
