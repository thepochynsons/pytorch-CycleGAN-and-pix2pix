
import torch
from configparser import ConfigParser
from os import path, makedirs

from segm_net.util import string_to_list, parse_boolean
from segm_net.learning.architectures.model_factory import ModelFactory
from segm_net.learning.data_loaders import initialize_data_loader
from segm_net.learning.training import setup_lr_scheduler, EarlyStopper, update_optimizer
from segm_net.learning.visualization import get_plotter

def train_on_single_split(config, training_data_filenames, validation_data_filenames, resume_training=False):
    '''
    Train a model on a given split into training and validation
    '''

    # get the experiment name from the configuration file
    experiment_name = config['experiment']['name']

    # retrieve and prepare the output path
    dir_output = path.join(config['data']['output-folder'], experiment_name)
    makedirs(dir_output, exist_ok=True)
    # initialize a folder for the checkpoints
    dir_checkpoints = path.join(dir_output, 'checkpoints')
    if not path.exists(dir_checkpoints):
        makedirs(dir_checkpoints)

    # build the model architecture using the model factory
    model = ModelFactory.get_model(config)
    # create a training wrapper for it
    training_wrapper = ModelFactory.get_training_wrapper(model, config)

    # setup the training and validation data sets and data loaders
    _, train_loader = initialize_data_loader(config, 'training', training_data_filenames)
    validation_set, validation_loader = initialize_data_loader(config, 'validation', validation_data_filenames)
    # collect a random sample from the validation set for the plot
    data_sample, filenames_on_sample = validation_set.get_sample_for_plot(sample_size=int(config['training']['sample-for-validation']))

    # get total number of epochs
    total_number_epochs = int(config['training']['epochs'])
    
    # setup the visualizer
    plotter = get_plotter(config)

    # load pretrained weights if possible
    if resume_training:
        # load the checkpoint
        successful = training_wrapper.load_checkpoint(dir_checkpoints)
        if successful:
            # plot the statistics in the training wrapper
            training_wrapper.plot_log(plotter)
            # plot the outputs
            training_wrapper.plot_outputs(plotter, data_sample)

    # **********************
    # *** Start training ***
    # **********************

    # repeat training sequence until the training wrapper decides to stop
    while training_wrapper.should_continue_training():

        # -------------------------------------
        # ----------- Training part -----------
        # -------------------------------------

        # start epoch
        training_wrapper.start_epoch()

        print("========== Epoch [{}/{}] ==========".format(training_wrapper.get_current_epoch()+1, total_number_epochs))

        # iterate for each batch and do a forward and backward pass
        for _, loaded_data in enumerate(train_loader):
            # compute the loss function
            _ = training_wrapper.evaluate_loss(loaded_data, training=True, plotter=plotter)

        # finish epoch
        training_wrapper.finish_epoch()

        # -------------------------------------
        # --------- Validation part -----------
        # -------------------------------------

        # run validation
        print('Evaluating on validation set...')
        # iterate for each batch of validation samples
        for _, loaded_data in enumerate(validation_loader):
            # compute the loss function
            _ = training_wrapper.evaluate_loss(loaded_data, training=False)
        print('Validation finished')

        # update the stats
        validation_performance = training_wrapper.update_log(plotter)
        print('Validation performance: {}'.format(validation_performance))

        # evaluate the model on the image sample
        print('Retrieving responses on random sample from validation data...')
        training_wrapper.plot_outputs(plotter, data_sample)
        print('Done')

        # -------------------------------------
        #  Update schedulers and early stopper
        # -------------------------------------

        training_wrapper.update_schedulers()

        # -------------------------------------
        # ---------- Checkpointing ------------
        # -------------------------------------

        # save the checkpoint
        training_wrapper.checkpoint(dir_checkpoints)

    # return the training wrapper and output dir
    return training_wrapper, dir_output




def train_on_holdout_set(config, resume_training=False):
    '''
    Train a model using a hold-out set (rigit split into training, validation and test)
    '''
    
    # read the data split
    data_split = ConfigParser()
    data_split.read(config['data-split']['split-file'])
    print(data_split)
    # get the filenames of the training and validation samples
    training_data_filenames = string_to_list(data_split['split']['training'])
    validation_data_filenames = string_to_list(data_split['split']['validation'])

    # train on current split
    training_wrapper, dir_output = train_on_single_split(config, training_data_filenames, validation_data_filenames, resume_training)

    # save the final model
    torch.save(training_wrapper, path.join(dir_output, '{}.pkl'.format(config['experiment']['name'])))

    


import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="full path and filename of the configuration file", type=str)
    parser.add_argument("--resume", help="a boolean indicating if we have to resume training", type=str, default='False')
    args = parser.parse_args()

    # read the configuration file
    config = ConfigParser()
    config.read(args.config)

    # depending on the split type, the training routine
    if config['data-split']['type'] == 'hold-out':
        # train the model on a holdout set
        train_on_holdout_set(config, parse_boolean(args.resume))

