
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..util import parse_boolean
from torch.autograd import Variable

# List of supported schedulers
SUPPORTED_SCHEDULERS = ['ReduceLROnPlateau']



def setup_lr_scheduler(config, optimizer):
    '''
    Return a learning rate scheduler based on the given configuration
    '''

    if 'scheduler' in config['training']:

        # ReduceLROnPlateau (https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
        if config['training']['scheduler'] == 'ReduceLROnPlateau':
            
            # default configuration
            mode = 'max'
            threshold_mode= 'abs'
            factor = 0.5
            patience = 20
            verbose = True

            # redefine configuration based on the provided configuration
            if ('ReduceLROnPlateau' in config):
                if ('mode' in config['ReduceLROnPlateau']):
                    mode = config['ReduceLROnPlateau']['mode']
                elif ('threshold_mode' in config['ReduceLROnPlateau']):
                    threshold_mode = config['ReduceLROnPlateau']['threshold_mode']
                elif ('factor' in config['ReduceLROnPlateau']):
                    factor = float(config['ReduceLROnPlateau']['factor'])
                elif ('patience' in config['ReduceLROnPlateau']):
                    patience = int(config['ReduceLROnPlateau']['patience'])
                elif ('verbose' in config['ReduceLROnPlateau']):
                    verbose = parse_boolean(config['ReduceLROnPlateau']['verbose'])    

            # create the scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, threshold_mode=threshold_mode, factor=factor, patience=patience, verbose=verbose)

        else:
            # raise an error
            raise ValueError('Unknown learning rate scheduler ({} given, {} expected)'.format(config['training']['scheduler'], SUPPORTED_SCHEDULERS))

        # return the optimizer with the assign scheduler
        return optimizer, scheduler

    else:

        # return the optimizer and no scheduler
        return optimizer, None




class EarlyStopper(object):
    '''
    Class that implements an early stopper controller
    '''
    
    def __init__(self, config):
        '''
        Default initializer
        '''
        super(EarlyStopper, self).__init__()

        # assign the metric and the mode of analysis
        self.which_metric = config['training']['evaluation-metric']
        self.mode = config['training']['evaluation-metric-mode']
        # stop if needed
        self.use_early_stopping = False     # boolean saying if we use it or not
        if parse_boolean(config['training']['early-stopping']):
            self.use_early_stopping = True  # flag that we will use early stopping
        # patience before update
        self.patience = 20
        if 'patience' in config['early-stopping']:
            self.patience = int(config['early-stopping']['patience'])

        # this boolean is used to determine if the model has to stop or not
        self.should_stop = False
        # counter of remaining steps
        self.counter = 0

        # if mode is max
        if self.mode == 'max':
            self.best_performance = 0
        elif self.mode == 'min':
            self.best_performance = -10000


    def update(self, training_wrapper):
        '''
        Update after an epoch
        '''

        # get the validation metric
        if self.mode == 'max':
            validation_performance = training_wrapper.model_statistics['validation_stats'][self.which_metric][-1]
        elif self.mode == 'min':
            validation_performance = -1 * training_wrapper.model_statistics['validation_stats'][self.which_metric][-1]

        # check if the validation performance is better now
        if self.best_performance <= validation_performance:
            
            if self.mode == 'max':
                print('New best performance (Previous: {} - Now: {})'.format(self.best_performance, validation_performance))
            elif self.mode == 'min':
                print('New best performance (Previous: {} - Now: {})'.format(-1 * self.best_performance, -1 * validation_performance))
            self.best_performance = validation_performance
            print('Counter for early stopping reinitialized')
            self.counter = 0
            # don't stop
            self.should_stop = False

        else:

            # increase the counter
            self.counter = self.counter + 1
            # check if we need to stop or not
            if self.counter >= self.patience:
                self.should_stop = True
            else:
                self.should_stop = False


    def should_stop_training(self):
        '''
        Check if we need to stop
        '''

        # if using early stopping...
        if self.use_early_stopping:
            # print the message
            if self.should_stop:
                print('Early stopping!')
            else:
                print('{} epochs remaining for early stopping'.format(self.patience - self.counter))

        return (self.use_early_stopping) and (self.should_stop)




def update_optimizer(config, training_wrapper, learning_rate):
    '''
    Setup and update the optimizer
    '''

    if not ('weight-decay' in config['training']):
        weight_decay = 0.0
    else:
        weight_decay = float(config['training']['weight-decay'])
    
    # initialize the optimizer
    if config['training']['optimizer']=='SGD':
        optimizer = torch.optim.SGD(training_wrapper.parameters(), 
                                    lr = learning_rate, 
                                    momentum = float(config['training']['momentum']), 
                                    weight_decay = weight_decay)
    elif config['training']['optimizer']=='Adam':
        optimizer = torch.optim.Adam(training_wrapper.parameters(), 
                                    betas=[0.9, 0.999], 
                                    lr = learning_rate,
                                    weight_decay = weight_decay)
    elif config['training']['optimizer']=='Adamax':
        optimizer = torch.optim.Adamax(training_wrapper.parameters(), 
                                    betas=[0.9, 0.999], 
                                    lr = learning_rate,
                                    weight_decay = weight_decay)

    return optimizer




import sys
import time
import warnings

import _pickle as pickle
import scipy.io as io
import numpy as np

from ntpath import basename
from os import path, makedirs, remove
from glob import glob

from segm_net.util import natural_key, get_available_device
from segm_net.learning.losses import get_loss_function
from segm_net.learning.metrics import get_evaluation_metrics_function
from .architectures import segmentation_networks as segmentation_networks


class TrainingWrapper(nn.Module):
    '''
    This class wraps all the routines for training a given model
    '''

    def __init__(self, config, model):
        '''
        Constructor of the basic training wrapper
        '''
        super(TrainingWrapper, self).__init__()

        # set the name of the experiment
        self.experiment_name = config['experiment']['name']

        # the configuration of the experiment
        self.config = config
        # the model
        self.model = model
        # the loss 
        self.loss, self.loss_name = get_loss_function(self.config)
        # the optimizer
        self.optimizer = update_optimizer(config, self, float(config['training']['learning-rate']))
        # the learning rate scheduler
        self.optimizer, self.lr_scheduler = setup_lr_scheduler(config, self.optimizer)                     
        # the evaluation metric
        self.eval_metric_name = self.config['training']['evaluation-metric']
        if not(self.eval_metric_name == 'loss'):
            self.eval_metric = get_evaluation_metrics_function(self.eval_metric_name)
        else:
            self.eval_metric_name = 'loss_metric'
            self.eval_metric = None
        self.eval_metric_mode = self.config['training']['evaluation-metric-mode']
        # the early stopper
        if parse_boolean(config['training']['early-stopping']):
            self.early_stopper = EarlyStopper(config)
        else:
            self.early_stopper = None

        # a boolean value indicating if the best model was changed
        self.best_model_has_changed = False
        # initialize the iteration number
        self.iteration_num = 0
        # assign metric to monitor the best model and the mode (if using early stopping, we will use the same)
        self.metric_to_monitor_for_best = self.eval_metric_name
        self.metric_to_monitor_for_best_mode = self.eval_metric_mode
        # initialize the log
        self.model_statistics = dict()
        self.model_statistics['training_stats'] = dict()
        self.model_statistics['training_stats']['loss'] = []
        self.model_statistics['validation_stats'] = dict()
        self.model_statistics['validation_stats']['loss'] = []
        if self.metric_to_monitor_for_best_mode == 'max':
            self.model_statistics['validation_stats']['best_metric'] = -sys.maxsize
        elif self.metric_to_monitor_for_best_mode == 'min':
            self.model_statistics['validation_stats']['best_metric'] = sys.maxsize
        self.model_statistics['validation_stats'][self.eval_metric_name] = []
        self.model_statistics['epochs'] = 0
        self.model_statistics['time'] = []
        # initialize the iteration log
        self.iteration_log = dict()
        self.iteration_log['training_stats'] = dict()
        self.iteration_log['training_stats']['loss'] = []
        self.iteration_log['validation_stats'] = dict()
        self.iteration_log['validation_stats']['loss'] = []
        self.iteration_log['validation_stats'][self.eval_metric_name] = []
        # initialize a folder to output intermediate results for visualization
        self.intermediate_results_folder = path.join(config['data']['output-folder'], self.experiment_name, 'intermediate-results')
        makedirs(self.intermediate_results_folder, exist_ok=True)

        # detect if we have a GPU available
        self.device = get_available_device()
        # move the model to the device
        self.model = self.model.to(self.device)

    
    def forward(self, inputs):
        '''
        Forward pass
        '''
        # a forward pass of the model
        return self.model.forward(inputs)

    
    def start_epoch(self):
        '''
        Start the epoch
        '''
        # initialize the timer
        self.initial_time = time.time()
        # flush the iteration log to start again from scratch
        self.iteration_log = dict()
        self.iteration_log['training_stats'] = dict()
        self.iteration_log['training_stats']['loss'] = []
        self.iteration_log['validation_stats'] = dict()
        self.iteration_log['validation_stats']['loss'] = []
        self.iteration_log['validation_stats'][self.eval_metric_name] = []

    
    def get_metric_to_monitor_for_best(self):
        '''
        Return the validation metric used for monitoring the best model
        '''

        # retrieve the value
        retrieved_value = self.model_statistics['validation_stats'][self.eval_metric_name][-1]
        # correct it depending on the mode
        if self.metric_to_monitor_for_best_mode == 'min':
            retrieved_value = retrieved_value * -1

        return retrieved_value
        

    def finish_epoch(self):
        '''
        Finish the epoch
        '''
        self.model_statistics['time'].append( time.time() - self.initial_time )
        print('This epoch took {} seconds to finish'.format(self.model_statistics['time'][-1]))


    def load_checkpoint(self, dir_checkpoints):
        '''
        Load a checkpoint to resume training
        '''

        print('Loading checkpoint...')
        print('Loading model:')
        # get the list of all the checkpoints in the folder
        checkpoints_filenames = sorted(glob(path.join(dir_checkpoints, '*_checkpoint.pt')), key=natural_key)
        # and log filenames
        log_filenames = sorted(glob(path.join(dir_checkpoints, '*_log.pkl')), key=natural_key)
        # and early stoppers
        if parse_boolean(self.config['training']['early-stopping']):
            early_stopper_filenames = sorted(glob(path.join(dir_checkpoints, '*_early_stopper.pkl')), key=natural_key)

        # if there is one
        if len(checkpoints_filenames) > 0:

            # identify the last checkpoint
            checkpoint_name = basename(checkpoints_filenames[-1])
            print('Last model found: {}'.format(checkpoint_name))

            # prepare log filename
            log_pickle_file = log_filenames[-1]
            # and the early stopper
            if parse_boolean(self.config['training']['early-stopping']):
                early_stopper_pickle_file = early_stopper_filenames[-1]

            # load the model in that checkpoint
            loaded_checkpoint = torch.load(checkpoints_filenames[-1])
            self.model.load_state_dict(loaded_checkpoint['model'])
            print('Model successfully loaded')

            # load the log
            print('Loading previous log:')
            with open(log_pickle_file, 'rb') as handle:
                self.model_statistics = pickle.load(handle)
            print('Log loaded')
            self.save_log(path.join(dir_checkpoints, 'log.mat'), 'mat')

            # load optimizer, scheduler and early stopper
            self.optimizer.load_state_dict(loaded_checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(loaded_checkpoint['scheduler'])
            # load the early stopping object
            if parse_boolean(self.config['training']['early-stopping']):
                if path.exists(early_stopper_pickle_file):
                    print('Loading early stopper:')
                    with open(early_stopper_pickle_file, 'rb') as handle:
                        self.early_stopper = pickle.load(handle)
                    print('Early stopper loaded')
                else:
                    raise ValueError('Unable to find {} file with the early stopping object.')

            return True

        else:

            warnings.warn('Unable to find checkpoints in {}. Starting from epoch = 0.'.format(dir_checkpoints))
            return False


    def get_current_epoch(self):
        '''
        Get current epoch
        '''
        return self.model_statistics['epochs']


    def should_continue_training(self):
        '''
        Determine if training must continue or not
        '''

        # check if the early stopper wants to stop training
        if parse_boolean(self.config['training']['early-stopping']):
            should_stop_by_early_stopping = self.early_stopper.should_stop_training()
        else:
            should_stop_by_early_stopping = False

        # continue training if the epoch is not the last one and if we should not do early stopping
        return (self.get_current_epoch() < int(self.config['training']['epochs'])) and not (should_stop_by_early_stopping)


    def load_log(self, log_pickle_file):
        '''
        Load the file
        '''
        with open(log_pickle_file, 'rb') as handle:
            self.model_statistics = pickle.load(handle)

        
    def save_log(self, log_pickle_file, extension='pkl'):
        '''
        Save the log to a pickle file
        '''

        if extension=='pkl':
            # save the file
            with open(log_pickle_file, 'wb') as handle:
                pickle.dump(self.model_statistics, handle)
        elif extension=='mat':
            # save the file
            io.savemat(log_pickle_file, self.model_statistics)


    def plot_log(self, plotter):
        '''
        Plot the log 
        '''
        for i in range(len(self.model_statistics['training_stats']['loss'])):
            self.update_plot(plotter, epoch=i)

        
    def plot_outputs(self, plotter, input_data):
        '''
        Plot the outputs of the model
        '''
        # use the model routine to plot the outputs and get the outputs to save them
        outputs = self.model.plot_outputs(plotter, input_data)
        # save intermediate results
        self.model.save_intermediate_results(outputs, self.intermediate_results_folder, self.model_statistics['epochs'])


    def update_plot(self, plotter, epoch=None):
        '''
        Update the plot of the log with the new values
        '''

        # if epoch is none, plot the last one
        if epoch is None:
            epoch = self.get_current_epoch()
        # plot the epoch
        plotter.plot_multiple_statistics('complete loss',
                                         epoch,
                                         {'train': self.model_statistics['training_stats']['loss'][epoch],
                                         'validation': self.model_statistics['validation_stats']['loss'][epoch]})
        plotter.plot_scalar(self.eval_metric_name,
                            epoch,
                            self.model_statistics['validation_stats'][self.eval_metric_name][epoch],
                            'validation')

        
    def compute_loss(self, inputs, targets):
        '''
        Compute the loss value
        '''
        # do a forward pass to get the output
        predicted = self.forward(inputs)
        # compute the loss value and return
        return self.loss.forward(predicted, targets), predicted


    def evaluate_loss(self, loaded_data, training=True, plotter=None):
        '''
        Evaluate the model on a batch inputs according to the targets,
        and update the iteration statistics
        '''

        # format input data and move it to the right device
        inputs, targets = self.model.format_loaded_data(loaded_data)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # do a forward pass
        if training:
            # put the model in train mode
            self.train()
            self.model.train()
            # clear the gradients
            self.optimizer.zero_grad()
            # compute the loss function
            loss_value, predicted = self.compute_loss(inputs, targets)
            # perform a backward pass
            loss_value.backward()
            # update parameters
            self.optimizer.step()
            # indicate the statistics to update
            log_to_update = 'training_stats'
        else:
            # put the model in evaluation mode
            self.eval()
            self.model.eval()
            # don't compute any gradient
            with torch.no_grad():
                # compute the loss function
                loss_value, predicted = self.compute_loss(inputs, targets)
                # indicate the statistics to update
                log_to_update = 'validation_stats'
        
        # accumulate the loss value in the iteration log
        self.iteration_log[log_to_update]['loss'].append(loss_value.item())
        if not training:
            # accumulate the self.eval_metric_name values if it is not a training phase
            if self.eval_metric_name == 'loss_metric':
                self.iteration_log['validation_stats'][self.eval_metric_name].append(loss_value.item())
            else:
                metric_value = self.model.evaluate(targets, predicted, self.eval_metric)
                self.iteration_log['validation_stats'][self.eval_metric_name].append(metric_value)

        # update the iteration number
        self.iteration_num +=1

        # plot the minibatch loss if necessary
        if not (plotter is None):
            plotter.plot_scalar('minibatch loss (complete)', self.iteration_num-1, loss_value.data.item(), str(self.get_current_epoch()))

        # return the loss value
        return loss_value


    def update_log(self, plotter=None):
        '''
        Updates the log. Use this at the end of an epoch, to update
        the statistics and the best model
        '''

        # reset the iteration number
        self.iteration_num = 0

        # compute the mean training and validation losses
        self.model_statistics['training_stats']['loss'].append( np.mean(np.asarray(self.iteration_log['training_stats']['loss'])) )
        self.model_statistics['validation_stats']['loss'].append( np.mean(np.asarray(self.iteration_log['validation_stats']['loss'])) )
        # assign the mean SANTI- CAMBIE MEAN x NANMEAN
        self.model_statistics['validation_stats'][self.eval_metric_name].append( np.nanmean(np.asarray(self.iteration_log['validation_stats'][self.eval_metric_name])) )

        # if a certain metric in the current epoch is lower than the best metric, update
        current_epoch_metric = self.get_metric_to_monitor_for_best()
        if current_epoch_metric > self.model_statistics['validation_stats']['best_metric']:

            # print the update of the best value
            if (self.metric_to_monitor_for_best_mode == 'max'):
                previous_best = self.model_statistics['validation_stats']['best_metric']
                current_best = current_epoch_metric
            elif (self.metric_to_monitor_for_best_mode == 'min'):
                previous_best = self.model_statistics['validation_stats']['best_metric'] * -1
                current_best = current_epoch_metric * -1
            print(' *** New best model: Previous val {} was {} and now is {}).'.format(self.metric_to_monitor_for_best, current_best, previous_best))    

            # update the best loss
            self.model_statistics['validation_stats']['best_metric'] = current_epoch_metric
            # indicate that the best model was changed
            self.best_model_has_changed = True
        else:
            # indicate that the best model has not changed
            self.best_model_has_changed = False

        # update plot
        if not (plotter is None):
            self.update_plot(plotter, self.get_current_epoch())

        # one more epoch finished
        self.model_statistics['epochs'] += 1 

        return self.model_statistics['validation_stats'][self.eval_metric_name][-1]


    def checkpoint(self, dir_checkpoints):
        '''
        Save a checkpoint
        '''

        # turn the wrapper and the model in train mode, just in case
        self.train()
        self.model.train()
        # initialize checkpoint name and filename
        checkpoint_filename = path.join(dir_checkpoints, "{}_{}_checkpoint.pt".format(self.experiment_name, self.model_statistics['epochs']))
        # save current checkpoint: model, optimizer, learning rate scheduler
        if self.lr_scheduler is None:
            state_to_save = {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict()}
        else:
            state_to_save = {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'scheduler': self.lr_scheduler.state_dict()}
        torch.save(state_to_save, checkpoint_filename)
        # remove previous checkpoint file
        previous_checkpoint_filename = path.join(dir_checkpoints, "{}_{}_checkpoint.pt".format(self.experiment_name, self.model_statistics['epochs']-1))
        if path.exists(previous_checkpoint_filename):
            remove(previous_checkpoint_filename)

        # save the best model in a different file, if necessary
        if self.best_model_has_changed:
            # retrieve the state of the mode
            state_to_save = {'model': self.model.state_dict()}
            torch.save(state_to_save, path.join(dir_checkpoints, 'best_model.pt'))

        # save early stopper (if available)
        if parse_boolean(self.config['training']['early-stopping']):
            # save early stopper    
            early_stopper_file = path.join(dir_checkpoints, '{}_early_stopper.pkl'.format(self.model_statistics['epochs']))
            with open(early_stopper_file, 'wb') as handle:
                pickle.dump(self.early_stopper, handle)
            # remove previous early stopper
            previous_early_stopper_filename = path.join(dir_checkpoints, "{}_early_stopper.pkl".format(self.model_statistics['epochs']-1))
            if path.exists(previous_early_stopper_filename):
                remove(previous_early_stopper_filename)

        # save the log
        self.save_log(path.join(dir_checkpoints, '{}_log.pkl'.format(self.model_statistics['epochs']))) 
        self.save_log(path.join(dir_checkpoints, '{}_log.mat'.format(self.model_statistics['epochs'])), 'mat') 
        # remove previous log file
        previous_log_filename_pkl = path.join(dir_checkpoints, "{}_log.pkl".format(self.model_statistics['epochs']-1))
        previous_log_filename_mat = path.join(dir_checkpoints, "{}_log.mat".format(self.model_statistics['epochs']-1))
        if path.exists(previous_log_filename_pkl):
            remove(previous_log_filename_pkl)
            remove(previous_log_filename_mat)


    def update_schedulers(self):
        '''
        Update the schedulers
        '''

        # update the early stopper
        if parse_boolean(self.config['training']['early-stopping']):
            self.early_stopper.update(self)

        # update the learning rate using the scheduler
        if not (self.lr_scheduler is None):
            self.lr_scheduler.step(self.model_statistics['validation_stats'][self.eval_metric_name][-1])
            # print current learning rate
            for param_group in self.optimizer.param_groups:
                print('Learning rate: {}'.format(param_group['lr']))

################################################################
#Training wrapper para modelos con una adversarial loss




class AdversarialLossTrainingWrapper(TrainingWrapper):
    '''
    This class wraps all the routines for training a given model
    '''

    def __init__(self, config, model):
        '''
        Constructor of the Multinet-multiloss training wrapper
        '''
        super(AdversarialLossTrainingWrapper, self).__init__(config,model)
        # losses
        self.loss, self.loss_name = get_loss_function(self.config)
        # Los pesos puede venir de archivo config
        self.loss_weights = loss_weights = np.ones(2, dtype=np.float)
        # add Discriminator Network
        # por default usa batchnorm
        self.discriminator = segmentation_networks.NLayerDiscriminator(n_layers=3)
        self.d_loss = nn.MSELoss()
        self.d_optimizer = update_optimizer(config, self, float(config['training']['learning-rate']))
        # lo metemos en la gpu
        self.discriminator.cuda()
        self.ones = torch.tensor(1.0).cuda()
        self.zeros = torch.tensor(0.0).cuda()
        
        
        # add new fields to the logs, one for each loss function
        for i in range(len(self.loss_name)):
            self.model_statistics['training_stats'][self.loss_name[i]+str(i+1)] = []
            self.model_statistics['validation_stats'][self.loss_name[i]+str(i+1)] = []    
            self.iteration_log['training_stats'][self.loss_name[i]+str(i+1)] = []
            self.iteration_log['validation_stats'][self.loss_name[i]+str(i+1)] = []
        # D-loss
        self.model_statistics['training_stats']['d_loss'] = []
        self.model_statistics['validation_stats']['d_loss'] = []    
        self.iteration_log['training_stats']['d_loss'] = []
        self.iteration_log['validation_stats']['d_loss'] = []

    def start_epoch(self):
        '''
        Start the time
        '''
        # start epoch
        super().start_epoch()
        self.iteration_log['training_stats']['weights'] = dict()
        # flush the iteration log to start again from scratch
        for i in range(len(self.loss_name)):
            self.iteration_log['training_stats'][self.loss_name[i]+str(i+1)] = []
            self.iteration_log['validation_stats'][self.loss_name[i]+str(i+1)] = []
            self.iteration_log['training_stats']['weights'][self.loss_name[i]+str(i+1)] = []
        self.iteration_log['training_stats']['d_loss'] = []
        self.iteration_log['validation_stats']['d_loss'] = []
        self.iteration_log['training_stats']['weights']['d_loss'] = []
    
    def update_log(self, plotter=None):
        '''
        Updates the log. Use this at the end of an epoch, to update
        the statistics and the best model
        '''

        # do the classical update
        validation_performance = super().update_log()
        # also update the individual losses
        for i in range(len(self.loss_name)):
            self.model_statistics['training_stats'][self.loss_name[i]+str(i+1)].append( np.mean(np.asarray(self.iteration_log['training_stats'][self.loss_name[i]+str(i+1)])) )
            self.model_statistics['validation_stats'][self.loss_name[i]+str(i+1)].append( np.mean(np.asarray(self.iteration_log['validation_stats'][self.loss_name[i]+str(i+1)])) )
        self.model_statistics['training_stats']['d_loss'].append(np.mean(np.asarray(self.iteration_log['training_stats']['d_loss']))) 
        self.model_statistics['validation_stats']['d_loss'].append(np.mean(np.asarray(self.iteration_log['validation_stats']['d_loss']))) 
        
        return validation_performance

    def update_plot(self, plotter, epoch=None):
        '''
        Update the plot of the log with the new values
        '''
        # plot the standard losses
        super().update_plot(plotter, epoch)
        # if epoch is none, plot the last one
        if epoch is None:
            epoch = self.get_current_epoch()
        # plot each of the complementary losses
        for i in range(len(self.loss_name)):
            plotter.plot(self.loss_name[i]+str(i+1), 'train', epoch, self.model_statistics['training_stats'][self.loss_name[i]+str(i+1)][epoch])
            plotter.plot(self.loss_name[i]+str(i+1), 'validation', epoch, self.model_statistics['validation_stats'][self.loss_name[i]+str(i+1)][epoch])
            # close the minibatch plot every 5 iterations
            if (epoch > 0) and ((epoch % 5) == 0):
                plotter.close_plot(self.loss_name[i]+str(i+1) + ' - minibatch')
        # plot discriminator losses
        plotter.plot('d_loss', 'train', epoch, self.model_statistics['training_stats']['d_loss'][epoch])
        plotter.plot('d_loss', 'validation', epoch, self.model_statistics['validation_stats']['d_loss'][epoch])
    
    def compute_loss(self, inputs, labels):
        '''
        Compute the loss value
        '''
        # do a forward pass to get the output
        predicted = self.forward(inputs)
        fake_targets = torch.argmax(predicted, dim=1)
        fake_targets =  torch.unsqueeze(fake_targets, dim=1).float()
        # initialize a list to accumulate all the intermediate losses
        losses_values = []
        # compute all the losses
        losses_values.append(self.loss[0].forward(predicted, labels))
        # adversarial loss
        d_predicted = self.discriminator.forward(fake_targets)
        target = self.ones.expand_as(d_predicted) #True
        losses_values.append(self.loss[1].forward(d_predicted, target)) 
        # return this list of losses and the predicted value
        return losses_values, predicted
    
    def d_backward(self, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            real (tensor array) -- real label
            fake (tensor array) -- label generated by a segmenter

        Return the discriminator loss.
        """
        # Real
        pred_real = self.discriminator.forward(real)
        target = self.ones.expand_as(pred_real) #True
        loss_value_real = self.d_loss(pred_real, target) 
        # Fake
        pred_fake = self.discriminator.forward(fake.detach())
        target = self.zeros.expand_as(pred_fake) #False
        loss_value_fake = self.d_loss(pred_fake, target) 
        # Combined loss and calculate gradients
        loss_value = (loss_value_real + loss_value_fake) * 0.5
        return loss_value

    def evaluate_loss(self, loaded_data, training=True, plotter=None):
        '''
        Evaluate the model on a batch inputs according to the targets,
        and update the iteration statistics
        '''

        # format input data and move it to the right device
        inputs, targets = self.model.format_loaded_data(loaded_data)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # do a forward pass
        if training:
            # put the model in train mode
            self.train()
            self.model.train()
            self.discriminator.train()
            
            # clear the gradients
            self.optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            
            # compute the loss function
            losses_values, predicted = self.compute_loss(inputs, targets)
            fake_targets = torch.argmax(predicted, dim=1)
            fake_targets =  torch.unsqueeze(fake_targets, dim=1).float()
            # perform a backward pass
            # discriminator
            d_loss_value = self.d_backward(torch.unsqueeze(targets, dim=1).float(),fake_targets)
            d_loss_value.backward()
            # model
            loss_value = torch.mul(losses_values[0], self.loss_weights[0]) 
            for i in range(1, len(self.loss_name)):
                loss_value = loss_value + torch.mul(losses_values[i], self.loss_weights[i])
            loss_value.backward()
            
            # update parameters
            self.optimizer.step()
            self.d_optimizer.step()
            # indicate the statistics to update
            log_to_update = 'training_stats'
        else:
            # put the model in evaluation mode
            self.model.eval()
            self.eval()
            self.discriminator.eval()
            # don't compute any gradient
            with torch.no_grad():
                # compute the loss function
                losses_values, predicted = self.compute_loss(inputs, targets)
                # compute the actual discriminator loss value
                fake_targets = torch.argmax(predicted, dim=1)
                fake_targets =  torch.unsqueeze(fake_targets, dim=1).float()
                d_loss_value = self.d_backward(torch.unsqueeze(targets, dim=1).float(),fake_targets)
                # compute the actual loss value, doing the weighted sum # of the individual losses
                loss_value = torch.mul(losses_values[0], self.loss_weights[0])
                for i in range(1, len(self.loss_name)):
                    loss_value = loss_value + torch.mul(losses_values[i], self.loss_weights[i])
                # indicate the statistics to update
                log_to_update = 'validation_stats'
                
        
        # accumulate the loss value in the iteration log
        self.iteration_log[log_to_update]['loss'].append(loss_value.item())
        # also incorporate discriminator loss
        self.iteration_log[log_to_update]['d_loss'].append(d_loss_value.item())
        # also incorporate the individual losses
        for i in range(len(self.loss_name)):
            self.iteration_log[log_to_update][self.loss_name[i]+str(i+1)].append(losses_values[i].item())
        if not training:
            # accumulate the self.eval_metric_name values if it is not a training phase
            if self.eval_metric_name == 'loss_metric':
                self.iteration_log['validation_stats'][self.eval_metric_name].append(loss_value.item())
            else:
                metric_value = self.model.evaluate(targets, predicted, self.eval_metric)
                self.iteration_log['validation_stats'][self.eval_metric_name].append(metric_value)

        # update the iteration number
        self.iteration_num +=1

        # plot the minibatch loss if necessary
        if not (plotter is None):
            plotter.plot('minibatch loss (complete)', self.model_statistics['epochs'], self.iteration_num-1, loss_value.data.item(), 'minibatch')
            for i in range(len(self.loss_name)):
                plotter.plot(self.loss_name[i]+str(i+1) + ' - minibatch', self.model_statistics['epochs'], self.iteration_num-1, losses_values[i].item(), 'minibatch')

        # return the loss value
        return loss_value

    def checkpoint(self, dir_checkpoints):
        '''
        Save a checkpoint
        '''
        
        # turn the wrapper and the model in train mode, just in case
        self.train()
        self.model.train()
        # initialize checkpoint name and filename
        checkpoint_filename = path.join(dir_checkpoints, "{}_{}_checkpoint.pt".format(self.experiment_name, self.model_statistics['epochs']))
        # save current checkpoint: model, discriminator, optimizers, learning rate scheduler
        if self.lr_scheduler is None:
            state_to_save = {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'discriminator': self.discriminator.state_dict(),
                             'd_optimizer':self.d_optimizer.state_dict()}
        else:
            state_to_save = {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'discriminator': self.discriminator.state_dict(),
                             'd_optimizer':self.d_optimizer.state_dict(),
                             'scheduler': self.lr_scheduler.state_dict()}
        torch.save(state_to_save, checkpoint_filename)
        # remove previous checkpoint file
        previous_checkpoint_filename = path.join(dir_checkpoints, "{}_{}_checkpoint.pt".format(self.experiment_name, self.model_statistics['epochs']-1))
        if path.exists(previous_checkpoint_filename):
            remove(previous_checkpoint_filename)

        # save the best model in a different file, if necessary
        if self.best_model_has_changed:
            # retrieve the state of the mode
            state_to_save = {'model': self.model.state_dict()}
            torch.save(state_to_save, path.join(dir_checkpoints, 'best_model.pt'))

        # save early stopper (if available)
        if parse_boolean(self.config['training']['early-stopping']):
            # save early stopper    
            early_stopper_file = path.join(dir_checkpoints, '{}_early_stopper.pkl'.format(self.model_statistics['epochs']))
            with open(early_stopper_file, 'wb') as handle:
                pickle.dump(self.early_stopper, handle)
            # remove previous early stopper
            previous_early_stopper_filename = path.join(dir_checkpoints, "{}_early_stopper.pkl".format(self.model_statistics['epochs']-1))
            if path.exists(previous_early_stopper_filename):
                remove(previous_early_stopper_filename)

        # save the log
        self.save_log(path.join(dir_checkpoints, '{}_log.pkl'.format(self.model_statistics['epochs']))) 
        self.save_log(path.join(dir_checkpoints, '{}_log.mat'.format(self.model_statistics['epochs'])), 'mat') 
        # remove previous log file
        previous_log_filename_pkl = path.join(dir_checkpoints, "{}_log.pkl".format(self.model_statistics['epochs']-1))
        previous_log_filename_mat = path.join(dir_checkpoints, "{}_log.mat".format(self.model_statistics['epochs']-1))
        if path.exists(previous_log_filename_pkl):
            remove(previous_log_filename_pkl)
            remove(previous_log_filename_mat)


    def load_checkpoint(self, dir_checkpoints):
        '''
        Load a checkpoint to resume training
        '''
        # Load Base Model
        super().load_checkpoint(dir_checkpoints)
        print('Loading checkpoint...')
        print('Loading model:')
        # get the list of all the checkpoints in the folder
        checkpoints_filenames = sorted(glob(path.join(dir_checkpoints, '*_checkpoint.pt')), key=natural_key)
        # if there is one
        if len(checkpoints_filenames) > 0:
            # load the discriminator and it's optimizer
            loaded_checkpoint = torch.load(checkpoints_filenames[-1])
            self.discriminator.load_state_dict(loaded_checkpoint['discriminator'])
            print('Model successfully loaded')
            self.d_optimizer.load_state_dict(loaded_checkpoint['d_optimizer'])
    
    