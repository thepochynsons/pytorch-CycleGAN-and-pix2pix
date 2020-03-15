import torch
from PIL import Image
from configparser import ConfigParser
from models import create_model
import scipy.io as spio
import numpy as np
from options.test_options import TestOptions
from models import networks
import glob


from scipy.misc import imsave
from torchvision import transforms

from os import listdir, makedirs, path


from data import CreateDataLoader


if __name__ == '__main__':

    dir_dataset='/home/svitale/proyectos/pytorch-CycleGAN-and-pix2pix/datasets/new_polaresSIM2US/trainB'
    output_path='/home/svitale/proyectos/pytorch-CycleGAN-and-pix2pix/results/trainRUStoAUS'
    models_dir='/home/svitale/proyectos/pytorch-CycleGAN-and-pix2pix/checkpoints/new_polaresSIM2US'

    # -------------------------------------
    # -------- Loading test-set -----------
    # -------------------------------------

    # collect the filenames
    test_data_filenames = listdir(dir_dataset) 
    # make results- directory
    makedirs(output_path, exist_ok=True)
      
    # -------------------------------------
    # -------- Loading the model ----------
    # -------------------------------------
    
    # Loading best_model
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.name = 'new_polaresSIM2US'
    #opt.name = 'polaresCT2US'
    opt.model = 'cycle_gan'
    opt.dataroot = './datasets/new_polaresSIM2US'
    #opt.dataroot = './datasets/polaresCT2US'
    opt.dataset_mode = 'unaligned'
        
    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    
    
    # for each saved epoch
    models_path = glob.glob(path.join(models_dir,'*_net_G_us.pth'))
    for p in models_path:
        model_file = path.basename(p)
        opt.epoch = model_file.split('_')[0]
        # Make epoch-result directory
        epoch_directory = path.join(output_path,opt.epoch)
        makedirs(epoch_directory, exist_ok=True)
        # load model

        model = create_model(opt)
        model.setup(opt)
        print('Model successfully loaded')
        '''
        netG_us = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, [])
        state_dict = torch.load(models_path)
        print(state_dict)
        netG_us.load_state_dict(state_dict)
        '''
        model.eval()
        model.netG_us.eval()
        # Paso por la red todas las imagenes del validation-set
        for filename in test_data_filenames:
            # Load and set imagen 
            image = Image.open(path.join(dir_dataset, filename))
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
            transformation = transforms.Compose(transform_list)
            image = transformation(image)
            image = image.numpy()
            image = np.expand_dims(image[0,:,:],axis=0)
            image = np.expand_dims(image, axis=0)
            image = torch.Tensor(image)
            image = image.cuda()
            #Forward
            with torch.no_grad():
                output = model.netG_us(image)
            
            # Save image
            transformed_image = output.cpu().numpy()
            transformed_image = transformed_image[0,0,:,:]
            filename = filename.split('.')[0]
            imsave(path.join(epoch_directory,filename + '.png'), transformed_image)
        
        print( str(opt.epoch) +' finished' )
    print('finished')
