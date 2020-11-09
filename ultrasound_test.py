import torch
from PIL import Image
from configparser import ConfigParser
from models import create_model
import scipy.io as spio
import numpy as np
from options.test_options import TestOptions
from models import networks
import glob
from imageio import imwrite
from torchvision import transforms
from os import listdir, makedirs, path
from data.base_dataset import __make_power_2, get_transform

if __name__ == '__main__':

    # -------------------------------------
    #      Loading lastest_model
    # -------------------------------------
        
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.model = 'cycle_gan'
    opt.dataset_mode = 'unaligned'
    opt.netG =  'unet_256'
    opt.preprocess = 'none'
    
    # Load CycleGan model
    cyclegan = create_model(opt)
    cyclegan.setup(opt)
    print('CycleGAN successfully loaded')
    # Set to evaluate model
    cyclegan.eval()
    
    # -------------------------------------
    #      set up directories and generator model
    # -------------------------------------      
    
    # Collect the filenames of test-set and select correct generator
    # Translate austorus 
    if (opt.direction == 'austorus'):
        test_data_dir = path.join(opt.dataroot,'testA')
        test_data_filenames = listdir(test_data_dir)
        current_generator =  cyclegan.netG_aus
    # Translate rustoaus
    else:
        test_data_dir = path.join(opt.dataroot,'testB')
        test_data_filenames = listdir(test_data_dir)
        current_generator =  cyclegan.netG_rus
    
    # Make results directory
    makedirs(opt.results_dir, exist_ok=True)

    # -------------------------------------
    #      Evaluate images
    # -------------------------------------
    
    # Transformation list    
    transformation = get_transform(opt, grayscale = True)
    
    # Select correct generator and test set.
    
    for filename in test_data_filenames:
        # Load and set imagen 
        image = Image.open(path.join(test_data_dir, filename)).convert('RGB')
        image = transformation(image)
        #Agrego el batchsize de 1
        image = torch.unsqueeze(image, dim=0)
        image = image.cuda()
       
        with torch.no_grad():
                fake_image = current_generator(image)
       
        # Save image 
        fake_image_output = fake_image[0,:,:,:].cpu().detach().numpy()
        filename = filename.split('.')[0]
        if fake_image.shape[0] == 1:  # grayscale to RGB
            fake_image_output = np.tile(fake_image_output, (3, 1, 1))
            fake_image_output = (np.transpose(fake_image_output, (1, 2, 0)) + 1) / 2.0 * 255.0
        imwrite(path.join(opt.results_dir,filename + '.png'), fake_image_output) 

    print('finished')
