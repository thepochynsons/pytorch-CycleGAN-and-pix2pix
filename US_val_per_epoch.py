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
from segm_net.util import string_to_list, parse_boolean, natural_key

if __name__ == '__main__':

    # -------------------------------------
    #      Preparar los datos para levantar la red
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
    
    # -------------------------------------
    #      set up directories and generator model
    # -------------------------------------      
    
    # Collect the filenames of validation-set and select correct gene
    # Translate austorus 
    test_data_dir = opt.dataset_dir
    test_data_filenames = listdir(test_data_dir)
    if (opt.direction == 'austorus'):
        # lista de todos los generadores aus guardados
        cyclegan_path = sorted(glob.glob(path.join(opt.model_dir,'*_net_G_aus.pth')), key=natural_key)
    # Translate rustoaus
    else:
        # lista de todos los generadores rus guardados
        cyclegan_path = sorted(glob.glob(path.join(opt.model_dir,'*_net_G_rus.pth')), key=natural_key)
    
    # Si me quiero quedar con una sola epoca   
    #cyclegan_path = [cyclegan_path[-2]]
    
    # Make results directory
    makedirs(opt.results_dir, exist_ok=True)

    # -------------------------------------
    #      Evaluate images
    # -------------------------------------
    
    # Transformation list    
    transformation = get_transform(opt, grayscale = True)
    
    # for each saved epoch
    for p in cyclegan_path:
        # obtengo el numero de epoca para setear el parametro epoch 
        cyclegan_file = path.basename(p)
        opt.epoch = cyclegan_file.split('_')[0]
        # Make epoch-result directory
        epoch_directory = path.join(opt.results_dir,opt.epoch)
        makedirs(epoch_directory, exist_ok=True)
 
        # Load cycleGAN model 
        cyclegan = create_model(opt)
        cyclegan.setup(opt)
        print('CycleGAN successfully loaded')
        # Lo pongo en modo eval
        cyclegan.eval()
        # Selecciono el generador correcto
        if (opt.direction == 'austorus'):
            current_generator =  cyclegan.netG_aus
        else:
            current_generator =  cyclegan.netG_rus
                
        # Paso por la red todas las imagenes del validation-set
        for filename in test_data_filenames:
            # Load and set imagen 
            image = Image.open(path.join(test_data_dir, filename)).convert('RGB')
            image = transformation(image)
            #Agrego el batchsize de 1
            image = torch.unsqueeze(image, dim=0)
            image = image.cuda()
       
            # Forward cycleGAN
            with torch.no_grad():
                    fake_image = current_generator(image)
        
            # Save image 
            fake_image_output = fake_image[0,:,:,:].cpu().detach().numpy()
            filename = filename.split('.')[0]
            if fake_image.shape[0] == 1:  # grayscale to RGB
                fake_image_output = np.tile(fake_image_output, (3, 1, 1))
                fake_image_output = (np.transpose(fake_image_output, (1, 2, 0)) + 1) / 2.0 * 255.0
            imwrite(path.join(epoch_directory,filename + '.png'), fake_image_output) 
       
        print( str(opt.epoch) +' finished' )
   
    print('finished')
