import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as numpy
from math import exp
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 0.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)



class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_simeco', 'G_simeco', 'cycle_simeco', 'idt_simeco', 'D_us', 'G_us', 'cycle_us', 'idt_us']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_simeco', 'fake_us', 'rec_simeco']
        visual_names_B = ['real_us', 'fake_simeco', 'rec_us']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_simeco')
            visual_names_B.append('idt_us')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_simeco', 'G_us', 'D_simeco', 'D_us']
        else:  # during test time, only load Gs
            self.model_names = ['G_simeco', 'G_us']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_simeco = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_us = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_simeco = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_us = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_simeco_pool = ImagePool(opt.pool_size)
            self.fake_us_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSSIM = SSIM(window_size=11, size_average=True)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_simeco.parameters(), self.netG_us.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_simeco.parameters(), self.netD_us.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_simeco = input['A' if AtoB else 'B'].to(self.device)
        self.real_us = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_us = self.netG_simeco(self.real_simeco)
        self.rec_simeco = self.netG_us(self.fake_us)

        self.fake_simeco = self.netG_us(self.real_us)
        self.rec_us = self.netG_simeco(self.fake_simeco)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_us = self.fake_us_pool.query(self.fake_us)
        self.loss_D_simeco = self.backward_D_basic(self.netD_simeco, self.real_us, fake_us)

    def backward_D_B(self):
        fake_simeco = self.fake_simeco_pool.query(self.fake_simeco)
        self.loss_D_us = self.backward_D_basic(self.netD_us, self.real_simeco, fake_simeco)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_simeco = self.netG_simeco(self.real_us)
            self.loss_idt_simeco = self.criterionIdt(self.idt_simeco, self.real_us) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_us = self.netG_us(self.real_simeco)
            self.loss_idt_us = self.criterionIdt(self.idt_us, self.real_simeco) * lambda_A * lambda_idt
        else:
            self.loss_idt_simeco = 0
            self.loss_idt_us = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_simeco = self.criterionGAN(self.netD_simeco(self.fake_us), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_us = self.criterionGAN(self.netD_us(self.fake_simeco), True)
        # Forward cycle loss
        self.loss_cycle_simeco = self.criterionCycle(self.rec_simeco, self.real_simeco) * lambda_A
        # Backward cycle loss
        self.loss_cycle_us = self.criterionCycle(self.rec_us, self.real_us) * lambda_B
        # Forward cycle loss based on SSIM
        self.loss_cycle_ssim_simeco = self.criterionSSIM(self.rec_simeco, self.real_simeco) * lambda_A
        # Backward cycle loss based on SSIM
        self.loss_cycle_ssim_us = self.criterionSSIM(self.rec_us, self.real_us) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_simeco + self.loss_G_us + self.loss_cycle_simeco + self.loss_cycle_us + self.loss_idt_simeco + self.loss_idt_us + self.loss_cycle_ssim_simeco + self.loss_cycle_ssim_us
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_simeco, self.netD_us], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_simeco, self.netD_us], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
