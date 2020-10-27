import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_aus, lambda_rus, and lambda_identity for the following losses.
        aus (artificial us domain), rus (real us domain).
        Generators: G_aus: aus -> rus; G_rus: rus -> aus.
        Discriminators: D_aus: G_aus(aus) vs. rus; D_rus: G_rus(rus) vs. aus.
        Forward cycle loss:  lambda_aus * ||G_rus(G_aus(aus)) - aus|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_rus * ||G_aus(G_rus(rus)) - rus|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_aus(rus) - rus|| * lambda_rus + ||G_rus(aus) - aus|| * lambda_aus) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_aus', type=float, default=10.0, help='weight for cycle loss (aus -> rus -> aus)')
            parser.add_argument('--lambda_rus', type=float, default=10.0, help='weight for cycle loss (rus -> aus -> rus)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_aus', 'G_aus', 'cycle_aus', 'idt_aus', 'D_rus', 'G_rus', 'cycle_rus', 'idt_rus']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_aus = ['real_aus', 'fake_rus', 'rec_aus']
        visual_names_rus = ['real_rus', 'fake_aus', 'rec_rus']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_rus=G_aus(rus) ad idt_aus=G_aus(rus)
            visual_names_aus.append('idt_rus')
            visual_names_rus.append('idt_aus')

        self.visual_names = visual_names_aus + visual_names_rus  # combine visualizations for aus and rus
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_aus', 'G_rus', 'D_aus', 'D_rus']
        else:  # during test time, only load Gs
            self.model_names = ['G_aus', 'G_rus']

        # define networks (both Generators and discriminators)

        self.netG_aus = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_rus = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_aus = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_rus = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_aus_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_rus_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_aus.parameters(), self.netG_rus.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_aus.parameters(), self.netD_rus.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        austorus = self.opt.direction == 'austorus'
        self.real_aus = input['aus' if austorus else 'rus'].to(self.device)
        self.real_rus = input['rus' if austorus else 'aus'].to(self.device)
        self.image_paths = input['aus_paths' if austorus else 'rus_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_rus = self.netG_aus(self.real_aus)  # G_aus(aus)
        self.rec_aus = self.netG_rus(self.fake_rus)   # G_rus(G_aus(aus))
        self.fake_aus = self.netG_rus(self.real_rus)  # G_rus(rus)
        self.rec_rus = self.netG_aus(self.fake_aus)   # G_aus(G_rus(rus))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_aus(self):
        """Calculate GAN loss for discriminator D_aus"""
        fake_rus = self.fake_rus_pool.query(self.fake_rus)
        self.loss_D_aus = self.backward_D_basic(self.netD_aus, self.real_rus, fake_rus)

    def backward_D_rus(self):
        """Calculate GAN loss for discriminator D_rus"""
        fake_aus = self.fake_aus_pool.query(self.fake_aus)
        self.loss_D_rus = self.backward_D_basic(self.netD_rus, self.real_aus, fake_aus)

    def backward_G(self):
        """Calculate the loss for generators G_aus and G_rus"""
        lambda_idt = self.opt.lambda_identity
        lambda_aus = self.opt.lambda_aus
        lambda_rus = self.opt.lambda_rus
        # Identity loss
        if lambda_idt > 0:
            # G_aus should be identity if real_rus is fed: ||G_aus(rus) - rus||
            self.idt_aus = self.netG_aus(self.real_rus)
            self.loss_idt_aus = self.criterionIdt(self.idt_aus, self.real_rus) * lambda_rus * lambda_idt
            # G_rus should be identity if real_aus is fed: ||G_rus(aus) - aus||
            self.idt_rus = self.netG_rus(self.real_aus)
            self.loss_idt_rus = self.criterionIdt(self.idt_rus, self.real_aus) * lambda_aus * lambda_idt
        else:
            self.loss_idt_aus = 0
            self.loss_idt_rus = 0

        # GAN loss D_aus(G_aus(aus))
        self.loss_G_aus = self.criterionGAN(self.netD_aus(self.fake_rus), True)
        # GAN loss D_rus(G_rus(rus))
        self.loss_G_rus = self.criterionGAN(self.netD_rus(self.fake_aus), True)
        # Forward cycle loss || G_rus(G_aus(aus)) - aus||
        self.loss_cycle_aus = self.criterionCycle(self.rec_aus, self.real_aus) * lambda_aus
        # Backward cycle loss || G_aus(G_rus(rus)) - rus||
        self.loss_cycle_rus = self.criterionCycle(self.rec_rus, self.real_rus) * lambda_rus
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_aus + self.loss_G_rus + self.loss_cycle_aus + self.loss_cycle_rus + self.loss_idt_aus + self.loss_idt_rus
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_aus and G_rus
        self.set_requires_grad([self.netD_aus, self.netD_rus], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_aus and G_rus's gradients to zero
        self.backward_G()             # calculate gradients for G_aus and G_rus
        self.optimizer_G.step()       # update G_aus and G_rus's weights
        # D_aus and D_rus
        self.set_requires_grad([self.netD_aus, self.netD_rus], True)
        self.optimizer_D.zero_grad()   # set D_aus and D_rus's gradients to zero
        self.backward_D_aus()      # calculate gradients for D_aus
        self.backward_D_rus()      # calculate graidents for D_rus
        self.optimizer_D.step()  # update D_aus and D_rus's weights