import numbers
from re import T
from cv2 import norm
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import pywt
import numpy as np
from torch.nn.functional import interpolate
from kymatio.torch import Scattering2D  # for scattering
from torchvision.transforms.functional import rgb_to_grayscale
import pdb

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm _type (str) -- the name of the normalization layer: batch | instance | none --> default norm is " instance" 

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], is_A=True, input_size=512):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        is_a (bool) -- ellen made, for different model for G_A and G_B. True for G_A and False for G_B
        input_size (int) --ellen made, opt.load_sized 값 받아옴 -> used for scattering function 
 
    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blo
        cks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    # elif netG == 'resnet_6blocks':
    #     net = ResnetGenerator(
    #         input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    # elif netG == 'resnet_3blocks':
    #     net = ResnetGenerator(
    #         input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)


    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256' or netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 8, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)\
    # elif netG == 'unet_512': # ellen made
    #     net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    # elif netG == 'ellen_uresnet': 
    #     # uresent - base for 2_3 
    #     net = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                         use_dropout=use_dropout, num_downs=2, n_blocks=9)

    # elif netG == 'ellen_dwt_uresnet2_1':
    #     #dwgan + uresnet
    #     net = ellen_dwt_uresnet2_1(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=2, n_blocks=9)

    # elif netG == 'ellen_dwt_uresnet2_2':
    #     if is_A:
    #         net = ellen_dwt_uresnet2_2A(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                     use_dropout=use_dropout)
    #     elif not is_A:
    #         net = ellen_dwt_uresnet2_2B(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                     use_dropout=use_dropout)

    elif netG == 'ellen_dwt_uresnet2_3':  # 2_1 based - scattering branch instead of dwt branch
        net = ellen_dwt_uresnet2_3(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)

    elif netG == 'ellen_dwt_uresnet2_3_b1':  # 2_1 based - scattering branch instead of dwt branch
        net = ellen_dwt_uresnet2_3_b1(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)
    
    elif netG == 'ellen_dwt_uresnet2_3_b2':  # 2_1 based - scattering branch instead of dwt branch
        net = ellen_dwt_uresnet2_3_b2(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)

    elif netG == 'ellen_dwt_uresnet2_5':  # 2_3 based - sum
        net = ellen_dwt_uresnet2_5(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)
    elif netG == 'ellen_dwt_uresnet2_5_1':  # 2_5 based batch norm O
        net = ellen_dwt_uresnet2_5_1(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)

    elif netG == 'ellen_dwt_uresnet2_5_2':  # 2_5_1 based scattering attention 
        net = ellen_dwt_uresnet2_5_2(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)

    elif netG == 'ellen_dwt_uresnet2_6':  # 2_5_1 based with less ngf, more layer in unet, less layer in scattering branch
        net = ellen_dwt_uresnet2_6(input_nc, output_nc, 8, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=4, n_blocks=0, input_size=input_size)

    elif netG == 'ellen_dwt_uresnet1_7':  # scattering and uresnet in one branch
        net = ellen_dwt_uresnet1_7(use_dropout=use_dropout,  input_size=input_size)

    elif netG == 'ellen_dwt_uresnet1_8':  # scattering and uresnet in one branch
        net = ellen_dwt_uresnet1_8(use_dropout=use_dropout,  input_size=input_size)
    
    # elif netG == 'ellen_dwt_uresnet2_3_2':  # 2_3 based - Unet: Tconv->resize&conv
    #     net = ellen_dwt_uresnet2_3_2(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=2, n_blocks=0, input_size=input_size)
    
    # elif netG == 'ellen_scattering':  # 2_1 based - scattering branch instead of dwt branch
    #     net = ellen_scattering(input_size=input_size)

    # elif netG == 'ellen_dwt_uresnet2_4':  # 2_3 based - gray scale input for scattering branch
    #     net = ellen_dwt_uresnet2_4(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=3, n_blocks=9, input_size=input_size)

    # elif netG == 'ellen_dwt_uresnet1_1':
    #     #uresnet(with dwt)
    #     net = ellen_dwt_uresnet1_1(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=2, n_blocks=2)

    # elif netG == 'ellen_dwt_uresnet1_2':
    #     if is_A:
    #         net = ellen_dwt_uresnet1_2A(
    #             input_nc, output_nc, nf=16, norm_layer=norm_layer, use_dropout=use_dropout)
    #     elif not is_A:
    #         net = ellen_dwt_uresnet1_2B(
    #             input_nc, output_nc, nf=16, norm_layer=norm_layer, use_dropout=use_dropout)

    # elif netG == 'ellen_dwt_uresnet1_3':
    #     #uresent(with dwt)
    #     net = ellen_dwt_uresnet1_3(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=3)
    # elif netG == 'ellen_dwt_uresnet1_4':
    #     #uresent(with dwt)
    #     net = ellen_dwt_uresnet1_4(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=3)

    # elif netG == 'ellen_dwt_uresnet1_5':
    #     #uresent(with dwt)
    #     net = ellen_dwt_uresnet1_5(input_nc, output_nc, ngf, norm_layer=norm_layer,
    #                                use_dropout=use_dropout, num_downs=6)

    # elif netG == 'ellen_dwt_uresnet1_6':
    #     if is_A:
    #         net = ellen_dwt_uresnet1_6A(
    #             input_nc, output_nc, nf=16, norm_layer=norm_layer, use_dropout=use_dropout)
    #     elif not is_A:
    #         net = ellen_dwt_uresnet1_6B(
    #             input_nc, output_nc, nf=16, norm_layer=norm_layer, use_dropout=use_dropout)

    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        # either use real images, fake images, or a linear interpolation of two.
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement(
            ) // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(
                                            disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) -
                            constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7,
                           padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + \
            self.conv_block(x)  # add skip connections -- cat이 아니라 +라 채널수 안바뀜
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules -전에 define한 모듈
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()

        # 정의
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)  # outer -> inner
        # inplace를 하면, input으로 들어온 것 자체를 수정. memory usage가 좀 좋아짐. 하지만, input을 없앰
        downrelu = nn.LeakyReLU(0.2, True)
        # inner_nc:the # of filters in the inner conv layer
        downnorm = norm_layer(inner_nc)
        # uprelu = nn.LeakyReLU(0.2, True) # ellen 10.12 ReLU -> LeakyReLU로 변경
        uprelu = nn.ReLU(True)

        upnorm = norm_layer(outer_nc)

        #실질적 모델 구성 - becuase it has a word "model" in it & upconv가 keep changing
        if outermost:  # 제일 마지막 module
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 제일 처음 module
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # 그 중간 module
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock_2(nn.Module):
    """
    2022.12.07 Ellen Made
    
    
    Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules -전에 define한 모듈
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_2, self).__init__()

        # 정의
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)  # outer -> inner
        # inplace를 하면, input으로 들어온 것 자체를 수정. memory usage가 좀 좋아짐. 하지만, input을 없앰
        downrelu = nn.LeakyReLU(0.2, True)
        # inner_nc:the # of filters in the inner conv layer
        downnorm = norm_layer(inner_nc)
        # uprelu = nn.LeakyReLU(0.2, True) # ellen 10.12 ReLU -> LeakyReLU로 변경
        uprelu = nn.ReLU(True)

        upnorm = norm_layer(outer_nc)

        #실질적 모델 구성 - becuase it has a word "model" in it & upconv가 keep changing
        if outermost:  # 제일 마지막 module
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            upconv = transpose_resize_conv(inner_nc*2, outer_nc) # 2022.12.07
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 제일 처음 module
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upconv = transpose_resize_conv(inner_nc, outer_nc)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # 그 중간 module
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upconv = transpose_resize_conv(inner_nc*2, outer_nc)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock_Ellen(nn.Module):
    """
    ellen modified
    1. innermost 에서 submodule을 사용하고 있지 않았음(resnet block를 사용 X) --> 사용하도록 바꿈
    2. Down & up Activation 모두 LeakyReLU사용 (마지막에 Tanh사용하므로) up 할때 ReLU -> LeakyReLU 변경

    2022.12.02 edit
    3. Tconv -> resize & Conv
    4. pad -> reflection pad
    
    """

    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules -전에 define한 모듈
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_Ellen, self).__init__()

        # 정의
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
        #                      stride=2, padding=1, bias=use_bias)  # outer -> inner #original
        downconv = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, bias=use_bias)]) # outer -> inner # 2022.12.02
                             
        # inplace를 하면, input으로 들어온 것 자체를 수정. memory usage가 좀 좋아짐. 하지만, input을 없앰
        downrelu = nn.LeakyReLU(0.2, True)
        # inner_nc:the # of filters in the inner conv layer
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True) # ellen 10.12 ReLU -> LeakyReLU로 변경

        upnorm = norm_layer(outer_nc)

        #실질적 모델 구성 - becuase it has a word "model" in it & upconv가 keep changing
        if outermost:  # 제일 마지막 module
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1) # original
            upconv = transpose_resize_conv(inner_nc*2, outer_nc) # 2022.12.02
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 제일 처음 module
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias) # original
            upconv = transpose_resize_conv(inner_nc, outer_nc) # 2022.12.02
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down +[submodule] +up
        else:  # 그 중간 module
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias) # original
            upconv = transpose_resize_conv(inner_nc*2, outer_nc) # 2022.12.02
            
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class transpose_resize_conv(nn.Module):
    # 2022.12.02
    def __init__(self,in_c, out_c):
        super(transpose_resize_conv, self).__init__()
        model = [nn.ReflectionPad2d(1), nn.Conv2d(in_c, out_c, kernel_size=3, stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        sizee = x.shape[-1]*2
        out = interpolate(x, size=(sizee,sizee), mode='bilinear')
        return self.model(out)



# ---------------------------------------------------------------------------------


class ellen_uresnet(nn.Module):
    """
    made by ellen 
    Unet + Resnet 
    encoder & decoder 부분 : Unet 으로 
    Transformer 부분: Resnet으로 

    왜? Unet이 세부 디테일을 잘 살려 줄 것 같은데, 너무 압축하다보니 오히려 안좋은 결과를 내는 것 같아서
    (unet 256일때는 nums of down이 8, 512 일때는 9로 했는데, 후자의 결과가 훨씬 안좋았음.)
    그래서 unet부분은 nums of down을 적게하고, (resnet이 할때와 비슷하게)

    Transformer part를 늘리기 like resnet  

    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=9)  
    
    [수정]
    2022.10.14: classname typo 수정 (uresent -> uresnet)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, padding_type='reflect',batch_norm=False):
        super(ellen_uresnet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # e1 크기는 일정, chanel 수 변환3-> 64
        model = [nn.ReflectionPad2d(3), nn.Conv2d(
            input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        print("========================================================================")
        print("num downs = ", num_downs, " n_blocks=", n_blocks)
        print("----e1")
        # resnet 구조(unet안에 있는)
        # default: 64*8 -> 64*8 반복
        multi = 2**(num_downs)  # 내려가는 만큼 채널수 : 64*2^n
        
        unetblock = UnetSkipConnectionBlock(ngf*multi, ngf*multi, input_nc=None,
                                            norm_layer=norm_layer, innermost=True,use_dropout=use_dropout)  # 젤 안쪽에서 resnet과 닿아 있는 것 > 64*8 64*8
        unetblock = UnetSkipConnectionBlock(int(
            ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer,use_dropout=use_dropout)   # > 64*4 64*8
        print(">> in, out: ", ngf*(multi/2), ", ", ngf*multi)
        for i in range(num_downs-2):
            multi = int(multi/2)
            unetblock = UnetSkipConnectionBlock(int(
                ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer,use_dropout=use_dropout) 
            print("-----unet", i, " 번째")
            print(">> in, out: ", ngf*(multi/2), ", ", ngf*multi)
        multi = int(multi/2)
        unetblock = UnetSkipConnectionBlock(int(
            ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer, outermost=True,use_dropout=use_dropout) 

        model = model + [unetblock]
        print("----- e1+unet+resent+uent+")
        
        if not batch_norm:
            # d1 크기는 일정, chanel 수 변환64->3
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                    output_nc, kernel_size=7, padding=0), nn.Tanh()]
        else: 
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                    output_nc, kernel_size=7, padding=0),norm_layer(output_nc), nn.Tanh()]
        
        print("----- e1+unet+resent+uent+d1")

        self.model = nn.Sequential(*model)
        print(model)

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])

        return self.model(input)



class ellen_uresnet_new(nn.Module):
    """
    for 2_4
    2022.10.13
    1. 기존 모델 오류 수정 (기존에는 num_down과 실제 down이 맞지 않았음)
    2. scattering branch와 유사하게
    3. ngf: 64 -> 16 (channel수 너무 많음)
    4. UnetSkipConnectionBlock_Ellen 로 바꿈(Resnetblock사용, up: LeakyReLU)

    5. num_downs >1 1이면 channel수 안맞음
    추가 2022.12.02
    6. Transpose->resize & Conv
    7. 그냥 pad->reflectionpad
    """

    def __init__(self, input_nc, output_nc, ngf=16, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, padding_type='reflect'):
        super(ellen_uresnet_new, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # torch.autograd.set_detect_anomaly(True)

        # Encoder 1: first down
        model = [nn.ReflectionPad2d(1), nn.Conv2d(
            input_nc, ngf, kernel_size=4, stride=2, padding=0, bias=use_bias), norm_layer(ngf), nn.LeakyReLU(0.2, False)] #[10/16] 여기에 relu를 False로 relu inplace 문제 해결됨. uresnet에서도 outter layer에는 down ReLU없음 Tanh()도 안됨

        # 내려가는 만큼 채널수 : 16 * multi (1down: 16*(2**0) , 2down : 16*(2**1), 3down : 16*(2**2) )
        multi = 2**(num_downs-1)

        # Inner most part: Resnet blocks
        resnet_inUnet = []
        for i in range(n_blocks):
            resnet_inUnet += [ResnetBlock(ngf*multi, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            print("-----resnet", i, " 번째")
        resnet_inUnets = nn.Sequential(*resnet_inUnet)  # submodule

        # Encoder & Decoder: Unet block
        unetblock = UnetSkipConnectionBlock_Ellen(int(ngf*(multi/2)), ngf*multi, input_nc=None, submodule=resnet_inUnets,
                                            norm_layer=norm_layer, innermost=True)  # Incubates resnet blocks(innermost part)- same input and output size as resnet
        for i in range(num_downs-2):  # -2: innermost & outermost - total 2
            multi = int(multi/2)
            unetblock = UnetSkipConnectionBlock_Ellen(int(
                ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer)
            print("-----unet", i, " 번째")
        multi = int(multi/2)

        model = model + [unetblock]

        # Decoder 1: last up
        # model += [nn.LeakyReLU(0.2, False), nn.ConvTranspose2d(ngf*2,
        #                              ngf, kernel_size=4, stride=2, padding=1), nn.Tanh()] #original
        model += [nn.LeakyReLU(0.2, False), transpose_resize_conv(ngf*2, ngf), nn.Tanh()] # 2022.12.02

        # Decoder 2: same size conv - channel reduction
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                   output_nc, kernel_size=7, padding=0), nn.Tanh()] 
      

        self.model = nn.Sequential(*model)
        print(model)

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        return self.model(input)


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False, resize=False):
    block = nn.Sequential()
    if relu:# up
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else: # down
        block.add_module('%s_leakyrelu' %
                         name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:  # when going down
        block.add_module('%s_conv' % name, nn.Conv2d(
            in_c, out_c, 4, 2, 1, bias=False)) # SIZE 1/2 - K, S, P
    elif transposed and not resize:  # original going up
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(
            in_c, out_c, 4, 2, 1, bias=False))
    # to reduce checkerboard artifact -> for model 1_6 (same size conv)
    elif transposed and resize:
        block.add_module('%s_conv' % name, nn.Conv2d(
            in_c, out_c, 3, 1, 1, bias=False))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block




class scatter_transform(nn.Module):
    def __init__(self, in_channels, out_channels, size, level, kind=0,dropout=False,scattering_attention=False):
        super(scatter_transform, self).__init__()
        self.kind = kind # [0: cat & S0, S1], [1: sum & only S1]
        self.dropout=dropout
        self.scattering_attention=scattering_attention
        if dropout:
            self.dropout_layer=nn.Dropout(0.5)
        self.output_size = int(size/(2**level))
        input_size = int(size/(2**(level-1)))
        J = 1
        self.Scattering = Scattering2D(
            J, (input_size, input_size))  # backend='torch_skcuda
        if self.kind ==0:
            self.conv1x1 = nn.Conv2d(
                in_channels*9, out_channels, kernel_size=1, padding=0)
        elif self.kind ==1:
            self.conv1x1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)
        self.leakyrelu=nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        scatter_output = self.Scattering.scattering(x)
        if self.kind ==0:
            scatter_output = scatter_output.view(scatter_output.size(
                0), -1, self.output_size, self.output_size)
        elif self.kind ==1:
            scatter_output=torch.sum(scatter_output[:,:,1:,:,:],2) #batch, channel, 9, H, W 
        
        if self.scattering_attention:
            scatter_output=scatter_output*100.
        scatter_output = self.conv1x1(scatter_output)
        scatter_output=self.leakyrelu(scatter_output) #added with kind
        if self.dropout:
            scatter_output=self.dropout_layer(scatter_output)
        return scatter_output


class scattering_Unet(nn.Module):
    def __init__(self, input_size, output_nc, nf=16,kind=0,dropout=False,batch_norm=False,scattering_attention=False):
        super(scattering_Unet, self).__init__()
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(3, nf-1, 4, 2, 1, bias=False)) # SIZE 1/2 동일
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8-8, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8-16, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False,
                           bn=False, relu=False, dropout=dropout)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 8, nf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16, nf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16, nf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4, nf, name, transposed=True,
                            bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)

        self.layer1 = layer1
        self.scattering_down_1 = scatter_transform(3, 1, input_size, 1, kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer2 = layer2
        self.scattering_down_2 = scatter_transform(16, 2, input_size, 2,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer3 = layer3
        self.scattering_down_3 = scatter_transform(32, 4, input_size, 3,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer4 = layer4
        self.scattering_down_4 = scatter_transform(64, 8, input_size, 4,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer5 = layer5
        self.scattering_down_5 = scatter_transform(128, 16, input_size, 5,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.batch_norm = batch_norm
        if batch_norm:
            self.l1norm=nn.BatchNorm2d(nf)
            self.l2norm=nn.BatchNorm2d(nf*2)
            self.l3norm=nn.BatchNorm2d(nf*4)
            self.l4norm=nn.BatchNorm2d(nf*8)
            self.l5norm=nn.BatchNorm2d(nf*8)
            self.l6norm=nn.BatchNorm2d(nf*8)

            self.dl6norm=nn.BatchNorm2d(nf*8)
            self.dl5norm=nn.BatchNorm2d(nf*8)
            self.dl4norm=nn.BatchNorm2d(nf*4)
            self.dl3norm=nn.BatchNorm2d(nf*2)
            self.dl2norm=nn.BatchNorm2d(nf)
            self.dl1norm=nn.BatchNorm2d(nf*2)
            self.tailnorm=nn.BatchNorm2d(output_nc)

        # self.tail_conv1 = nn.Conv2d(32, output_nc, 3, padding=1, bias=True) # 2_3_ori
        self.tail_conv1 = nn.Sequential(nn.Conv2d(32, output_nc, 3, padding=1, bias=True), nn.Tanh()) # made - 2022.11.21

    def forward(self, x):
        if not self.batch_norm:
            conv_out1 = self.layer1(x)
            scattering1 = self.scattering_down_1(x)
            out1 = torch.cat([conv_out1, scattering1], 1)
            conv_out2 = self.layer2(out1)
            scattering2 = self.scattering_down_2(out1)
            out2 = torch.cat([conv_out2, scattering2], 1)
            conv_out3 = self.layer3(out2)
            scattering3 = self.scattering_down_3(out2)
            out3 = torch.cat([conv_out3, scattering3], 1)
            conv_out4 = self.layer4(out3)
            scattering4 = self.scattering_down_4(out3)
            out4 = torch.cat([conv_out4, scattering4], 1)
            conv_out5 = self.layer5(out4)
            scattering5 = self.scattering_down_5(out4)
            out5 = torch.cat([conv_out5, scattering5], 1)
            out6 = self.layer6(out5)

            sizee = out6.shape[-1]*2

            tin6 = interpolate(out6, size=(sizee, sizee), mode='bilinear')
            dout6 = self.dlayer6(tin6)

            Tout6_out5 = torch.cat([dout6, out5], 1)
            sizee = sizee*2
            tin5 = interpolate(Tout6_out5, size=(sizee, sizee), mode='bilinear')
            Tout5 = self.dlayer5(tin5)

            Tout5_out4 = torch.cat([Tout5, out4], 1)
            sizee = sizee*2
            tin4 = interpolate(Tout5_out4, size=(sizee, sizee), mode='bilinear')
            Tout4 = self.dlayer4(tin4)

            Tout4_out3 = torch.cat([Tout4, out3], 1)
            sizee = sizee*2
            tin3 = interpolate(Tout4_out3, size=(sizee, sizee), mode='bilinear')
            Tout3 = self.dlayer3(tin3)

            Tout3_out2 = torch.cat([Tout3, out2], 1)
            sizee = sizee*2
            tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
            Tout2 = self.dlayer2(tin2)

            Tout2_out1 = torch.cat([Tout2, out1], 1)
            sizee = sizee*2
            tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
            Tout1 = self.dlayer1(tin1)

            tail1 = self.tail_conv1(Tout1)
            return tail1

        else: 
            conv_out1 = self.layer1(x)
            scattering1 = self.scattering_down_1(x)
            out1 = torch.cat([conv_out1, scattering1], 1)
            out1=self.l1norm(out1)
            conv_out2 = self.layer2(out1)
            scattering2 = self.scattering_down_2(out1)
            out2 = torch.cat([conv_out2, scattering2], 1)
            out2=self.l2norm(out2)
            conv_out3 = self.layer3(out2)
            scattering3 = self.scattering_down_3(out2)
            out3 = torch.cat([conv_out3, scattering3], 1)
            out3=self.l3norm(out3)
            conv_out4 = self.layer4(out3)
            scattering4 = self.scattering_down_4(out3)
            out4 = torch.cat([conv_out4, scattering4], 1)
            out4=self.l4norm(out4)
            conv_out5 = self.layer5(out4)
            scattering5 = self.scattering_down_5(out4)
            out5 = torch.cat([conv_out5, scattering5], 1)
            out5=self.l5norm(out5)
            out6 = self.layer6(out5)
            out6=self.l6norm(out6)

            sizee = out6.shape[-1]*2

            tin6 = interpolate(out6, size=(sizee, sizee), mode='bilinear')
            dout6 = self.dlayer6(tin6)
            dout6=self.dl6norm(dout6)
            

            Tout6_out5 = torch.cat([dout6, out5], 1)
            sizee = sizee*2
            tin5 = interpolate(Tout6_out5, size=(sizee, sizee), mode='bilinear')
            Tout5 = self.dlayer5(tin5)
            Tout5=self.dl5norm(Tout5)

            Tout5_out4 = torch.cat([Tout5, out4], 1)
            sizee = sizee*2
            tin4 = interpolate(Tout5_out4, size=(sizee, sizee), mode='bilinear')
            Tout4 = self.dlayer4(tin4)
            Tout4=self.dl4norm(Tout4)

            Tout4_out3 = torch.cat([Tout4, out3], 1)
            sizee = sizee*2
            tin3 = interpolate(Tout4_out3, size=(sizee, sizee), mode='bilinear')
            Tout3 = self.dlayer3(tin3)
            Tout3=self.dl3norm(Tout3)

            Tout3_out2 = torch.cat([Tout3, out2], 1)
            sizee = sizee*2
            tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
            Tout2 = self.dlayer2(tin2)
            Tout2=self.dl2norm(Tout2)

            Tout2_out1 = torch.cat([Tout2, out1], 1)
            sizee = sizee*2
            tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
            Tout1 = self.dlayer1(tin1)
            Tout1=self.dl1norm(Tout1)

            tail1 = self.tail_conv1(Tout1)
            tail1=self.tailnorm(tail1)
            return tail1


class scattering_Unet2_6(nn.Module):
    # scattering unet for 2_6 : layer 6-> 4
    def __init__(self, input_size, output_nc, nf=16,kind=0,dropout=False,batch_norm=False,scattering_attention=False):
        super(scattering_Unet2_6, self).__init__()
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(3, nf-1, 4, 2, 1, bias=False)) # SIZE 1/2 동일
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
       
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 8, nf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4, nf, name, transposed=True,
                            bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)

        self.layer1 = layer1
        self.scattering_down_1 = scatter_transform(3, 1, input_size, 1, kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer2 = layer2
        self.scattering_down_2 = scatter_transform(16, 2, input_size, 2,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer3 = layer3
        self.scattering_down_3 = scatter_transform(32, 4, input_size, 3,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer4 = layer4
      
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.batch_norm = batch_norm
        if batch_norm:
            self.l1norm=nn.BatchNorm2d(nf)
            self.l2norm=nn.BatchNorm2d(nf*2)
            self.l3norm=nn.BatchNorm2d(nf*4)
            self.l4norm=nn.BatchNorm2d(nf*8)

            self.dl4norm=nn.BatchNorm2d(nf*4)
            self.dl3norm=nn.BatchNorm2d(nf*2)
            self.dl2norm=nn.BatchNorm2d(nf)
            self.dl1norm=nn.BatchNorm2d(nf*2)
            self.tailnorm=nn.BatchNorm2d(output_nc)

        # self.tail_conv1 = nn.Conv2d(32, output_nc, 3, padding=1, bias=True) # 2_3_ori
        self.tail_conv1 = nn.Sequential(nn.Conv2d(32, output_nc, 3, padding=1, bias=True), nn.Tanh()) # made - 2022.11.21

    def forward(self, x):
        if not self.batch_norm:
            conv_out1 = self.layer1(x)
            scattering1 = self.scattering_down_1(x)
            out1 = torch.cat([conv_out1, scattering1], 1)
            conv_out2 = self.layer2(out1)
            scattering2 = self.scattering_down_2(out1)
            out2 = torch.cat([conv_out2, scattering2], 1)
            conv_out3 = self.layer3(out2)
            scattering3 = self.scattering_down_3(out2)
            out3 = torch.cat([conv_out3, scattering3], 1)
            out4 = self.layer4(out3)

            sizee =out4.shape[-1]*2
            tin4= interpolate(out4, size=(sizee, sizee), mode='bilinear')
            dout4 = self.dlayer4(out4)

            Tout4_out3=torch.cat([dout4, out3],1)
            sizee = sizee*2
            tin3 = interpolate(Tout4_out3, size=(sizee, sizee), mode='bilinear')
            Tout3 = self.dlayer3(tin3)

            Tout3_out2 = torch.cat([Tout3, out2], 1)
            sizee = sizee*2
            tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
            Tout2 = self.dlayer2(tin2)

            Tout2_out1 = torch.cat([Tout2, out1], 1)
            sizee = sizee*2
            tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
            Tout1 = self.dlayer1(tin1)

            tail1 = self.tail_conv1(Tout1)
            return tail1

        else: 
            conv_out1 = self.layer1(x)
            scattering1 = self.scattering_down_1(x)
            out1 = torch.cat([conv_out1, scattering1], 1)
            out1=self.l1norm(out1)
            conv_out2 = self.layer2(out1)
            scattering2 = self.scattering_down_2(out1)
            out2 = torch.cat([conv_out2, scattering2], 1)
            out2=self.l2norm(out2)
            conv_out3 = self.layer3(out2)
            scattering3 = self.scattering_down_3(out2)
            out3 = torch.cat([conv_out3, scattering3], 1)
            out3=self.l3norm(out3)
            out4 = self.layer4(out3)
            out4=self.l4norm(out4)

            sizee =out4.shape[-1]*2
            tin4= interpolate(out4, size=(sizee, sizee), mode='bilinear')
            dout4 = self.dlayer4(tin4)
            dout4=self.dl4norm(dout4)

            Tout4_out3=torch.cat([dout4, out3],1)
            sizee = sizee*2
            tin3 = interpolate(Tout4_out3, size=(sizee, sizee), mode='bilinear')
            Tout3 = self.dlayer3(tin3)
            Tout3=self.dl3norm(Tout3)

            Tout3_out2 = torch.cat([Tout3, out2], 1)
            sizee = sizee*2
            tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
            Tout2 = self.dlayer2(tin2)
            Tout2=self.dl2norm(Tout2)

            Tout2_out1 = torch.cat([Tout2, out1], 1)
            sizee = sizee*2
            tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
            Tout1 = self.dlayer1(tin1)
            Tout1=self.dl1norm(Tout1)

            tail1 = self.tail_conv1(Tout1)
            tail1=self.tailnorm(tail1)
            
            return tail1

class scattering_Uresnet1_7(nn.Module):
    # scattering unet for 2_6 : layer 6-> 4
    def __init__(self, input_size, output_nc, nf=32,kind=0,dropout=True,batch_norm=False,scattering_attention=False, resnet_nblocks=3):
        super(scattering_Uresnet1_7, self).__init__()

        layer_idx = 1
        name = 'layer%d' % layer_idx
        scattering_ch1=nf//8
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(3, nf-scattering_ch1, 4, 2, 1, bias=False)) # SIZE 1/2 동일

        layer_idx += 1
        name = 'layer%d' % layer_idx
        scattering_ch2 =scattering_ch1*2
        layer2 = blockUNet(nf, nf*2-scattering_ch2, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        scattering_ch3=scattering_ch2*2
        layer3 = blockUNet(nf*2, nf*4-scattering_ch3, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)

        resnet_inUnet = []
        for i in range(resnet_nblocks):
            resnet_inUnet += [ResnetBlock(nf*8, padding_type='reflect',
                                          norm_layer=nn.BatchNorm2d, use_dropout=dropout, use_bias=True)]
            print("-----resnet", i, " 번째")
        self.resnet_inUnets = nn.Sequential(*resnet_inUnet)  # submodule
       
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 8, nf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 4 , nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 2 + scattering_ch2, nf, name, transposed=True,
                            bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf  +scattering_ch1, nf , name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)

        self.layer1 = layer1
        self.scattering_down_1 = scatter_transform(3, scattering_ch1, input_size, 1, kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer2 = layer2
        self.scattering_down_2 = scatter_transform(nf, scattering_ch2, input_size, 2,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer3 = layer3
        self.scattering_down_3 = scatter_transform(nf*2, scattering_ch3, input_size, 3,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer4 = layer4
      
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.batch_norm = batch_norm
        if batch_norm:
            self.c1norm = nn.BatchNorm2d(nf-scattering_ch1)
            self.s1norm = nn.BatchNorm2d(scattering_ch1)
            self.c2norm=nn.BatchNorm2d(nf*2-scattering_ch2)
            self.s2norm = nn.BatchNorm2d(scattering_ch2)
            self.l3norm=nn.BatchNorm2d(nf*4)
            self.l4norm=nn.BatchNorm2d(nf*8)

            self.dl4norm=nn.BatchNorm2d(nf*4)
            self.dl3norm=nn.BatchNorm2d(nf*2)
            self.dl2norm=nn.BatchNorm2d(nf)
            self.dl1norm=nn.BatchNorm2d(nf)
            # self.tailnorm=nn.BatchNorm2d(output_nc)

        # self.tail_conv1 = nn.Conv2d(32, output_nc, 3, padding=1, bias=True) # 2_3_ori
        self.tail_conv1 = nn.Sequential(nn.Conv2d(nf, output_nc, 3, padding=1, bias=True),nn.BatchNorm2d(output_nc), nn.Tanh()) # made - should place tnah last 

        
        

    def forward(self, x):
        conv_out1 = self.layer1(x)
        conv_out1=self.c1norm(conv_out1)
        scattering1 = self.scattering_down_1(x)
        scattering1 =self.s1norm(scattering1)
        out1 = torch.cat([conv_out1, scattering1], 1)

        conv_out2 = self.layer2(out1)
        conv_out2=self.c2norm(conv_out2)
        scattering2 = self.scattering_down_2(out1)
        scattering2 = self.s2norm(scattering2)
        out2 = torch.cat([conv_out2, scattering2], 1)

        conv_out3 = self.layer3(out2)
        scattering3 = self.scattering_down_3(out2)
        out3 = torch.cat([conv_out3, scattering3], 1)
        out3=self.l3norm(out3)
        out4 = self.layer4(out3)
        out4=self.l4norm(out4)

        out4 =self.resnet_inUnets(out4)

        sizee =out4.shape[-1]*2
        tin4= interpolate(out4, size=(sizee, sizee), mode='bilinear')
        dout4 = self.dlayer4(tin4)
        dout4=self.dl4norm(dout4)

        sizee = sizee*2
        tin3 = interpolate(dout4, size=(sizee, sizee), mode='bilinear')
        Tout3 = self.dlayer3(tin3)
        Tout3=self.dl3norm(Tout3)

        Tout3_out2 = torch.cat([Tout3, scattering2], 1)
        sizee = sizee*2
        tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
        Tout2 = self.dlayer2(tin2)
        Tout2=self.dl2norm(Tout2)

        Tout2_out1 = torch.cat([Tout2, scattering1], 1)
        sizee = sizee*2
        tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
        Tout1 = self.dlayer1(tin1)
        Tout1=self.dl1norm(Tout1)

        tail1 = self.tail_conv1(Tout1)
        # tail1=self.tailnorm(tail1)
        
        return tail1



class scattering_Uresnet1_8(nn.Module):
    # scattering unet for 2_6 : layer 6-> 4
    # skip connection like origina scattering unet (scattering+conv) and at every layer
    # resnet block just two 
    def __init__(self, input_size, output_nc, nf=32,kind=0,dropout=True,batch_norm=False,scattering_attention=False, resnet_nblocks=2):
        super(scattering_Uresnet1_8, self).__init__()

        layer_idx = 1
        name = 'layer%d' % layer_idx
        scattering_ch1=nf//8
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(3, nf-scattering_ch1, 4, 2, 1, bias=False)) # SIZE 1/2 동일

        layer_idx += 1
        name = 'layer%d' % layer_idx
        scattering_ch2 =scattering_ch1*2
        layer2 = blockUNet(nf, nf*2-scattering_ch2, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        scattering_ch3=scattering_ch2*2
        layer3 = blockUNet(nf*2, nf*4-scattering_ch3, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False,
                           bn=True, relu=False, dropout=dropout)

        resnet_inUnet = []
        for i in range(resnet_nblocks):
            resnet_inUnet += [ResnetBlock(nf*8, padding_type='reflect',
                                          norm_layer=nn.BatchNorm2d, use_dropout=dropout, use_bias=True)]
            print("-----resnet", i, " 번째")
        self.resnet_inUnets = nn.Sequential(*resnet_inUnet)  # submodule
       
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 8, nf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8 , nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4, nf, name, transposed=True,
                            bn=True, relu=True, dropout=dropout, resize=True)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf *2, nf , name,
                            transposed=True, bn=True, relu=True, dropout=dropout, resize=True)

        self.layer1 = layer1
        self.scattering_down_1 = scatter_transform(3, scattering_ch1, input_size, 1, kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer2 = layer2
        self.scattering_down_2 = scatter_transform(nf, scattering_ch2, input_size, 2,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer3 = layer3
        self.scattering_down_3 = scatter_transform(nf*2, scattering_ch3, input_size, 3,kind,dropout=dropout,scattering_attention=scattering_attention)
        self.layer4 = layer4
      
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.batch_norm = batch_norm
        if batch_norm:
            self.c1norm = nn.BatchNorm2d(nf)
            self.c2norm=nn.BatchNorm2d(nf*2)
            self.l3norm=nn.BatchNorm2d(nf*4)
            self.l4norm=nn.BatchNorm2d(nf*8)

            self.dl4norm=nn.BatchNorm2d(nf*4)
            self.dl3norm=nn.BatchNorm2d(nf*2)
            self.dl2norm=nn.BatchNorm2d(nf)
            self.dl1norm=nn.BatchNorm2d(nf)
            # self.tailnorm=nn.BatchNorm2d(output_nc)

        # self.tail_conv1 = nn.Conv2d(32, output_nc, 3, padding=1, bias=True) # 2_3_ori
        self.tail_conv1 = nn.Sequential(nn.Conv2d(nf, output_nc, 3, padding=1, bias=True),nn.BatchNorm2d(output_nc), nn.Tanh()) # made - should place tnah last 

        
        

    def forward(self, x):
        conv_out1 = self.layer1(x)
        scattering1 = self.scattering_down_1(x)
        out1 = torch.cat([conv_out1, scattering1], 1)
        out1 = self.c1norm(out1)

        conv_out2 = self.layer2(out1)
        scattering2 = self.scattering_down_2(out1)
        out2 = torch.cat([conv_out2, scattering2], 1)
        out2 = self.c2norm(out2)

        conv_out3 = self.layer3(out2)
        scattering3 = self.scattering_down_3(out2)
        out3 = torch.cat([conv_out3, scattering3], 1)
        out3=self.l3norm(out3)
        out4 = self.layer4(out3)
        out4=self.l4norm(out4)

        out4 =self.resnet_inUnets(out4)

        sizee =out4.shape[-1]*2
        tin4= interpolate(out4, size=(sizee, sizee), mode='bilinear')
        dout4 = self.dlayer4(tin4)
        dout4=self.dl4norm(dout4)

        Tout4_out3 =torch.cat([dout4,out3],1)
        sizee = sizee*2
        tin3 = interpolate(Tout4_out3, size=(sizee, sizee), mode='bilinear')
        Tout3 = self.dlayer3(tin3)
        Tout3=self.dl3norm(Tout3)

        Tout3_out2 = torch.cat([Tout3, out2], 1)
        sizee = sizee*2
        tin2 = interpolate(Tout3_out2, size=(sizee, sizee), mode='bilinear')
        Tout2 = self.dlayer2(tin2)
        Tout2=self.dl2norm(Tout2)

        Tout2_out1 = torch.cat([Tout2, out1], 1)
        sizee = sizee*2
        tin1 = interpolate(Tout2_out1, size=(sizee, sizee), mode='bilinear')
        Tout1 = self.dlayer1(tin1)
        Tout1=self.dl1norm(Tout1)

        tail1 = self.tail_conv1(Tout1)
        # tail1=self.tailnorm(tail1)
        
        return tail1



class ellen_dwt_uresnet2_3(nn.Module):
    """
    made by ellen _2022.10.11 
    > model 2_1 based
        dwt barnch -> scattering branch
        Tconv -> resize & conv
    > edit 23.02.15 
        can use drop out in scattering_unet
        can use batch norm in scattering_unet & ellen_uresnet tail (unet block은 원래 썼음)
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_3, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks,batch_norm=False)
        self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16,kind=0,dropout=use_dropout,batch_norm=False)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(x)
        # return result_uresnet


class ellen_dwt_uresnet2_3_b1(nn.Module):
    """
    made by ellen _2022.12.09 
    > model2_3 - only unet
    > edit 23.02.15 
        can use drop out in scattering_unet
        can use batch norm in scattering_unet & ellen_uresnet tail (unet block은 원래 썼음)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_3_b1, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks)
        # self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(3, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        # result_scattering = self.scattering_model(input)
        # x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(result_uresnet)
        # return result_uresnet


class ellen_dwt_uresnet2_3_b2(nn.Module):
    """
    made by ellen _2022.12.09 
    > model2_3 - only scattering
    > edit 23.02.15 
        can use drop out in scattering_unet
        can use batch norm in scattering_unet & ellen_uresnet tail (unet block은 원래 썼음)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_3_b2, self).__init__()
        # self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
        #                              use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks)
        self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16,kind=0,dropout=use_dropout,batch_norm=False)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(3, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        # result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        # x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(result_scattering)
        # return result_uresnet


class ellen_dwt_uresnet2_5(nn.Module):
    """
    made by ellen _2022.02.15 
    > model 2_3 based
        * scattering branch > scattering_Unet > scatter_transform "kind" 추가 ( 0:cat사용 1:sum )
        * kind =1
        1) cat -> sum 
        2) scattering coefficient S0 사용 X
        3) 1X1 conv 이후 leakyRelu 추가(원래 없었음)
    
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_5, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks,batch_norm=False)
        self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16,kind=1,dropout=use_dropout,batch_norm=False)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(x)
        # return result_uresnet

class ellen_dwt_uresnet2_5_1(nn.Module):
    """
    made by ellen _2022.02.15 
    > model 2_5 based
        1) batch_norm =True
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_5_1, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks,batch_norm=True)
        self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16,kind=1,dropout=use_dropout,batch_norm=True)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(x)
        # return result_uresnet


class ellen_dwt_uresnet2_6(nn.Module):
    """
    made by ellen _2022.02.18
    > model 2_5_1 based
        1) uresnet layer 수 증가 3-> 5 (checker boad artifact없애보기 위해서) - down은 2-> 4
        2) scattering layer 수 감소 6-> 4 (연산 줄이기 위해서) down 은 5-> 3 
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=8, norm_layer=nn.BatchNorm2d, use_dropout=True, num_downs=5, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_6, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=5, n_blocks=n_blocks,batch_norm=True)
        self.scattering_model = scattering_Unet2_6(input_size, output_nc=3, nf=16,kind=1,dropout=use_dropout,batch_norm=True)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(x)
        # return result_uresnet

class ellen_dwt_uresnet1_7(nn.Module):
    """
    made by ellen _2022.02.18
    > scattering Unet2_6 based
        1) just using one branch with scattering 
        2) unet architecture
        3) last layer -> resnet block added (uresnet)
        4) more scattering channel
    """

    def __init__(self, use_dropout=True, input_size=512):
        super(ellen_dwt_uresnet1_7, self).__init__()

        self.scattering_model = scattering_Uresnet1_7(input_size, output_nc=3, nf=32,kind=1,dropout=use_dropout,batch_norm=True,scattering_attention=False)

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        return self.scattering_model(input)
        # return result_uresnet

class ellen_dwt_uresnet1_8(nn.Module):
    """
    made by ellen _2022.02.18
    > scattering Unet1_7 based
        2) unet architecture
        3) last layer -> resnet block added (uresnet)
        4) more scattering channel
        5) skip connection with both conv and scattering
    """

    def __init__(self, use_dropout=True, input_size=512):
        super(ellen_dwt_uresnet1_8, self).__init__()

        self.scattering_model = scattering_Uresnet1_8(input_size, output_nc=3, nf=16,kind=1,dropout=use_dropout,batch_norm=True,scattering_attention=False)

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        return self.scattering_model(input)
        # return result_uresnet


class ellen_dwt_uresnet2_5_2(nn.Module):
    """
    made by ellen _2022.02.15 
    > model 2_5_1 based
        1) scattering attention = True --> scattering 결과에 * 100 
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, input_size=512):
        super(ellen_dwt_uresnet2_5_2, self).__init__()
        self.uresnet = ellen_uresnet(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks,batch_norm=True)
        self.scattering_model = scattering_Unet(input_size, output_nc=3, nf=16,kind=1,dropout=use_dropout,batch_norm=True, scattering_attention=True)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_scattering = self.scattering_model(input)
        x = torch.cat([result_scattering, result_uresnet], 1)

        return self.fusion(x)
        # return result_uresnet

#------------------------------------------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------