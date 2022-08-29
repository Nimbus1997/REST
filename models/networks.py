import numbers
from cv2 import norm
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import pywt
import numpy as np

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


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
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

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256' or netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 8, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'unet_512': # ellen made
    #     net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'ellen_uresent':
        # uresent
        net = ellen_uresent(input_nc, output_nc, ngf, norm_layer=norm_layer,
                            use_dropout=use_dropout, num_downs=2, n_blocks=9)
        print(net)
        #defalut - num_down
    elif netG == 'ellen_dwt_uresnet2_1':
        #dwgan + uresnet
        net = ellen_dwt_uresnet2_1(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=9)
    elif netG == 'ellen_dwt_uresnet1_1':
        #uresnet(with dwt)
        net = ellen_dwt_uresnet1_1(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=2, n_blocks=9)
    
    elif netG == 'ellen_dwt_uresnet1_3':
        #uresent(with dwt)
        net = ellen_dwt_uresnet1_3(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=3)
    elif netG == 'ellen_dwt_uresnet1_4':
        #uresent(with dwt)
        net = ellen_dwt_uresnet1_4(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=3)

    elif netG == 'ellen_dwt_uresnet1_5':
        #uresent(with dwt)
        net = ellen_dwt_uresnet1_5(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, num_downs=6)
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
        out = x + self.conv_block(x)  # add skip connections -- cat이 아니라 +라 채널수 안바뀜
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
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        # inner_nc:the # of filters in the inner conv layer
        downnorm = norm_layer(inner_nc)
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

# ---------------------------------------------------------------------------------


class ellen_uresent(nn.Module):
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
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, padding_type='reflect'):
        super(ellen_uresent, self).__init__()
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
        resnet_inUnet = []
        for i in range(n_blocks):
            resnet_inUnet += [ResnetBlock(ngf*multi, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            print("-----resnet", i, " 번째")

        # unet 구조 시작(resnet을 감싸는)
        # i : 0,1,2,3,, n
        # default: 안 64*8
        resnet_inUnets = nn.Sequential(*resnet_inUnet)  # submodule
        unetblock = UnetSkipConnectionBlock(ngf*multi, ngf*multi, input_nc=None, submodule=resnet_inUnets,
                                            norm_layer=norm_layer, innermost=True)  # 젤 안쪽에서 resnet과 닿아 있는 것 > 64*8 64*8
        unetblock = UnetSkipConnectionBlock(int(
            ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer)  # > 64*4 64*8
        print(">> in, out: ", ngf*(multi/2), ", ", ngf*multi)
        for i in range(num_downs-2):
            multi = int(multi/2)
            unetblock = UnetSkipConnectionBlock(int(
                ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer)
            print("-----unet", i, " 번째")
            print(">> in, out: ", ngf*(multi/2), ", ", ngf*multi)
        multi = int(multi/2)
        unetblock = UnetSkipConnectionBlock(int(
            ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer, outermost=True)

        model = model + [unetblock]
        print("----- e1+unet+resent+uent+")

        # d1 크기는 일정, chanel 수 변환64->3
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                   output_nc, kernel_size=7, padding=0), nn.Tanh()]
        print("----- e1+unet+resent+uent+d1")

        self.model = nn.Sequential(*model)
        print(model)

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])

        return self.model(input)


# -------------------------------------------------------------------------------------------------------------
# ellen - from DWGAN _ 2022.04.19

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(
            in_channels*3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' %
                         name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(
            in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(
            in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class dwt_Unet(nn.Module):
    def __init__(self, output_nc=3, nf=16):
        super(dwt_Unet, self).__init__()
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(16, nf-1, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, transposed=False,
                           bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, transposed=False,
                           bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8-8, name, transposed=False,
                           bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8-16, name, transposed=False,
                           bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False,
                           bn=False, relu=False, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 8, nf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16+16, nf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16+8, nf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8+4, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4+2, nf, name, transposed=True,
                            bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2+1, nf * 2, name,
                            transposed=True, bn=True, relu=True, dropout=False)

        self.initial_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = layer1
        self.DWT_down_0 = DWT_transform(3, 1)
        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)
        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)
        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)
        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.tail_conv2 = nn.Conv2d(nf*2, output_nc, 3, padding=1, bias=True)

    def forward(self, x):
        conv_start = self.initial_conv(x)
        conv_start = self.bn1(conv_start)
        conv_out1 = self.layer1(conv_start)
        dwt_low_0, dwt_high_0 = self.DWT_down_0(x)
        out1 = torch.cat([conv_out1, dwt_low_0], 1)
        conv_out2 = self.layer2(out1)
        dwt_low_1, dwt_high_1 = self.DWT_down_1(out1)
        out2 = torch.cat([conv_out2, dwt_low_1], 1)
        conv_out3 = self.layer3(out2)
        dwt_low_2, dwt_high_2 = self.DWT_down_2(out2)
        out3 = torch.cat([conv_out3, dwt_low_2], 1)
        conv_out4 = self.layer4(out3)
        dwt_low_3, dwt_high_3 = self.DWT_down_3(out3)
        out4 = torch.cat([conv_out4, dwt_low_3], 1)
        conv_out5 = self.layer5(out4)
        dwt_low_4, dwt_high_4 = self.DWT_down_4(out4)
        out5 = torch.cat([conv_out5, dwt_low_4], 1)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)

        Tout6_out5 = torch.cat([dout6, out5, dwt_high_4], 1)
        Tout5 = self.dlayer5(Tout6_out5)
        Tout5_out4 = torch.cat([Tout5, out4, dwt_high_3], 1)
        Tout4 = self.dlayer4(Tout5_out4)
        Tout4_out3 = torch.cat([Tout4, out3, dwt_high_2], 1)
        Tout3 = self.dlayer3(Tout4_out3)
        Tout3_out2 = torch.cat([Tout3, out2, dwt_high_1], 1)
        Tout2 = self.dlayer2(Tout3_out2)
        Tout2_out1 = torch.cat([Tout2, out1, dwt_high_0], 1)
        Tout1 = self.dlayer1(Tout2_out1)
        Tout1_outinit = torch.cat([Tout1, conv_start], 1)
        tail1 = self.tail_conv1(Tout1_outinit)
        tail2 = self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)
        return dout1


class ellen_dwt_uresnet2_1(nn.Module):
    """
    made by ellen _2022.04.19 
    dwt_network & uresnet 
    dwt 부분: https://github.com/liuh127/NTIRE-2021-Dehazing-DWGAN/blob/e0d2f4f2dfcfd66bdb77d6aa122392e1bb51cef0/model.py#L184 참고
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, padding_type='reflect'):
        super(ellen_dwt_uresnet2_1, self).__init__()
        self.uresnet = ellen_uresent(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, num_downs=num_downs, n_blocks=n_blocks)
        self.dwt_model = dwt_Unet()
        self.fusion = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        result_uresnet = self.uresnet(input)
        result_dwt = self.dwt_model(input)
        x = torch.cat([result_dwt, result_uresnet], 1)

        return self.fusion(x)

#------------------------------------------------------------------------------------------------------------


class ellen_dwt_uresnet1_1(nn.Module):
    """
    made by ellen _2022.04.19 
    modified by ellen_2022.08.12
        1. dwt-> pywt 안쓰고, DWGAN에서 사용한 dwt사용(cuda issue)
        2. size맞게 변경
    
    => uresnet + dwt transform
    
    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4, n_blocks=3)  
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, n_blocks=9, padding_type='reflect'):
        super(ellen_dwt_uresnet1_1, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # resnet 구조(unet안에 있는)
        # default: 64*4 -> 64*4 반복
        multi = 2**(num_downs-1)  # 내려가는 만큼 채널수 : 64*2^n
        resnet_inUnet = []
        for i in range(n_blocks):
            resnet_inUnet += [ResnetBlock(ngf*multi, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # unet 구조 시작(resnet을 감싸는)
        # i : 0,1,2,3,, n
        # default: 안 64*4
        resnet_inUnets = nn.Sequential(*resnet_inUnet)  # submodule
        unetblock = UnetSkipConnectionBlock(ngf*multi, ngf*multi, input_nc=None, submodule=resnet_inUnets,
                                            norm_layer=norm_layer, innermost=True)  # 젤 안쪽에서 resnet과 닿아 있는 것 > 64*8 64*8
        unetblock = UnetSkipConnectionBlock(int(
            ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer)  # > 64*4 64*8
        for i in range(num_downs-3):  # 0이하면, 어짜피 for문 안들어감
            multi = int(multi/2)
            unetblock = UnetSkipConnectionBlock(int(
                ngf*(multi/2)), ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer)
        multi = int(multi/2)
        unetblock = UnetSkipConnectionBlock(
            output_nc, ngf*multi, input_nc=None, submodule=unetblock, norm_layer=norm_layer, outermost=True)

        model = [unetblock]

        self.model = nn.Sequential(*model)

        print(model)

        self.channel9to3 = nn.Conv2d(9, 3, kernel_size=1, padding=0)
        self.channel6to3nsizeUp = nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1)
        self.dwt = DWT()

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512])
        low_fq, h_fq = self.dwt(input)  # l:[1,3,256,256], h:[1,9,256,256]
        high_f = self.channel9to3(h_fq)  # high: [1,3,256,256]
        low_result = self.model(low_fq)  # low_result:[1,3,256,256]
        total_result = self.channel6to3nsizeUp(torch.cat((low_result, high_f), 1))
        return total_result




class ellen_dwt_uresnet1_3(nn.Module):
    """
    made by ellen _2022.08.12 
    
    dwt_network + uresnet 
        1. dwt 로 hf, lf 나눔
        2. lf는 그냥 내리고, hf는 내리면서 층마다 resnet block
        3. 젤 안쪽 layer에서 lf와 hf합쳐서 올리기 & 위에는 resnet block를 통과한 hf 합치기
        +. 우선은 num_downs=3으로

    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4)  
    input (들어갈때 채널수, 나올때 채널수, conv하고난 후 채널수, dropout 사용, 몇층 내려갈지(=layer1의 resnet block 개수))
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, padding_type='reflect'):
        super(ellen_dwt_uresnet1_3, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d      

        # 통과할 model들 한층한층 따로 정의----
        level2c = ngf
        level3c = ngf*2
        level4c = ngf*(2**2)
        # (1) 앞뒤로 특정 역할 하는 층
        # self.channel9to3 = nn.Conv2d(9, 3, kernel_size=1, padding=0)
        self.channel18to3nsizeUp = nn.ConvTranspose2d(18, output_nc, kernel_size=4, stride=2, padding=1)
        self.dwt = DWT()

        # (2) high& low frequency - down 층
        down_layer1 = nn.Conv2d(3, level2c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        hdown_layer1 = nn.Conv2d(9, level2c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer2 = nn.Conv2d(level2c, level3c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer3 = nn.Conv2d(level3c, level4c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        # for i in range(num_downs-1): #이렇게 한번에 정의 해보려 했으나, 아래서 불러오는게 안되는 것 같음
        #     globals()["down_layer{}".format(i+2)] = nn.Conv2d(ngf*(2**(i+1)), ngf*(2**(i+2)), kernel_size=4, stride=2, padding =1, bais = use_bias)
        
        # (3) high frequency - resnet 층
        resnet11 = []
        for i in range(num_downs):
            resnet11+=[ResnetBlock(9,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet1=nn.Sequential(*resnet11)

        resnet22 = []
        for i in range(num_downs-1):
            resnet22+=[ResnetBlock(level2c,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet2=nn.Sequential(*resnet22)

        resnet33 = []
        for i in range(num_downs-2):
            resnet33+=[ResnetBlock(level3c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet3=nn.Sequential(*resnet33)
        

        # (4) up 층
        up_layer3 = nn.ConvTranspose2d(level4c*2, level3c, kernel_size=4, stride=2, padding=1)
        up_layer2 = nn.ConvTranspose2d(level3c*2, level2c, kernel_size=4, stride=2, padding=1)
        up_layer1 = nn.ConvTranspose2d(level2c*2, 9, kernel_size=4, stride=2, padding=1)

        # 실제 통과할(forward) layer들 정의----
        lowsequence = [down_layer1, down_layer2, down_layer3]
        self.low_downmodel= nn.Sequential(*lowsequence)
        self.hd1= hdown_layer1
        self.hd2= down_layer2
        self.hd3= down_layer3

        self.hr1=resnet1
        self.hr2=resnet2
        self.hr3=resnet3

        self.u3=up_layer3
        self.u2=up_layer2
        self.u1=up_layer1

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512]) [batch 수, 채널수, w, h]
        # print("input.shape", input.shape)
        # print()

        low_fq, h_fq = self.dwt(input)  # l:[1,3,256,256], h:[1,9,256,256]
        low_out4= self.low_downmodel(low_fq)
        # print("h_fq.shape", h_fq.shape)
        # print("low_out4.shape", low_out4.shape)
        # print()
        
        high_dout2=self.hd1(h_fq)
        high_rout1=self.hr1(h_fq)
        # print("high_dout2.shape", high_dout2.shape)
        # print("high_rout1.shape", high_rout1.shape)
        # print()

        high_dout3=self.hd2(high_dout2)
        high_rout2=self.hr2(high_dout2)
        # print("high_dout3.shape", high_dout3.shape)
        # print("high_rout2.shape", high_rout2.shape)
        # print()

        high_dout4=self.hd3(high_dout3)
        high_rout3=self.hr3(high_dout3)
        # print("high_dout4.shape", high_dout4.shape)
        # print("high_rout3.shape", high_rout3.shape)
        # print()
        # print("u3 input size", torch.cat([high_dout4, low_out4],1).shape)
        up3=self.u3(torch.cat([high_dout4, low_out4],1))
        up2=self.u2(torch.cat([high_rout3,up3],1))
        up1=self.u1(torch.cat([high_rout2,up2],1))

        result= self.channel18to3nsizeUp(torch.cat([high_rout1, up1],1))

        return result


class ellen_dwt_uresnet1_4(nn.Module):
    """
    made by ellen _2022.08.16 
    
    dwt_network + uresnet 
        1. dwt 로 hf, lf 나눔
        2. lf는 그냥 내리고, hf는 내리면서 층마다 resnet block
        3. 젤 안쪽 layer에서 lf와 hf합쳐서 올리기 & 위에는 resnet block를 통과한 hf 합치기
        +. 우선은 num_downs=3으로

    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4)  
    input (들어갈때 채널수, 나올때 채널수, conv하고난 후 채널수, dropout 사용, 몇층 내려갈지(=layer1의 resnet block 개수))
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=3, padding_type='reflect'):
        super(ellen_dwt_uresnet1_4, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d      

        # 통과할 model들 한층한층 따로 정의----
        level2c = ngf
        level3c = ngf*2
        level4c = ngf*(2**2)
        # (1) 앞뒤로 특정 역할 하는 층
        # self.channel9to3 = nn.Conv2d(9, 3, kernel_size=1, padding=0)
        self.channel18to3nsizeUp = nn.ConvTranspose2d(18, output_nc, kernel_size=4, stride=2, padding=1)
        self.dwt = DWT()

        # (2) high& low frequency - down 층
        odown_layer0 = nn.Conv2d(3, 9, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer1 = nn.Conv2d(9, level2c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer2 = nn.Conv2d(level2c, level3c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer3 = nn.Conv2d(level3c, level4c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        # for i in range(num_downs-1): #이렇게 한번에 정의 해보려 했으나, 아래서 불러오는게 안되는 것 같음
        #     globals()["down_layer{}".format(i+2)] = nn.Conv2d(ngf*(2**(i+1)), ngf*(2**(i+2)), kernel_size=4, stride=2, padding =1, bais = use_bias)
        
        # (3) high frequency - resnet 층
        resnet11 = []
        for i in range(num_downs):
            resnet11+=[ResnetBlock(9,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet1=nn.Sequential(*resnet11)

        resnet22 = []
        for i in range(num_downs-1):
            resnet22+=[ResnetBlock(level2c,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet2=nn.Sequential(*resnet22)

        resnet33 = []
        for i in range(num_downs-2):
            resnet33+=[ResnetBlock(level3c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet3=nn.Sequential(*resnet33)
        

        # (4) up 층
        up_layer3 = nn.ConvTranspose2d(level4c*2, level3c, kernel_size=4, stride=2, padding=1)
        up_layer2 = nn.ConvTranspose2d(level3c*2, level2c, kernel_size=4, stride=2, padding=1)
        up_layer1 = nn.ConvTranspose2d(level2c*2, 9, kernel_size=4, stride=2, padding=1)

        # 실제 통과할(forward) layer들 정의----
        orisequence = [odown_layer0, down_layer1, down_layer2, down_layer3]
        self.ori_downmodel= nn.Sequential(*orisequence)
        self.hd1= down_layer1
        self.hd2= down_layer2
        self.hd3= down_layer3

        self.hr1=resnet1
        self.hr2=resnet2
        self.hr3=resnet3

        self.u3=up_layer3
        self.u2=up_layer2
        self.u1=up_layer1

    def forward(self, input):
        """Standard forward"""
   

        _low_fq, h_fq = self.dwt(input)  # l:[1,3,256,256], h:[1,9,256,256]
        ori_out4= self.ori_downmodel(input)
        # print("h_fq.shape", h_fq.shape)
        # print("low_out4.shape", low_out4.shape)
        # print()
        
        high_dout2=self.hd1(h_fq)
        high_rout1=self.hr1(h_fq)
        # print("high_dout2.shape", high_dout2.shape)
        # print("high_rout1.shape", high_rout1.shape)
        # print()

        high_dout3=self.hd2(high_dout2)
        high_rout2=self.hr2(high_dout2)
        # print("high_dout3.shape", high_dout3.shape)
        # print("high_rout2.shape", high_rout2.shape)
        # print()

        high_dout4=self.hd3(high_dout3)
        high_rout3=self.hr3(high_dout3)
        # print("high_dout4.shape", high_dout4.shape)
        # print("high_rout3.shape", high_rout3.shape)
        # print()
        # print("u3 input size", torch.cat([high_dout4, low_out4],1).shape)
        up3=self.u3(torch.cat([high_dout4, ori_out4],1))
        up2=self.u2(torch.cat([high_rout3,up3],1))
        up1=self.u1(torch.cat([high_rout2,up2],1))

        result= self.channel18to3nsizeUp(torch.cat([high_rout1, up1],1))

        return result

class ellen_dwt_uresnet1_5(nn.Module):
    """
    made by ellen _2022.08.17
    
    dwt1_3과 같은 구조인데, down을 3번 말고 6번

    input (input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_downs=4)  
    input (들어갈때 채널수, 나올때 채널수, conv하고난 후 채널수, dropout 사용, 몇층 내려갈지(=layer1의 resnet block 개수))
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, num_downs=6, padding_type='reflect'):
        super(ellen_dwt_uresnet1_5, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d      

        # 통과할 model들 한층한층 따로 정의----
        level2c = ngf
        level3c = ngf*2
        level4c = ngf*(2**2)
        level5c = ngf*(2**3)
        level6c = ngf*(2**4)
        level7c = ngf*(2**5)

        # (1) 앞뒤로 특정 역할 하는 층
        # self.channel9to3 = nn.Conv2d(9, 3, kernel_size=1, padding=0)
        self.channel18to3nsizeUp = nn.ConvTranspose2d(18, output_nc, kernel_size=4, stride=2, padding=1)
        self.dwt = DWT()

        # (2) high& low frequency - down 층
        down_layer1 = nn.Conv2d(3, level2c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        hdown_layer1 = nn.Conv2d(9, level2c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer2 = nn.Conv2d(level2c, level3c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer3 = nn.Conv2d(level3c, level4c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer4 = nn.Conv2d(level4c, level5c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer5 = nn.Conv2d(level5c, level6c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_layer6 = nn.Conv2d(level6c, level7c, kernel_size=4, stride=2, padding=1, bias=use_bias)

        # for i in range(num_downs-1): #이렇게 한번에 정의 해보려 했으나, 아래서 불러오는게 안되는 것 같음
        #     globals()["down_layer{}".format(i+2)] = nn.Conv2d(ngf*(2**(i+1)), ngf*(2**(i+2)), kernel_size=4, stride=2, padding =1, bais = use_bias)
        
        # (3) high frequency - resnet 층
        resnet11 = []
        for i in range(num_downs):
            resnet11+=[ResnetBlock(9,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet1=nn.Sequential(*resnet11)

        resnet22 = []
        for i in range(num_downs-1):
            resnet22+=[ResnetBlock(level2c,padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet2=nn.Sequential(*resnet22)

        resnet33 = []
        for i in range(num_downs-2):
            resnet33+=[ResnetBlock(level3c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet3=nn.Sequential(*resnet33)

        resnet44 = []
        for i in range(num_downs-3):
            resnet44+=[ResnetBlock(level4c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet4=nn.Sequential(*resnet44)

        resnet55 = []
        for i in range(num_downs-4):
            resnet55+=[ResnetBlock(level5c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet5=nn.Sequential(*resnet55)

        resnet66 = []
        for i in range(num_downs-5):
            resnet66+=[ResnetBlock(level6c, padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        resnet6=nn.Sequential(*resnet66)
        

        # (4) up 층
        up_layer6 = nn.ConvTranspose2d(level7c*2, level6c, kernel_size=4, stride=2, padding=1)
        up_layer5 = nn.ConvTranspose2d(level6c*2, level5c, kernel_size=4, stride=2, padding=1)
        up_layer4 = nn.ConvTranspose2d(level5c*2, level4c, kernel_size=4, stride=2, padding=1)
        up_layer3 = nn.ConvTranspose2d(level4c*2, level3c, kernel_size=4, stride=2, padding=1)
        up_layer2 = nn.ConvTranspose2d(level3c*2, level2c, kernel_size=4, stride=2, padding=1)
        up_layer1 = nn.ConvTranspose2d(level2c*2, 9, kernel_size=4, stride=2, padding=1)

        # 실제 통과할(forward) layer들 정의----
        lowsequence = [down_layer1, down_layer2, down_layer3,down_layer4, down_layer5, down_layer6]
        self.low_downmodel= nn.Sequential(*lowsequence)
        self.hd1= hdown_layer1
        self.hd2= down_layer2
        self.hd3= down_layer3
        self.hd4= down_layer4
        self.hd5= down_layer5
        self.hd6= down_layer6
        
        self.hr1=resnet1
        self.hr2=resnet2
        self.hr3=resnet3
        self.hr4=resnet4
        self.hr5=resnet5
        self.hr6=resnet6

        self.u6=up_layer6
        self.u5=up_layer5
        self.u4=up_layer4
        self.u3=up_layer3
        self.u2=up_layer2
        self.u1=up_layer1

    def forward(self, input):
        """Standard forward"""
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.shape) # torch.Size([1, 3, 512, 512]) [batch 수, 채널수, w, h]

        low_fq, h_fq = self.dwt(input)  # l:[1,3,256,256], h:[1,9,256,256]
        low_out7= self.low_downmodel(low_fq)
        
        high_dout2=self.hd1(h_fq)
        high_rout1=self.hr1(h_fq)

        high_dout3=self.hd2(high_dout2)
        high_rout2=self.hr2(high_dout2)

        high_dout4=self.hd3(high_dout3)
        high_rout3=self.hr3(high_dout3)

        high_dout5=self.hd4(high_dout4)
        high_rout4=self.hr4(high_dout4)

        high_dout6=self.hd5(high_dout5)
        high_rout5=self.hr5(high_dout5)

        high_dout7=self.hd6(high_dout6)
        high_rout6=self.hr6(high_dout6)

        up6=self.u6(torch.cat([high_dout7, low_out7],1))
        up5=self.u5(torch.cat([high_rout6, up6],1))
        up4=self.u4(torch.cat([high_rout5, up5],1))
        up3=self.u3(torch.cat([high_rout4, up4],1))
        up2=self.u2(torch.cat([high_rout3,up3],1))
        up1=self.u1(torch.cat([high_rout2,up2],1))

        result= self.channel18to3nsizeUp(torch.cat([high_rout1, up1],1))

        return result

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
