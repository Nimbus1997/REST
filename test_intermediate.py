"""ellen made
date: 2022.11.18

to see the image made by intermediate model like unet and scattering branch.
both model outputs 3*H*W images so it can be save as images.


"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images # self made
from util import html
import random
import torch# To fix the random seed -- ellen
import numpy as np# To fix the random seed -- ellen
import random # To fix the random seed -- ellen

####################################START for makeing save_image _ellen
import numpy as np
import os
import sys
import ntpath
import time
from util import util, html # ellen changed
from subprocess import Popen, PIPE
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (tensor)    -- ellen change -> just an tensor
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    
    label="B"
    im_data=visuals
    im = util.tensor2im(im_data)
    image_name = '%s_%s.png' % (name, label)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(im, save_path, aspect_ratio=aspect_ratio)
    ims.append(image_name)
    txts.append(label)
    links.append(image_name)
    if use_wandb:
        ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)

####################################END for makeing save_image _ellen

# Randomness fix
random_seed = 42
torch.manual_seed(random_seed) #1.pytorch randomness
torch.backends.cudnn.deteministic = True #2.cuDNN randomness - might make computaion slow
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed) #3.numpy randomness
random.seed(random_seed) #4.python randomness
torch.cuda.manual_seed(random_seed) # 5. gpu randomness -> hanna 


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model=model.netG_A.module.scattering_model # ellen made
    # model=model.netG_A.module.uresnet # ellen made

    

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval: #false
        model.eval()
    
        
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        realA=data['A'].to(device) # ellen
        fake_B=model(realA) #elleh - output image
        img_path = data['A_paths']
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(webpage, fake_B, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
