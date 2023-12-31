"""ellen made
date: 2022.11.18

1. To see the image made by intermediate model like unet and scattering branch.
2. both model outputs 3*H*W images so it can be save as images.

for model 2_3


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
import pdb
from collections import OrderedDict
from util import util, html # ellen changed
from subprocess import Popen, PIPE
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def channel_spilt(tensorimage, ret):
    im = util.tensor2im(tensorimage)
    for i in range(3):
        imm = im1[:,:,i]


    return

def save_images_branch_1(webpage, visuals, typee,len, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (tensor)    -- ellen change -> just an tensor
        type(string)          --ellen made -> name of the model
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    typee = typee
    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    
    label="fakeB"
    im_data=visuals
    # pdb.set_trace()
    for i in range(len):
        image_name = '%s_%s_%s%s.png' % (name, label, typee,str(i))
        save_path = os.path.join(image_dir, image_name)
        im = im_data[:,i,:,:]
        
        imm = util.tensor2im_val(im)
        imm=imm[:,:,0]
        h = imm.shape[0]
        # pdb.set_trace()
        # imm =imm.reshape((h,h,1))
        util.save_image_(imm, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(imm)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)





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

    
    model_uresnet=model.netG_A.module.uresnet # ellen made change this
    model_scattering=model.netG_A.module.scattering_model # ellen made change this
    model_scattering1_layer=model.netG_A.module.scattering_model.layer1
    model_scatterings1_layer=model.netG_A.module.scattering_model.scattering_down_1
    model_scattering2_layer=model.netG_A.module.scattering_model.layer2

    model_totalG = model.netG_A # ellen made
    

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN_test_intermidiate', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN_test_intermidiate')

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
    
    label_c3 = ["realA", "fakeB"," fakeB_scatter","fakeB_unet"]
    lable_c1 =["fakeB_unet1", "fakeB_unet2", "fakeB_unet3", "fakeB_scatter1","fakeB_scatter2","fakeB_scatter3"]


    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        visual_retc3 = OrderedDict()
        visual_ret1 = OrderedDict()

        

        realA=data['A'].to(device) # ellen type:torch.Tensor
        output_1 = model_scattering1_layer(realA)
        soutput_1 = model_scatterings1_layer(realA)
        out_total_1= torch.cat([output_1,soutput_1],1)
        output_2 = model_scattering2_layer(out_total_1)


        img_path = data['A_paths']
    
        save_images_branch_1(webpage, output_1, "first",15,img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        save_images_branch_1(webpage, soutput_1, "first_scat",1,img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        save_images_branch_1(webpage, output_2, "second",30,img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    webpage.save()  # save the HTML
