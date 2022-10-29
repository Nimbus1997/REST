"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

----------------------------------------------------------------------------------------------------------------
Ellen modified
2022.06.30 : early stopping using validataion generator loss
2022.07.12 : early stopping using FID(frechet inception distance)
            - 개념 : https://jjuon.tistory.com/33 
            - []pytorch FID module official: https://pytorch.org/ignite/generated/ignite.metrics.FID.html , https://github.com/pytorch/ignite
            - [v]pytorch FID module base of the official: https://github.com/mseitzer/pytorch-fid/tree/3d604a25516746c3a4a5548c8610e99010b2c819 
"""
from cmath import inf 
from random import triangular
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt

####### To fix the random seed -- ellen ###
import torch
import numpy as np
import random 

random_seed = 42
torch.manual_seed(random_seed) #1.pytorch randomness
torch.backends.cudnn.deteministic = True #2.cuDNN randomness - might make computaion slow
torch.backedns.cudnn.benchmark = False
np.random.seed(random_seed) #3.numpy randomness
random.seed(random_seed) #4.python randomness
##########################################

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # for early stopping - ellen  --------------
    opt.phase = "val"
    val_dataset = create_dataset(opt)
    last_val_loss_G = inf
    patience = opt.patience
    print("patience:", patience)
    patience_count = 0

    opt.phase = "train"
    # for loss and early stopping visualization - ellen ---------------------
    train_loss_G =[]
    val_loss_G =[]
    stopped_epoch = opt.epoch_count + opt.n_epochs + opt.n_epochs_decay


    # create a model given opt.model and other options
    # print(">>>>>>>>>>>>>>>>>[1]")
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    # print(">>>>>>>>>>>>>>>>>[2]")
    model.setup(opt)
    # create a visualizer that display/save images and plots
    # print(">>>>>>>>>>>>>>>>>[3]")
    visualizer = Visualizer(opt)
    total_iters = 0                # the total number of training iterations

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # epoch_count: staring epoch count
        # n_epochs: number of epochs with the initial learning rate
        # n_epochs_decay: number of epochs to linearly deay learning rate to zero

        # training -----------------------------------
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of tpraining iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # print(">>>>>>>>>>>>>>>>>[4]")
        visualizer.reset()
        # update learning rates in the beginning of every epoch.
        # print(">>>>>>>>>>>>>>>>>[5]")
        model.update_learning_rate()

        iter_current_train_loss_G =[]#ellen -  for early stopping visualization 
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print(">>>>>>>>>>>>>>>>>[6]")
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            # print(">>>>>>>>>>>>>>>>>[7]")
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            # print(">>>>>>>>>>>>>>>>>[8]")
            model.optimize_parameters()
            # print(">>>>>>>>>>>>>>>>>[8]")


            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                # print(">>>>>>>>>>>>>>>>>[9]")
                model.compute_visuals()
                # print(">>>>>>>>>>>>>>>>>[10]")
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result)

            # print(">>>>>>>>>>>>>>>>>[11]")
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses)
            
            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' %
            #           (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)
            # print(">>>>>>>>>>>>>>>>>[12]")
            iter_current_train_loss_G.append(model.get_current_loss_G()) #ellen -  for early stopping visualization

            iter_data_time = time.time()
        current_train_loss_G = np.average(iter_current_train_loss_G)
        train_loss_G.append(current_train_loss_G)
        # validation for early stopping - ellen ------------------------------
        iter_current_val_loss_G = []
        for i, data, in enumerate(val_dataset):
            model.set_input(data)
            iter_current_val_loss_G.append(model.get_current_loss_G())

        current_val_loss_G = np.average(iter_current_val_loss_G)
        val_loss_G.append(current_val_loss_G) # for visualization 

        if current_val_loss_G > last_val_loss_G:
            patience_count += 1
            if patience_count >= patience:
                print("--------------------------------------------------------")
                print("Stopped because it passed patience %d times, Latest update: %d epoch" %(patience, stopped_epoch))
                break
        else:
            stopped_epoch = epoch
            model.save_networks('latest')
            patience_count =0
        last_val_loss_G = current_val_loss_G
        # --------------------------------------------------------

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:             
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            # model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # for early stopping visualization - ellen ------------------------------
    loss_fig= plt.figure(figsize=(10,8))
    plt.plot(range(1, len(train_loss_G)+1), train_loss_G, label="Training Loss")
    plt.plot(range(1,  len(val_loss_G)+1), val_loss_G, label= "Validation Loss")
    plt.axvline(stopped_epoch, linestyle='--', color='r', label="Early Stopping CheckPoint: "+str(stopped_epoch))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(opt.name)
    plt.ylim(0,5)
    plt.xlim(0,len(train_loss_G)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    loss_fig.savefig(opt.checkpoints_dir+"/"+opt.name+"/loss_plot.png", bbox_inches='tight')
    
