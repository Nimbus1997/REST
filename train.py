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

####### To fix the random seed & fiqa -- ellen ###########
import torch
import numpy as np
import random 
import pdb
import os
import sys # fiqa
sys.path.append("/home/guest1/ellen_code/eyeQ_ellen/MCF_Net") #fiqa -medi change!!
# sys.path.append("/root/jieunoh/ellen_code/eyeQ_ellen/MCF_Net") #fiqa -miv2 
from Main_EyeQuality_train_func import FIQA_during_training


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    pathh = os.path.join("/home/guest1/ellen_code/pytorch-CycleGAN-and-pix2pix_ellen/checkpoints", opt.name, "temp")#fiqa -medi change!!
    # pathh = os.path.join("/root/jieunoh/ellen_code/RetinaImage_model_MW/checkpoints", opt.name, "temp")#fiqa -miv2 change!! # eyeQ path change 필요
   

    random_seed = opt.random_seed
    np.random.seed(random_seed) #1.numpy randomness
    random.seed(random_seed) #2.python randomness
    torch.manual_seed(random_seed) #3.pytorch randomness
    torch.cuda.manual_seed(random_seed) # 4. gpu randomness 
    torch.cuda.manual_seed_all(random_seed) # 4. gpu randomness - multi gpu
    torch.backends.cudnn.deteministic = True #5.cuDNN randomness - might make computaion slow
    torch.backends.cudnn.benchmark = False
    # 6. Data loader randomness in multi process fix -> in /data/__init__.py -> ellen_made
    os.environ['PYTHONHASHSEED'] = str(random_seed)  #7.python hash seed 고정
    ##################################################

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
    # for loss and early stopping visualization & FIQA - ellen ---------------------
    train_loss_G =[]
    val_loss_G =[]
    stopped_epoch = opt.epoch_count + opt.n_epochs + opt.n_epochs_decay

    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, "temp")):
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name, "temp"))
    fiqa_list=[]
    # create a model given opt.model and other options
    model = create_model(opt)
    
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
   
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    total_iters = 0  # the total number of training iterations
    val_fiqa_iters =0

    # ellen 
    best_fiqa=0.0

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
        visualizer.reset()
        # update learning rates in the beginning of every epoch.
        model.update_learning_rate()

        iter_current_train_loss_G =[]#ellen -  for early stopping visualization 
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()


            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result)
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                    epoch, float(epoch_iter) / dataset_size, losses)
            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_current_train_loss_G.append(model.get_current_loss_G()) #ellen -  for early stopping visualization

            iter_data_time = time.time()
        current_train_loss_G = np.average(iter_current_train_loss_G)
        train_loss_G.append(current_train_loss_G)
        torch.cuda.empty_cache()

        model.eval()
        # validation ellen ----------------------------------------------------------------------
        iter_current_val_loss_G = []
        for i, data in enumerate(val_dataset):
            print("[val] (epoch:", epoch, ", iter: ", i,")")
            with torch.no_grad(): # 이렇게 해야지 gpu 사용량 안늘면서 돌아감 train아닐때
                model.set_input(data)                
                iter_current_val_loss_G.append(float(model.forward_val_get_loss()))# val loss가져오기
                if epoch%opt.fiqa_epoch==0:
                    model.save_fake_B() # image 저장
            torch.cuda.empty_cache()
        current_val_loss_G = np.average(iter_current_val_loss_G)
        val_loss_G.append(current_val_loss_G)

        if epoch%opt.fiqa_epoch == 0:
            fiqa=FIQA_during_training(opt.name, pathh)
            fiqa_list.append(fiqa)

            # LOSS
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('loss')
            ax1.plot(range(1, len(train_loss_G)+1), train_loss_G, color='mediumseagreen', label="Training Loss")
            ax1.plot(range(1,  len(val_loss_G)+1), val_loss_G, color='dodgerblue', label= "Validation Loss")
            ax1.legend(loc='upper left')
            ax1.set_ylim([0,10])
            #FIQA
            ax2=ax1.twinx()
            ax2.set_ylabel("FIQA")
            ax2.plot(range(1, len(val_loss_G)+1, opt.fiqa_epoch),fiqa_list, color='palevioletred', marker='o', linestyle='--', label= "FIQA")
            ax2.legend(loc='upper right')
            ax2.set_ylim([0,1])
            ax2.set_yticks(np.arange(0,1,0.05))
            #EARLY STOPPING
            plt.axvline(stopped_epoch, linestyle='--', color='r', label="Early Stopping CheckPoint: "+str(stopped_epoch))
            
            plt.title(opt.name)    
            plt.xlim(0,len(train_loss_G)+1)
            plt.xticks(range(0,len(train_loss_G)+1, 50))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(opt.checkpoints_dir+"/"+opt.name+"/0_loss_plot.png", bbox_inches='tight')
            # if epoch 몇 -> 저장한 val FIQA
            # 위에 impot Main_eyeQuality_trian_func
            # loss그래프에 같이 그리기 

            if fiqa >= best_fiqa: # fiqa가 가장 높으면, model save & output save
                print("best_fiqa")
                save_suffix = '0_best_fiqa_%d' % epoch 
                model.save_networks(save_suffix)
                model.save_best_fake_B()

        # # for early stopping visualization - ellen --------------------------------------------------
        # current_val_loss_G = np.average(iter_current_val_loss_G)
        # val_loss_G.append(current_val_loss_G) # for visualization 

        # if current_val_loss_G > last_val_loss_G:
        #     patience_count += 1
        #     if patience_count >= patience:
        #         print("--------------------------------------------------------")
        #         print("Stopped because it passed patience %d times, Latest update: %d epoch" %(patience, stopped_epoch))
        #         break
        # else:
        #     stopped_epoch = epoch
        #     model.save_networks('latest')
        #     patience_count =0
        # last_val_loss_G = current_val_loss_G
        # # ------------------------------------------------------------------------------------------------

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:             
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            # model.save_networks('latest')
            model.save_networks(epoch)



        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        
        model.train()

    # for early stopping visualization - ellen --------------------------------------------------
    # LOSS
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(range(1, len(train_loss_G)+1), train_loss_G, color='mediumseagreen', label="Training Loss")
    ax1.plot(range(1,  len(val_loss_G)+1), val_loss_G, color='dodgerblue', label= "Validation Loss")
    ax1.legend(loc='upper left')
    ax1.set_ylim([0,10])
    #FIQA
    ax2=ax1.twinx()
    ax2.set_ylabel("FIQA")
    ax2.plot(range(1, len(val_loss_G)+1,opt.fiqa_epoch),fiqa_list, color='palevioletred', marker='o', linestyle='--', label= "FIQA")
    ax2.legend(loc='upper right')
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.arange(0,1,0.05))
    #EARLY STOPPING
    plt.axvline(stopped_epoch, linestyle='--', color='r', label="Early Stopping CheckPoint: "+str(stopped_epoch))
    
    plt.title(opt.name)    
    plt.xlim(0,len(train_loss_G)+1)
    plt.xticks(range(0,len(train_loss_G)+1, 50))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(opt.checkpoints_dir+"/"+opt.name+"/0_loss_plot.png", bbox_inches='tight')