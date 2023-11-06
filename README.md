
# REST
This is the code for the MICCAI 2023 Paper "RESToring Clarity: Unpaired Retina Image Enhancement using Scattering Transform".  
![image](https://github.com/Nimbus1997/REST/assets/66589193/99a433fc-d648-4f08-ae80-2f53084d4100)


## Links
paper: https://link.springer.com/chapter/10.1007/978-3-031-43999-5_45  
reviews, feedback: https://conferences.miccai.org/2023/papers/540-Paper2936.html  

  

## Model Architecture
###  Architecture of the Generator
![image](https://github.com/Nimbus1997/REST/assets/66589193/c0e4abcd-73a1-4f1a-9926-90670434984f)


### Overall architecture of REST for training
![image](https://github.com/Nimbus1997/REST/assets/66589193/c773670c-2a5d-465f-a51e-b39823fe9339)


    
## Result
![image](https://github.com/Nimbus1997/REST/assets/66589193/e87746d7-ebc0-4c9c-96f7-0ae5db3805be)

## Implementation
### Dataset preparation (splitting)
Our code is based on [Cycle GAN] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.  
Check the 'dataset' folder on the code or [Cycle GAN] explanation for details.

### run
#### training example
```python train.py --dataroot /root/jieunoh/ellen_data/1_ukb_cyclegan_input/ukb_512_1 --name ukb1_ellen53_0508 --fiqa_epoch 10 --save_epoch_freq 10 --model cycle_gan --direction AtoB --gpu_ids 0 --batch_size 4 --no_flip --load_size 256 --crop_size 256 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --netG REST```  
#### test example
```python test.py --dataroot /root/jieunoh/ellen_data/1_ukb_cyclegan_input/ukb_512_1 --name ukb1_ellen23_b2_0227 --model cycle_gan --direction AtoB --gpu_ids 2 --use_wandb --no_flip --load_size 256 --crop_size 256 --netG REST --epoch 80 --ellen_test```

## Acknowledgments
Our code is inspired by [Cycle GAN] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
