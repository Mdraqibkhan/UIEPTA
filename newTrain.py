from __future__ import print_function
import argparse
import os
from math import log10

from utils import save_img,VGGPerceptualLoss,torchPSNR,ssim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import cv2
import kornia
from torchvision.transforms.functional import pad
import torchvision.transforms as transforms
from pytorch_msssim import MS_SSIM



from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate,rgb_to_y,ContrastLoss
from modifiedmodel1 import Restormer
from data import get_training_set, get_test_set
from loss import Gradient_Loss,L1_Charbonnier_loss



# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=False, default = './uw_data/', help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--finetune', default=True, help='to finetune')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=3000, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--edge_loss', default=True, help='apply edge loss for training')
parser.add_argument('--edge_loss_type', default='sobel', help='apply canny or sobel loss loss for training')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")


if opt.finetune is True :
    my_net = Restormer().to(device)
    # net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
    my_net = my_net.cuda(device = 0)
    net_d = net_d.cuda(device = 0)

else:
    G_path = "checkpoint/uw_data/netG_model_epoch_2.pth".format(opt.epoch_count)
    my_net = torch.load(G_path).cuda(device = 0)

    D_path = "checkpoint/uw_data/netD_model_epoch_2.pth".format(opt.epoch_count)
    net_d = torch.load(D_path).cuda(device = 0)


  

print('===> Loading datasets')
root_path = "uw_data/"
train_set = get_training_set(opt.dataset)
test_set = get_test_set(opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


print('===> Building models')
# net_g = Unet(3,3)#define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
# input_size = 256
# # arch = Uformer
# depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
# net_g = Uformer(img_size=input_size, embed_dim=16,depths=depths,
#                  win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
net_g= Restormer().to(device)

net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
L_per = VGGPerceptualLoss().to(device)
L1_loss = nn.L1Loss().cuda()

MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
Gradient_loss = Gradient_Loss().to(device)
Charbonnier_loss =L1_Charbonnier_loss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)
print('parameters of model are',sum(dict((p.data_ptr(), p.numel()) for p in net_g.parameters()).values()))
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        rgb, tar , rgb_grey= batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # rgb=pad_img( rgb, (256,256))
        # tar=pad_img(tar,(256,256))
        # rgb_grey=pad_img(rgb_grey, (256,256))
        # tar_grey=pad_img(tar_grey,(256,256))
        # real_a, real_b = batch[0].to(device), batch[1].to(device)
        # real_b = real_b.unsqueeze(0)
        # real_a = real_a.unsqueeze(0)
        # print(rgb.shape,rgb_grey.shape)
        fake_b = net_g(rgb,rgb_grey)
        fake_bY=rgb_to_y(fake_b)
        rgb_Y = rgb_to_y(rgb)
        tar_Y= rgb_to_y(tar)
        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake

        
        fake_ab = torch.cat((rgb, fake_b), 1)


        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)
        
        # loss_con=L_vgg(fake_b[0], real_b[0])
                     

        # train with real
      
        real_ab = torch.cat((rgb, tar), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # real_ab1 = torch.cat((fake_b, real_b), 1)
        # pred_real1 = net_d.forward(real_ab1)
        # loss_per=L_vgg(pred_real1, True)
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5 

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((rgb, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
        #same size as an input size
        
        # Second, G(A) = B
        # phase_loss=criterionL1((torch.fft.fftn(fake_b)).angle(),(torch.fft.fftn(tar)).angle())
        loss_msssim_SR = 1 - MS_SSIM_loss(fake_b, tar)
        loss_g_l1 = 2*L1_Charbonnier_loss(fake_b[:,:,:,1], tar[:,:,:,1])+3*L1_Charbonnier_loss(tar_Y,fake_bY) +1.5*L_per(fake_b,tar)+5*loss_msssim_SR
        #####edg_loss###33

        loss_g = loss_g_gan*0.01  + loss_g_l1 
        
        loss_g.backward()

        optimizer_g.step()

        if iteration % 1000==0:
            out_image=torch.cat((rgb, fake_b,tar), 3)
            out_image = out_image[0].detach().squeeze(0).cpu()

            # input_rgb = rgb[0].detach().squeeze(0).cpu()
            # GT_rgb = tar[0].detach().squeeze(0).cpu()
            
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} ".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item() ))
            save_img(out_image, 'images/'+str(iteration)+'.png')

            # save_img(input_rgb,'images/'+str(iteration)+'inp'+'.jpg')
            # save_img1(out_image,'images/'+str(iteration)+'out'+'.jpg')
            # save_img(GT_rgb,'images/'+str(iteration)+'GT'+'.jpg')


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    SSIM=0
    for test_iter, batch in enumerate(testing_data_loader,1):
        rgb_input, target,grey_input = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # input = input.unsqueeze(0)
        # target = target.unsqueeze(0)
        # rgb_input=pad_img(rgb_input, (256,256))
        # target=pad_img(target,(256,256))
        # grey_input=pad_img(rgb_grey, (256,256))
        # grey_target=pad_img(grey_target,(256,256))
        prediction = net_g(rgb_input,grey_input)
        out=torch.cat((rgb_input,prediction,target),3)
        output_cat =out[0].detach().squeeze(0).cpu()
        # inp =rgb_input[0].detach().squeeze(0).cpu()
        # out =prediction[0].detach().squeeze(0).cpu()
        # GT  =target[0].detach().squeeze(0).cpu()
        save_img(output_cat, 'images_test/'+str(test_iter)+'inp'+'.jpg')

        # save_img(inp, 'images_test/'+str(test_iter)+'inp'+'.jpg')
        # save_img1(out, 'images_test/'+str(test_iter)+'out'+'.jpg')
        # save_img(GT, 'images_test/'+str(test_iter)+'GT'+'.jpg')
    
    #     SSim=ssim(fake_b,tar,255)
    #     SSIM+=SSim
        # SSim=torchSSIM(fake_b,tar)
        # SSIM+=SSim
        psnr= torchPSNR(prediction, target).item()
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f}dB ".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 1== 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))