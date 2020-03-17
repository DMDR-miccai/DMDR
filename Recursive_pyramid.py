# Ameneh Jan 28, 2020. This code is for left aitrum but folder that its datas have patient in their name
# from skimage import io, measure
import numpy as np
import matplotlib.pyplot as plt
import torch
from losses import *
from Nets import *
import torch.nn.functional as F
import time
# from torch.autograd import Variable
import torch.optim as optim
from skimage.transform import pyramid_gaussian
from skimage.filters import gaussian
# from skimage.filters import threshold_otsu
# from torch.autograd import Function
from scipy.spatial.distance import directed_hausdorff
import skimage
import nibabel
from Sunny_Call import *

# import glob
# # from IPython.display import clear_output
import os
import glob
from skimage.transform import resize
from ACDC_Call import ACDC_data
import random
from opts import *
print(torch.cuda.current_device())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seed = 0
# from skimage.metrics import structural_similarity as ssim
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
Method ='Sunny' # 'ACDC' #'Sunny' #'Left' #
def _init_fn(worker_id):
    np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Is_ACDC = True
ch_list = ['4ch']
patient_list = []
# init_val = 0.01
n_neurons = 100

n_kernels = 24
kernel_size = 5
padding = int((kernel_size - 1) / 2)

downscale = 2.0
ifplot = True
sampling = 0.01  # 10% sampling
maxiter =800

lam = 4
L = 3# where registration starts (at the coarsest resolution)
SSIM = False
MI = False
MSE = True
Euler = True
R4K = False
loss_str= ""
if SSIM:
    loss_str =loss_str+ "SSIM_"
if MSE:
    loss_str =loss_str+ "MSE_"
if MI:
    loss_str =loss_str+ "MI_"
if R4K:
    loss_str = loss_str + "R4K2"
if Euler:
    loss_str = loss_str + "Euler"


def HD_to_mmHD(contor,pixdim):

    contor[:, 0] *= pixdim[0]
    contor[:, 1] *= pixdim[1]
    contor.sum()
    return contor

def deformation(net,mine_net,gauss1,gauss2, in_lst, h_lst, w_lst, xy_lst, J_lst, nChannel, I_lst, ind_lst,in_rev_lst, train=True):
    loss = 0.0
    # Note that (L+1)/2 levels are used in cost function
    # Also note that I am not using Gaussian smoothing,
    # however, I am using a L2 penalty for deformation in the cost function
    for s in np.arange(L - 1, -1, -2):
        if s == L - 1:
            d_ = net(torch.cat([in_lst[s], torch.zeros(h_lst[s], w_lst[s], 2).to(device)], 2))
            d_rev = net(torch.cat([in_rev_lst[s], torch.zeros(h_lst[s], w_lst[s], 2).to(device)], 2))
        else:
            # kumar's idea:
            # d_rev = 0.5 * d_ - 0.5 * d_rev
            # d_ = -d_rev
            if Euler:

                d_up = F.grid_sample(d_.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2, 0)
                in_ = torch.cat([in_lst[s], d_up], 2)
                out_ = net(in_)
                d_ = d_up + gauss2(out_.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
                # d_ = d_up + out_

                d_up_rev = F.grid_sample(d_rev.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1,
                                                                                                                        2,
                                                                                                                        0)
                in_rev = torch.cat([in_rev_lst[s], d_up_rev], 2)
                out_rev = net(in_rev)
                d_rev = d_up_rev + gauss2(out_rev.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
            # d_rev = d_up_rev + out_rev
            if R4K:
                # in_ = torch.cat([in_lst[s+2],d_],2)
                # out_ = net(in_)
                # k1 = F.grid_sample(out_.permute(2,0,1).unsqueeze(0),xy_lst[s+1].unsqueeze(0)).squeeze().permute(1,2,0)
                # k11 = F.grid_sample(out_.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # d_up = F.grid_sample(d_.permute(2,0,1).unsqueeze(0),xy_lst[s+1].unsqueeze(0)).squeeze().permute(1,2,0)
                # in_ = torch.cat([in_lst[s+1],d_up+(1.0/3.0)*k1],2)
                # out_ = net(in_)
                # k2 = out_
                # k21 = F.grid_sample(out_.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # in_ = torch.cat([in_lst[s+1],d_up-(1.0/3.0)*k1+k2],2)
                # k3 = F.grid_sample(net(in_).permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # d_up = F.grid_sample(d_.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                # in_ = torch.cat([in_lst[s],d_up+k11-k21+k3],2)
                # k4 = net(in_)
                #
                # out_ = (1./8.)*(k11+3.0*k21+3.0*k3+k4)
                # d_ = d_up + gauss2(out_.permute(2,0,1).unsqueeze(0)).squeeze().permute(1,2,0)
                # #d_ = d_up + out_
                #
                # in_rev = torch.cat([in_rev_lst[s+2],d_rev],2)
                # out_rev = net(in_rev)
                # k1_rev = F.grid_sample(out_rev.permute(2,0,1).unsqueeze(0),xy_lst[s+1].unsqueeze(0)).squeeze().permute(1,2,0)
                # k11_rev = F.grid_sample(out_rev.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # d_up_rev = F.grid_sample(d_rev.permute(2,0,1).unsqueeze(0),xy_lst[s+1].unsqueeze(0)).squeeze().permute(1,2,0)
                # in_rev = torch.cat([in_rev_lst[s+1],d_up_rev+(1.0/3.0)*k1_rev],2)
                # out_rev = net(in_rev)
                # k2_rev = out_rev
                # k21_rev = F.grid_sample(out_rev.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # in_rev = torch.cat([in_rev_lst[s+1],d_up_rev-(1.0/3.0)*k1_rev+k2_rev],2)
                # k3_rev = F.grid_sample(net(in_rev).permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                #
                # d_up_rev = F.grid_sample(d_rev.permute(2,0,1).unsqueeze(0),xy_lst[s].unsqueeze(0)).squeeze().permute(1,2,0)
                # in_rev = torch.cat([in_rev_lst[s],d_up_rev+k11_rev-k21_rev+k3_rev],2)
                # k4_rev = net(in_rev)
                #
                # out_rev = (1./8.)*(k11_rev+3.0*k21_rev+3.0*k3_rev+k4_rev)
                # d_rev = d_up_rev + gauss2(out_rev.permute(2,0,1).unsqueeze(0)).squeeze().permute(1,2,0)
                in_ = torch.cat([in_lst[s + 2], d_], 2)
                out_ = net(in_)
                k1 = F.grid_sample(out_.permute(2, 0, 1).unsqueeze(0), xy_lst[s + 1].unsqueeze(0)).squeeze().permute(1,
                                                                                                                     2,
                                                                                                                     0)
                k11 = F.grid_sample(out_.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2,
                                                                                                                  0)

                d_up = F.grid_sample(d_.permute(2, 0, 1).unsqueeze(0), xy_lst[s + 1].unsqueeze(0)).squeeze().permute(1,
                                                                                                                     2,
                                                                                                                     0)
                in_ = torch.cat([in_lst[s + 1], d_up + (1.0 / 2.0) * k1], 2)
                out_ = net(in_)
                k2 = out_
                k21 = F.grid_sample(out_.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2,
                                                                                                                  0)

                in_ = torch.cat([in_lst[s + 1], d_up + (1.0 / 2.0) * k2], 2)
                k3 = F.grid_sample(net(in_).permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1,
                                                                                                                     2,
                                                                                                                     0)

                d_up = F.grid_sample(d_.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2,
                                                                                                                 0)
                in_ = torch.cat([in_lst[s], d_up + k3], 2)
                k4 = net(in_)

                out_ = (1. / 6.) * (k11 + 2.0 * k21 + 2.0 * k3 + k4)
                d_ = d_up + gauss2(out_.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
                # d_ = d_up + out_

                in_rev = torch.cat([in_rev_lst[s + 2], d_rev], 2)
                out_rev = net(in_rev)
                k1_rev = F.grid_sample(out_rev.permute(2, 0, 1).unsqueeze(0),
                                       xy_lst[s + 1].unsqueeze(0)).squeeze().permute(1, 2, 0)
                k11_rev = F.grid_sample(out_rev.permute(2, 0, 1).unsqueeze(0),
                                        xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2, 0)

                d_up_rev = F.grid_sample(d_rev.permute(2, 0, 1).unsqueeze(0),
                                         xy_lst[s + 1].unsqueeze(0)).squeeze().permute(1, 2, 0)
                in_rev = torch.cat([in_rev_lst[s + 1], d_up_rev + (1.0 / 2.0) * k1_rev], 2)
                out_rev = net(in_rev)
                k2_rev = out_rev
                k21_rev = F.grid_sample(out_rev.permute(2, 0, 1).unsqueeze(0),
                                        xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2, 0)

                in_rev = torch.cat([in_rev_lst[s + 1], d_up_rev + (1.0 / 2.0) * k2_rev], 2)
                k3_rev = F.grid_sample(net(in_rev).permute(2, 0, 1).unsqueeze(0),
                                       xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2, 0)

                d_up_rev = F.grid_sample(d_rev.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(
                    1, 2, 0)
                in_rev = torch.cat([in_rev_lst[s], d_up_rev + k3_rev], 2)
                k4_rev = net(in_rev)

                out_rev = (1. / 6.) * (k11_rev + 2.0 * k21_rev + 2.0 * k3_rev + k4_rev)
                d_rev = d_up_rev + gauss2(out_rev.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
            #d_rev = d_up_rev + out_rev

        if train:
            # ind_ = torch.randperm(h_lst[s]*w_lst[s])[0:int(0.75*h_lst[s]*w_lst[s])].to(device)
            if nChannel > 1:
                Jw_ = F.grid_sample(J_lst[s].unsqueeze(0), (xy_lst[s] + d_).unsqueeze(0),
                                    padding_mode='reflection').squeeze()
                Iw_ = F.grid_sample(I_lst[s].unsqueeze(0), (xy_lst[s] + d_rev).unsqueeze(0),
                                    padding_mode='reflection').squeeze()
                for ch in range(nChannel):
                    # loss = loss + (1./(nChannel*L))*F.mse_loss(gauss1(Jw_[ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_], gauss1(I_lst[s][ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_])
                    loss = loss + (1. / (nChannel * L)) * F.mse_loss(
                        gauss1(Jw_[ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]),
                        gauss1(I_lst[s][ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]))
                    loss = loss + (1. / (nChannel * L)) * F.mse_loss(
                        gauss1(Iw_[ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]),
                        gauss1(J_lst[s][ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]))
            else:
                Jw_ = F.grid_sample(J_lst[s].unsqueeze(0).unsqueeze(0), (xy_lst[s] + d_).unsqueeze(0),
                                    padding_mode='reflection').squeeze()
                Iw_ = F.grid_sample(I_lst[s].unsqueeze(0).unsqueeze(0), (xy_lst[s] + d_rev).unsqueeze(0),
                                    padding_mode='reflection').squeeze()
                d_revw_ = F.grid_sample(d_rev.permute(2, 0, 1).unsqueeze(0), (xy_lst[s] + d_).unsqueeze(0),
                                        padding_mode='reflection').squeeze().permute(1, 2, 0)
                dw_ = F.grid_sample(d_.permute(2, 0, 1).unsqueeze(0), (xy_lst[s] + d_rev).unsqueeze(0),
                                    padding_mode='reflection').squeeze().permute(1, 2, 0)

                if MI:
                    mii = mine_net(torch.stack([I_lst[s], Jw_], 2), ind_lst[s])
                    mij = mine_net(torch.stack([J_lst[s], Iw_], 2), ind_lst[s])
                if SSIM:
                    Jw_exp = Jw_.unsqueeze(-1)
                    Jw_exp = torch.transpose(Jw_exp, 0, 2)
                    Jw_exp = torch.transpose(Jw_exp, 1, 2)
                    Jw_exp = Jw_exp.unsqueeze(0)
                    I_lst_exp = I_lst[s].unsqueeze(-1)
                    I_lst_exp = torch.transpose(I_lst_exp, 0, 2)
                    I_lst_exp = torch.transpose(I_lst_exp, 1, 2)
                    I_lst_exp = I_lst_exp.unsqueeze(0)

                    Iw_exp = Iw_.unsqueeze(-1)
                    Iw_exp = torch.transpose(Iw_exp, 0, 2)
                    Iw_exp = torch.transpose(Iw_exp, 1, 2)
                    Iw_exp = Iw_exp.unsqueeze(0)
                    J_lst_exp = J_lst[s].unsqueeze(-1)
                    J_lst_exp = torch.transpose(J_lst_exp, 0, 2)
                    J_lst_exp = torch.transpose(J_lst_exp, 1, 2)
                    J_lst_exp = J_lst_exp.unsqueeze(0)

                # MI +SSIM (R4K)
                if MI and SSIM and R4K:
                    loss = loss -  (2./(L+1)) * mii + (1 - ssim(J_lst_exp, Iw_exp))
                    loss = loss -  (2./(L+1))* mij + (1 - ssim(I_lst_exp, Jw_exp))
                if MI and SSIM and Euler:
                    loss = loss -  (1./(L)) * mii + (1 - ssim(J_lst_exp, Iw_exp))
                    loss = loss -  (1./(L))* mij + (1 - ssim(I_lst_exp, Jw_exp))

                if SSIM and R4K:
                    loss = loss + (2./(L+1)) * (1 - ssim(J_lst_exp, Iw_exp))
                    loss = loss + (2./(L+1)) * (1 - ssim(I_lst_exp, Jw_exp))
                if SSIM and Euler:
                    loss = loss + (1./(L)) * (1 - ssim(J_lst_exp, Iw_exp))
                    loss = loss + (1./(L)) * (1 - ssim(I_lst_exp, Jw_exp))

                ## loss = loss + (1./L)*F.mse_loss(gauss1(Jw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_], gauss1(I_lst[s].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_])
                if MSE and R4K:
                    loss = loss + (2./(L+1))*F.mse_loss(gauss1(Jw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]]), gauss1(I_lst[s].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]]))
                    loss = loss + (2./(L+1))*F.mse_loss(gauss1(Iw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]]), gauss1(J_lst[s].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]]))
                if MSE and Euler:
                    loss = loss + (1. / (L)) * F.mse_loss(gauss1(Jw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]),
                                                          gauss1(I_lst[s].unsqueeze(0).unsqueeze(0)).view(
                                                              [h_lst[s] * w_lst[s]]))
                    loss = loss + (1. / (L)) * F.mse_loss(gauss1(Iw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]]),
                                                          gauss1(J_lst[s].unsqueeze(0).unsqueeze(0)).view(
                                                              [h_lst[s] * w_lst[s]]))

                # if Euler:
                #     loss = loss + (2. / (L+1)) * F.mse_loss(dw_, -d_rev)  # consistency constraint
                #     loss = loss + (2. / (L+1)) * F.mse_loss(d_revw_, -d_)  # consistency constraint
                # if R4K:
                #     loss = loss +  (2./(L+1))*F.mse_loss(dw_,-d_rev) # consistency constraint
                #     loss = loss +  (2./(L+1))*F.mse_loss(d_revw_,-d_) # consistency constraint

            # regularization of the displacement vector
            if Euler:
                loss = loss + lam * (1. / (L)) * torch.mean(d_ ** 2) + lam * (1. / (L)) * torch.mean(d_rev ** 2)
            if R4K:
                loss = loss + lam * (2. / (L + 1)) * torch.mean(d_ ** 2) + lam * (2. / (L + 1)) * torch.mean(d_rev ** 2)

    return d_, d_rev, loss


def run_code(I,J,I_label, J_label, file_path_img,slice_,pixdim, n_max_sun=0,n_min_sun=0,ifplot = False):

    if Method == 'Left':
        slice_='left'
    start = time.time()
    if np.ndim(I) == 3:
        nChannel = I.shape[2]

        pyramid_I = tuple(
            pyramid_gaussian(gaussian(I, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
        pyramid_J = tuple(
            pyramid_gaussian(gaussian(J, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
    elif np.ndim(I) == 2:
        nChannel = 1
        # pyramid_I = tuple(pyramid_gaussian(I,max_layer=L, multichannel=False))
        # pyramid_J = tuple(pyramid_gaussian(J,max_layer=L,  multichannel=False))
        pyramid_I = tuple(
            pyramid_gaussian(gaussian(I, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
        pyramid_J = tuple(
            pyramid_gaussian(gaussian(J, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
    else:
        print("Unknown rank for an image")
        ifplot = False

    # if ifplot:
    #     # % matplotlib
    #     # inline
    #     fig = plt.figure()
    #     fig.add_subplot(2, 2, 1)
    #     plt.imshow(I)
    #     plt.title("Fixed Image")
    #     fig.add_subplot(2, 2, 2)
    #     plt.imshow(J)
    #     plt.title("Moving Image")
    #     fig.add_subplot(2, 2, 3)
    #     plt.imshow(I_label)
    #     plt.title("Fixed label")
    #     fig.add_subplot(2, 2, 4)
    #     plt.imshow(J_label)
    #     plt.title("Moving label")
    #     # plt.show()
    # print('#0 done')

    # create a list of necessary objects you will need and commit to GPU
    I_lst, J_lst, h_lst, w_lst, xy_lst, in_lst, in_rev_lst, ind_lst = [], [], [], [], [], [], [], []
    for s in range(L):
        I_, J_ = torch.tensor(pyramid_I[s].astype(np.float32)).to(device), torch.tensor(
            pyramid_J[s].astype(np.float32)).to(device)
        if nChannel > 1:
            I_lst.append(I_.permute(2, 0, 1))
            J_lst.append(J_.permute(2, 0, 1))
            h_, w_ = I_lst[s].shape[1], I_lst[s].shape[2]
        else:
            I_lst.append(I_)
            J_lst.append(J_)
            h_, w_ = I_lst[s].shape[0], I_lst[s].shape[1]

        # print(h_, w_)
        h_lst.append(h_)
        w_lst.append(w_)
        ind_ = torch.randperm(int(h_ * w_ * sampling))  # torch.arange(int(h_*w_))#
        ind_lst.append(ind_)
        y_, x_ = torch.meshgrid([torch.arange(0, h_).float().to(device), torch.arange(0, w_).float().to(device)])
        y_, x_ = 2.0 * y_ / (h_ - 1) - 1.0, 2.0 * x_ / (w_ - 1) - 1.0
        xy_ = torch.stack([x_, y_], 2)
        xy_lst.append(xy_)

        curr, prev = np.power(downscale, -s), np.power(downscale, -np.minimum(s + 1, L - 1))
        # print(prev, curr)
        in_ = torch.stack([torch.ones(h_, w_).to(device), x_, y_, prev * torch.ones(h_, w_).to(device),
                           curr * torch.ones(h_, w_).to(device)], 2)
        in_rev_ = torch.stack([-torch.ones(h_, w_).to(device), x_, y_, prev * torch.ones(h_, w_).to(device),
                               curr * torch.ones(h_, w_).to(device)], 2)
        # in_ = torch.stack([I_-J_,prev*torch.ones(h_,w_).to(device),curr*torch.ones(h_,w_).to(device)],2)
        # in_ = torch.stack([I_-J_],2)
        in_lst.append(in_)
        in_rev_lst.append(in_rev_)

    # 4
    gauss2 = GaussianFilter(2, 3.0).to(device)
    gauss1 = GaussianFilter(1, 1.5).to(device)








    # 5
    net = Net().to(device)
    mine_net = MINE(nChannel).to(device)
    # optimizer = torch.optim.LBFGS(net.parameters(),lr=1e-2) # not much luck with LBFGS

    #     learning_rate=0.001,
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-07,
    # amsgrad=False,
    # name='Adam',betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    if MI:
        optimizer = optim.Adam(
            [{'params': mine_net.parameters(), 'lr': 1e-3}, {'params': net.parameters(), 'lr': 5.0 * 1e-4}],
            amsgrad=False)
    else:
        optimizer = optim.Adam(net.parameters(), lr=5.0 * 1e-4, amsgrad=True)
    mi_list = []
    param_group = optimizer.param_groups
    # Loop


    thresh_min = 0.00011
    prev_loss = 1

    iterationnum = maxiter
    for itr in range(iterationnum):
        optimizer.zero_grad()
        d_,d_rev,loss =  deformation(net,mine_net, gauss1,gauss2,in_lst,h_lst,w_lst,xy_lst,J_lst,nChannel,I_lst, ind_lst,in_rev_lst, True)# deformation(net, True)

        # mi_list.append(-loss.item())
        loss.backward()
        # clear_output(wait=True)
        # plt.plot(mi_list)
        # plt.title("MI")
        # plt.show()
        optimizer.step()
        param_group = optimizer.param_groups

        if itr % 100 == 0:
            # print("learning: ", float(param_group[0]['lr']))
            # print("learning: ", float(param_group[1]['lr']))
            # print("Itr:", itr, "loss:", -loss.item())
            curr_loss = loss.item()
            # print('abs :', curr_loss, prev_loss)
            if curr_loss >= prev_loss and itr != 0:
                # print('break')
                break
            prev_loss = loss.item()
            # if itr%300 == 0 and itr!= 0 :
            #   oldlr=float(param_group[0]['lr'])
            #   lr =oldlr* 0.25
            #   param_group[0]['lr'] = lr
            #   oldlr=float(param_group[1]['lr'])
            #   lr =oldlr* 0.25
            #   param_group[1]['lr'] = lr
    print('time :' , time.time()-start)
    d_,d_rev,loss  = deformation(net,mine_net, gauss1,gauss2,in_lst,h_lst,w_lst,xy_lst,J_lst,nChannel,I_lst, ind_lst,in_rev_lst, False)#(net, False)
    # print(d_)
    if nChannel > 1:
        Jw_ = F.grid_sample(J_lst[0].unsqueeze(0), (xy_lst[0] + d_).unsqueeze(0), padding_mode='reflection').squeeze()
    else:
        Jw_ = F.grid_sample(J_lst[0].unsqueeze(0).unsqueeze(0), (xy_lst[0] + d_).unsqueeze(0),
                            padding_mode='reflection').squeeze()
        Iw_ = F.grid_sample(I_lst[0].unsqueeze(0).unsqueeze(0), (xy_lst[0] + d_rev).unsqueeze(0),
                            padding_mode='reflection').squeeze()
        Jw_l = F.grid_sample((torch.Tensor(J_label).to(device)).unsqueeze(0).unsqueeze(0),
                             (xy_lst[0] + d_).unsqueeze(0), padding_mode='reflection').squeeze()

        Iw_l = F.grid_sample((torch.Tensor(I_label).to(device)).unsqueeze(0).unsqueeze(0),
                             (xy_lst[0] + d_rev).unsqueeze(0), padding_mode='reflection').squeeze()




    if True:
        # save deformation field
        if Method == 'Left':
            file_name = file_path_img.split('/')[-2]
            # print(file_name)
            nummoving = movingimg_filename.split('_')[-2][8:]
            numfixed = fixedimg_filename.split('_')[-2][8:]

            image_name = movingimg_filename.split('_')[0]
            df_path = "/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration_R4K/Results/MICCAI2020/" + Method + "/Images/" + image_name + '_' + str(
                L) + '_' + str(maxiter)
        elif Method=='ACDC':
            file_name = file_path_img.split('/')[-2]
            # print(file_name)
            # nummoving = movingimg_filename.split('_')[-2][8:]
            # numfixed = fixedimg_filename.split('_')[-2][8:]

            # image_name = movingimg_filename.split('_')[0]
            df_path = "/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration_R4K/Results/MICCAI2020/" + Method + "/Images/" + file_name + '_' +loss_str+'_' + str(
                L) + '_' + str(maxiter)
        elif Method=='Sunny':

            file_name = file_path_img.split('/')[-1]
            image_name = file_name

            nummoving = n_min_sun
            numfixed = n_max_sun
            df_path = "/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration_R4K/Results/MICCAI2020/" + Method + "/Images/" + 'Data_' + loss_str + '_' + str(
                L) + '_' + str(maxiter)

        # fig = plt.figure()
        # fig.add_subplot(3, 2, 1)
        # if nChannel > 1:
        #     plt.imshow((J_lst[0] - I_lst[0]).permute(1, 2, 0).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((J_lst[0]-I_lst[0]).cpu().data)
        #     plt.imshow((I_lst[0]).cpu().data, cmap='gray')
        # plt.title("I")
        #
        # fig.add_subplot(3, 2, 2)
        # if nChannel > 1:
        #     plt.imshow((Jw_ - I_lst[0]).permute(1, 2, 0).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((Jw_-I_lst[0]).cpu().data)
        #     plt.imshow((J_lst[0]).cpu().data, cmap='gray')
        # plt.title("J")
        #
        # fig.add_subplot(3, 2, 3)
        # if nChannel > 1:
        #     plt.imshow((J_lst[0] - I_lst[0]).permute(1, 2, 0).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((J_lst[0]-I_lst[0]).cpu().data)
        #     plt.imshow((Jw_).cpu().data, cmap='gray')
        # plt.title("Warped J")
        #
        # fig.add_subplot(3, 2, 4)
        # if nChannel > 1:
        #     plt.imshow((Jw_ - I_lst[0]).permute(1, 2, 0).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((Jw_-I_lst[0]).cpu().data)
        #     plt.imshow((Iw_).cpu().data, cmap='gray')
        # plt.title("Warped I")
        # # labels
        # fig.add_subplot(3, 2, 5)
        # if nChannel > 1:
        #     plt.imshow((Jw_l - torch.Tensor(I_label).to(device)).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((Jw_-I_lst[0]).cpu().data)
        #     plt.imshow((Jw_l - torch.Tensor(I_label).to(device)).cpu().data, cmap='gray')
        # plt.title("Warped I")
        #
        # fig.add_subplot(3, 2, 6)
        # if nChannel > 1:
        #     plt.imshow((Iw_l - torch.Tensor(J_label).to(device)).cpu().data, cmap='gray')
        # else:
        #     # plt.imshow((Jw_-I_lst[0]).cpu().data)
        #     plt.imshow((Iw_l - torch.Tensor(J_label).to(device)).cpu().data, cmap='gray')
        # plt.title("Warped I")
        #
        # if Is_ACDC == False:
        #     plt.savefig(df_path + str(maxiter)+'_L_'+str(L)+'_'+'_all.png')
        # else:
        #     plt.savefig(df_path +  '_'+str(maxiter)+'_L_'+str(L)+'_'+'_all.png')



        #for paper I saved all images seperately

        plt.imshow((I_lst[0]).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' + image_name + '_' + str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) + '_I.png')



        plt.imshow((J_lst[0]).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' +image_name + '_' + str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) + '_J.png')


        plt.imshow((Jw_).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' + image_name + '_' +str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) +  '_WarpedJ.png')


        plt.imshow((Iw_).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' + image_name + '_' +str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) +  '_WarpedI.png')


        plt.imshow((Jw_l - torch.Tensor(I_label).to(device)).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' +image_name + '_' + str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) +  '_diffji.png')


        plt.imshow((Iw_l - torch.Tensor(J_label).to(device)).cpu().data, cmap='gray')
        plt.savefig(df_path + '_' + image_name + '_' +str(maxiter) + '_L_' + str(L) + '_slice_'+ str(slice_) +  '_diffij.png')

        # if Is_ACDC == False:
        #     plt.savefig(df_path + str(maxiter)+'_L_'+str(L)+'_'+'_all.png')
        # else:
        #     plt.savefig(df_path +  '_'+str(maxiter)+'_L_'+str(L)+'_'+'_all.png')









        np.save(df_path, d_.cpu().data.numpy())
        # print(Jw_l.cpu().data.numpy().shape[0], torch.Tensor(I_label).to(device).shape[0])
        if torch.Tensor(I_label).to(device).shape[0] == Jw_l.cpu().data.numpy().shape[0] and (Jw_l.cpu().data.numpy().max() != 0 and float(I_label.max()) != 0) :
            print(Jw_l.cpu().data.numpy().max())
            print('forward')

            dice_after = "%.6f" % (np.sum(Jw_l.cpu().data.numpy()[I_label == 1]) * 2.0 / (
                    np.sum(Jw_l.cpu().data.numpy()) + np.sum(I_label)))

            dice_before = "%.6f" % (np.sum(J_label[I_label == 1]) * 2.0 / (
                    np.sum(J_label) + np.sum(I_label)))

            print("unregiterde Dice: ", dice_before)
            print("regiterde Dice: ", dice_after)

            # HD
            contours_target = skimage.measure.find_contours(I_label, 0.5)
            contor_target_0 = HD_to_mmHD(contours_target[0],pixdim)

            cnt = skimage.measure.find_contours(Jw_l.cpu().data.numpy(), 0.5)
            cnt_0 = HD_to_mmHD(cnt[0],pixdim)
            HD1 = directed_hausdorff(contor_target_0, cnt_0)

            cnt_before = skimage.measure.find_contours(J_label, 0.5)
            cnt_before_0 =  HD_to_mmHD(cnt_before[0],pixdim)
            HD_before = directed_hausdorff(contor_target_0, cnt_before_0)

            print("registered HD: ", HD1[0])
            print("unregistered HD: ", HD_before[0])

            # loss_before = F.mse_loss(J_lst[0], I_lst[0])
            # loss_after = F.mse_loss(Jw_, I_lst[0])
            # print(loss_before.item(), loss_after.item())

            print('backward')

            dice_beforeb = "%.6f" % (np.sum(I_label[J_label == 1]) * 2.0 / (
                    np.sum(I_label) + np.sum(J_label)))

            print("unregiterde Dice: ", dice_beforeb)
            contours_targetb = skimage.measure.find_contours(I_label, 0.5)
            contours_targetb_0= HD_to_mmHD(contours_targetb[0], pixdim)

            cnt_beforeb = skimage.measure.find_contours(I_label, 0.5)
            cnt_beforeb_0= HD_to_mmHD(cnt_beforeb[0], pixdim)

            HD_beforeb = directed_hausdorff(contours_targetb_0, cnt_beforeb_0)
            print("unregistered HD: ", HD_beforeb[0])

            dice_afterb = "%.6f" % (np.sum(Iw_l.cpu().data.numpy()[J_label == 1]) * 2.0 / (
                    np.sum(Iw_l.cpu().data.numpy()) + np.sum(J_label)))
            print("regiterde Dice: ", dice_afterb)

            # HD
            contours_targetb = skimage.measure.find_contours(J_label, 0.5)
            contours_targetb_0 = HD_to_mmHD(contours_targetb[0], pixdim)

            cntb = skimage.measure.find_contours(Iw_l.cpu().data.numpy(), 0.5)
            cntb_0 = HD_to_mmHD(cntb[0], pixdim)
            HD1b = directed_hausdorff(contours_targetb_0,cntb_0)
            print("registered HD: ", HD1b[0])


            if Method =='Left':
                filename = df_path+ str(maxiter)+'_L_'+str(L)+'_'+loss_str+'twoway.txt'
                with open(filename, 'a') as the_file:
                    the_file.write(image_name + '_' + file_name + '_' + nummoving + '_to_' + numfixed + '\n')
                    the_file.write('Unregistered Dice: ' + str(dice_before) + '  \n')
                    the_file.write('registered Dice: ' + str(dice_after) + '  \n')
                    the_file.write('Unregistered HD: ' + str(HD_before[0]) + '  \n')
                    the_file.write('registered HD: ' + str(HD1[0]) + '  \n')
                    the_file.write('registered BDice: ' + str(dice_afterb) + '  \n')
                    the_file.write('registered BHDB: ' +  str(HD1b[0]) + '  \n')
                    the_file.write('  \n')
            elif Method == 'ACDC':
                filename = '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration_R4K/Results/MICCAI2020/ACDC/resultsfile_ACDC_'+ str(maxiter)+'_L_'+str(L)+'_'+loss_str+'twoway.txt'
                with open(filename, 'a') as the_file:
                    the_file.write(file_name + '_' + str(slice_) + '\n')
                    the_file.write('Unregistered Dice: ' + str(dice_before) + '  \n')
                    the_file.write('registered Dice: ' + str(dice_after) + '  \n')
                    the_file.write('Unregistered HD: ' + str(HD_before[0]) + '  \n')
                    the_file.write('registered HD: ' + str(HD1[0]) + '  \n')
                    the_file.write('registered BDice: ' + str(dice_afterb) + '  \n')
                    the_file.write('registered BHD: ' + str(HD1b[0]) + '  \n')
                    the_file.write('  \n')
            elif Method == 'Sunny':

                filename = df_path + str(maxiter) + '_L_' + str(L) + '_' + loss_str + 'twoway.txt'
                with open(filename, 'a') as the_file:
                    the_file.write( file_name + '_slice_' +str(slice_) + str(nummoving) + '_to_' + str(numfixed )+ '\n')
                    the_file.write('Unregistered Dice: ' + str(dice_before) + '  \n')
                    the_file.write('registered Dice: ' + str(dice_after) + '  \n')
                    the_file.write('Unregistered HD: ' + str(HD_before[0]) + '  \n')
                    the_file.write('registered HD: ' + str(HD1[0]) + '  \n')
                    the_file.write('registered BDice: ' + str(dice_afterb) + '  \n')
                    the_file.write('registered BHDB: ' + str(HD1b[0]) + '  \n')
                    the_file.write('  \n')

            df = 2
            fig = plt.figure(figsize=(5, 5))
            plt.quiver(d_.cpu().data[::df, ::df, 0], d_.cpu().data[::df, ::df, 1], color='r')
            plt.axis('equal')

            # if Is_ACDC == False:
            #     plt.savefig(df_path + str(maxiter)+'_L_'+str(L)+'_'+'optical.png')
            # else:
            plt.savefig(df_path +'_slice_'+str(slice_)+'_'+str(maxiter)+'_L_'+str(L)+'_'+ 'optical.png')

            down_factor = 0.5
            h_resize = int(down_factor * h_lst[0])
            w_resize = int(down_factor * w_lst[0])
            grid_x = resize(xy_lst[0].cpu()[:, :, 0].squeeze().numpy(), (h_resize, w_resize))
            grid_y = resize(xy_lst[0].cpu()[:, :, 1].squeeze().numpy(), (h_resize, w_resize))
            distx = resize((xy_lst[0] + d_).cpu()[:, :, 0].squeeze().detach().numpy(), (h_resize, w_resize))
            disty = resize((xy_lst[0] + d_).cpu()[:, :, 1].squeeze().detach().numpy(), (h_resize, w_resize))


            fig, ax = plt.subplots()
            plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
            plot_grid(distx, disty, ax=ax, color="C0")
            # if Is_ACDC == False:
            #     plt.savefig(df_path + str(maxiter)+'_L_'+str(L)+'_'+'forward.png')
            #     # plt.show()
            # else:
            plt.savefig(df_path+'_slice_'+str(slice_)+'_'+str(maxiter)+'_L_'+str(L)+'_'+'_forward.png')

            distx = resize((xy_lst[0] + d_rev).cpu()[:, :, 0].squeeze().detach().numpy(), (h_resize, w_resize))
            disty = resize((xy_lst[0] + d_rev).cpu()[:, :, 1].squeeze().detach().numpy(), (h_resize, w_resize))

            fig, ax = plt.subplots()
            plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
            plot_grid(distx, disty, ax=ax, color="C0")
            # if Is_ACDC == False:
            #     plt.savefig(df_path +str(maxiter)+'_L_'+str(L)+'_'+ '_back.png')
            #     # plt.show()
            # else:
            plt.savefig(df_path+'_slice_'+str(slice_)+'_'+str(maxiter)+'_L_'+str(L)+'_back.png')
            df_p1 = df_path +'_'+file_name +'_'+ str(slice_)+'_grid.png'
            df_pj = df_path +'_'+file_name +'_'+ str(slice_)+ '_contj.png'
            df_pi = df_path +'_'+file_name +'_'+ str(slice_)+ '_conti.png'
            df_p3 = df_path +'_'+file_name +'_'+ str(slice_)+ '_gridb.png'
            show_grid(J,I, d_rev,d_, Jw_l.cpu().data, Iw_l.cpu().data, I_label,xy_lst,h_lst,w_lst,df_p1, df_pj,df_pi,df_p3)

       # reset - f except variables(file_path_img, file_path_lb, g, a)
        # sys.exit()
        # %reset_selective -f Jw_l, Jw_, xy_,xy_lst,I,I_label,J,d,loss,x_,y_,net, J_label,I_lst,J_lst,Jw_l, name_of_frame_max,name_of_frame_min,pyramid_I,pyramid_J, h_resize,w_resize,grid_x,rid_y,distx,disty





if Method == 'ACDC':
    ACDC_PATH = '/home/ameneh/UofA/Datasets/ACDC/Diagnosis/training'
    for dir in sorted(os.listdir(ACDC_PATH)):
        if 'patient' in dir:
            patient_num = dir[7:10]
            if (int(patient_num)<1):
                # pass
                test_t = False
                continue
            #
            # else:
            #     test_t = True
            for file in os.listdir(os.path.join(ACDC_PATH, dir)):
                if 'gt' in file:
                    index_s = file.find('frame')
                    index_e = file.find('_gt')
                    if index_e == -1 or index_s == -1:
                        continue
                    num_frame = file[index_s + 5:-(len(file) - index_e)]
                    num_frame = (int(num_frame) * 10) // 10

                    # first gt has a larger shape and it is in the first part of sequence
                    if (num_frame - 0) < np.abs(num_frame - 10):
                        fgt_num = num_frame
                        f_gt_p = os.path.join(ACDC_PATH, dir, file)
                        # f_gt = nibabel.load(f_gt_p).get_data()

                    else:
                        egt_num = num_frame
                        e_gt_p = os.path.join(ACDC_PATH, dir, file)
                        # e_gt = nibabel.load(e_gt_p).get_data()
                        # e_gt = np.squeeze(e_gt)

                elif '4d' in file:
                    patient_num = file[8:10]
                    patient_num = (int(patient_num) * 100) // 100
                    test_t = True
                    Patient_path = os.path.join(ACDC_PATH, dir, file)
                    image_mr = nibabel.load(os.path.join(ACDC_PATH, dir, file))
                    images_4d = image_mr.get_fdata()
                    dim = [images_4d.shape[0], images_4d.shape[1]]
                    cardiacnumofimages = images_4d.shape[3]
                    pat_slice = images_4d.shape[2]
                    print('Patient : ' , patient_num, 'slice ' , pat_slice)
            # if test_t == False:
            #     print(patient_num, "break")
            #     continue
            for slice_num in range(0, pat_slice):

                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                np.random.seed(seed)  # Numpy module.
                random.seed(seed)  # Python random module.
                torch.manual_seed(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True



                print('slice = ', slice_num)
                I,J,I_label,J_label, No_gt ,pixdim = ACDC_data(file_path_img = Patient_path,file_I_path_lb = f_gt_p,file_J_path_lb = e_gt_p,slice_ = slice_num, fgt = fgt_num, lgt =  egt_num)
                if No_gt==False:

                    run_code(I,J,np.squeeze(I_label),np.squeeze(J_label),Patient_path,slice_num+1,pixdim)#ahmed used +1

                    plt.close('all')

elif Method == 'Left':
    for ch_name in ch_list:
        file_path_img = "/home/ameneh/UofA/Datasets/left_atrium_deep_learning/images/" + ch_name + "/"
        file_path_lb = "/home/ameneh/UofA/Datasets/left_atrium_deep_learning/labels/" + ch_name + "/"
        for filename in glob.glob(file_path_lb + '*label.npy'):

            if 'P074' in filename.split('/')[-1].split('_')[
                0]:  # or 'P015' in filename.split('/')[-1].split('_')[0] or 'P049' in filename.split('/')[-1].split('_')[0]:
                patient_list.append(filename.split('/')[-1].split('_')[0])

        patient_list = list(set(
            patient_list))  # to remove reapeted name from list we can convert it to the set and convert it back to the list
        # TODO: check globals() function
        g = globals()  # use this function to create dynamic list with unfixed name
        for i in range(len(patient_list)):
            g['ch_list_{}'.format(patient_list[i])] = []  # generate len(a) empty list

        for filename in glob.glob(file_path_lb + '*label.npy'):
            if filename.split('/')[-1].split('_')[0] in patient_list:
                # print(filename.split('/')[-1].split('_')[0])
                g['ch_list_{}'.format(filename.split('/')[-1].split('_')[0])].append(filename)

        for i in range(len(patient_list)):
            g['ch_list_{}'.format(patient_list[i])].sort(
                key=lambda x: x.split('/')[-1].split('_')[-2].split('-')[-1])  # sort with a function
            # print(g['ch_list_{}'.format(patient_list[i])])
            maxpix = 0
            minpix = 100000
            for j in range(len(g['ch_list_{}'.format(patient_list[i])])):
                # print(g['ch_list_{}'.format(patient_list[i])])
                numberofpix = np.sum(np.load(g['ch_list_{}'.format(patient_list[i])][j]) == 255)
                if numberofpix > maxpix:
                    maxpix = numberofpix
                    name_of_frame_max = g['ch_list_{}'.format(patient_list[i])][j].split('/')[-1]
                if numberofpix < minpix:
                    minpix = numberofpix
                    name_of_frame_min = g['ch_list_{}'.format(patient_list[i])][j].split('/')[-1]




            fixedimg_filename = name_of_frame_max[:-9] + "image.npy"
            print(fixedimg_filename)
            fixedlb_filename = name_of_frame_max
            movingimg_filename = name_of_frame_min[:-14] + "0001_image.npy"
            # print('min',movingimg_filename)
            movinglb_filename = name_of_frame_min[:-14] + "0001_label.npy"
            # print('min',movinglb_filename)
            # print('************************************')
            Patient_path = file_path_img + fixedimg_filename
            I = np.load(file_path_img + fixedimg_filename)
            print(file_path_img + fixedimg_filename)
            print(file_path_img + movingimg_filename)
            print(file_path_lb + fixedlb_filename)
            print(file_path_lb + movinglb_filename)
            I_label = np.load(file_path_lb + fixedlb_filename)
            J = np.load(file_path_img + movingimg_filename)
            J_label = np.load(file_path_lb + movinglb_filename)
            I_label = I_label / 255.
            J_label = J_label / 255.
            print('left')
            pixdim =[1,1]
            run_code(I, J, np.squeeze(I_label), np.squeeze(J_label), file_path_img, 1, pixdim)
            # run_code(I, J, np.squeeze(I_label), np.squeeze(J_label), Patient_path,0)  # ahmed used +1
            plt.close('all')
elif Method == 'Sunny':
    GT_Sunny_Path = r'/home/ameneh/UofA/Datasets/Sunnybrock/challenge_training/challenge_training_croppedanssuply/correct_data/Contours/'
    Suuny_PATH = '/home/ameneh/UofA/Datasets/Sunnybrock/challenge_training/challenge_training_croppedanssuply/correct_data/Data/'
    for dir in sorted(os.listdir(Suuny_PATH)):
        if dir  != 'alakiSC-HF-I-04':
            if os.path.isdir(os.path.join(Suuny_PATH,dir)):
                print(dir)
                for slice_ in range(0,10):

                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                    np.random.seed(seed)  # Numpy module.
                    random.seed(seed)  # Python random module.
                    torch.manual_seed(seed)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True

                    I_label, J_label, I , J,max_frame_name,min_frame_name,no_gt,pixdim = get_data(Suuny_PATH,dir,slice_,GT_Sunny_Path)
                    # pixdim =[1,1]
                    if no_gt ==False:
                        run_code(I, J, np.squeeze(I_label), np.squeeze(J_label), os.path.join(Suuny_PATH,dir), slice_ , pixdim,max_frame_name,min_frame_name)
                        plt.close('all')




