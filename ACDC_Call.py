
import nibabel
import numpy as np
seed = 0


np.random.seed(seed)  # Numpy module.

def correct_mask(image, value=2):
    image1 = image.copy()
    mask = image1 != value
    image1[mask] = 0
    mask = image1 == value
    image1[mask] = 1
    return image1
def ACDC_data(file_path_img,file_I_path_lb,file_J_path_lb,slice_,fgt,lgt):
    # file_path_img = file_path_img
    # file_I_path_lb = file_I_path_lb
    # file_J_path_lb = file_J_path_lb
    # slice_ = slice_
    No_gt = False
    image_mr = nibabel.load(file_path_img)
    # nibabel.inf
    # print(dataFile)
    pixdim = image_mr.header.get_zooms()
    # print(pixdim[0])
    data_im4d = image_mr.get_fdata()
    # Set the number cardiac frames
    nslice = data_im4d.shape[2]
    dim = [data_im4d.shape[0], data_im4d.shape[1]]
    print(fgt,lgt)
    I = data_im4d[:, :, slice_, fgt-1].astype(np.float32) / 255.0
    J = data_im4d[:, :, slice_, lgt-1].astype(np.float32) / 255.0
    I_label = correct_mask(((nibabel.load(file_I_path_lb)).get_fdata())[:, :, slice_], 3)
    J_label = correct_mask(((nibabel.load(file_J_path_lb)).get_fdata())[:, :, slice_], 3)
    print(I_label.max(), J_label.max())
    if I_label.max()==0 or  J_label.max() ==0 or J_label.max()>1 or  I_label.max() >1:
        No_gt = True
    return (I,J,I_label,J_label,No_gt,pixdim)


# if np.ndim(I) == 3:
#     nChannel = I.shape[2]
#
#     pyramid_I = tuple(
#         pyramid_gaussian(gaussian(I, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
#     pyramid_J = tuple(
#         pyramid_gaussian(gaussian(J, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
# elif np.ndim(I) == 2:
#     nChannel = 1
#     # pyramid_I = tuple(pyramid_gaussian(I,max_layer=L, multichannel=False))
#     # pyramid_J = tuple(pyramid_gaussian(J,max_layer=L,  multichannel=False))
#     pyramid_I = tuple(
#         pyramid_gaussian(gaussian(I, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
#     pyramid_J = tuple(
#         pyramid_gaussian(gaussian(J, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
# else:
#     print("Unknown rank for an image")
#     ifplot = False
#
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
#     plt.show()
# print('#0 done')
# start_time = time.time()
# # create a list of necessary objects you will need and commit to GPU
# I_lst, J_lst, h_lst, w_lst, xy_lst, in_lst, ind_lst = [], [], [], [], [], [], []
# for s in range(L):
#     I_, J_ = torch.tensor(pyramid_I[s].astype(np.float32)).to(device), torch.tensor(
#         pyramid_J[s].astype(np.float32)).to(device)
#     if nChannel > 1:
#         I_lst.append(I_.permute(2, 0, 1))
#         J_lst.append(J_.permute(2, 0, 1))
#         h_, w_ = I_lst[s].shape[1], I_lst[s].shape[2]
#         ind_ = torch.randperm(int(h_ * w_ * sampling))
#         ind_lst.append(ind_)
#     else:
#         I_lst.append(I_)
#         J_lst.append(J_)
#         h_, w_ = I_lst[s].shape[0], I_lst[s].shape[1]
#         ind_ = torch.randperm(int(h_ * w_ * sampling))
#         ind_lst.append(ind_)
#     print(h_, w_)
#     h_lst.append(h_)
#     w_lst.append(w_)
#
#     y_, x_ = torch.meshgrid([torch.arange(0, h_).float().to(device), torch.arange(0, w_).float().to(device)])
#     y_, x_ = 2.0 * y_ / (h_ - 1) - 1.0, 2.0 * x_ / (w_ - 1) - 1.0
#     xy_ = torch.stack([x_, y_], 2)
#     xy_lst.append(xy_)
#
#     curr, prev = np.power(downscale, -s), np.power(downscale, -np.minimum(s + 1, L - 1))
#     print(prev, curr, x_.shape, y_.shape, (prev * torch.ones(h_, w_)).shape,
#           (curr * torch.ones(h_, w_).to(device)).shape)
#     in_ = torch.stack([x_, y_, prev * torch.ones(h_, w_).to(device), curr * torch.ones(h_, w_).to(device)], 2)
#     # in_ = torch.stack([I_-J_,prev*torch.ones(h_,w_).to(device),curr*torch.ones(h_,w_).to(device)],2)
#     # in_ = torch.stack([I_-J_],2)
#     in_lst.append(in_)
#
# # 4
# gauss2 = GaussianFilter(2, 3.0).to(device)
# gauss1 = GaussianFilter(1, 1.5).to(device)
#
# np_img1 = I
# if len(np_img1.shape) == 2:  # if no channel dimension exists
#     np_img1 = np.expand_dims(np_img1, axis=-1)
# np_img1 = np.transpose(np_img1, (2, 0, 1))  # adjust dimensions for pytorch
# np_img1 = np.expand_dims(np_img1, axis=0)  # add batch dimension
# np_img1 = np_img1 / 255.0  # normalize values between 0-1
# np_img1 = np_img1.astype(np.float32)  # adjust type
#
# img1 = torch.from_numpy(np_img1)
# img2 = torch.zeros(img1.size())
# img2 = torch.sigmoid(img2)  # use sigmoid to map values between 0-1
#
# img1 = img1.to(device)
# img2 = img2.to(device)
#
# img1.requires_grad = False
# img2.requires_grad = True
#
# # loss_func = msssim
#
# # value = loss_func(img1, img2)
#
# ssim(img1, img2)
#
# ssim_loss = SSIM(window_size=11)
# lam = 1e-4
#
#
# def deformation(net, train=True):
#     loss = 0.0
#     for s in np.arange(L - 1, -1, -1):
#         if s == L - 1:
#             d_ = net(torch.cat([in_lst[s], torch.zeros(h_lst[s], w_lst[s], 2).to(device)], 2))
#         else:
#             d_up = F.grid_sample(d_.permute(2, 0, 1).unsqueeze(0), xy_lst[s].unsqueeze(0)).squeeze().permute(1, 2,
#                                                                                                              0)  # ,align_corners=True
#             in_ = torch.cat([in_lst[s], d_up], 2)
#             d_ = d_up + gauss2(net(in_).permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
#
#             # d_ = d_up + net(in_)
#         if train:
#             ind_ = torch.randperm(h_lst[s] * w_lst[s])[0:int(0.75 * h_lst[s] * w_lst[s])].to(device)
#             if nChannel > 1:
#                 Jw_ = F.grid_sample(J_lst[s].unsqueeze(0), (xy_lst[s] + d_).unsqueeze(0),
#                                     padding_mode='reflection').squeeze()  # , align_corners=True
#                 for ch in range(nChannel):
#                     loss = loss + (1. / (nChannel * L)) * F.mse_loss(
#                         gauss1(Jw_[ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]])[ind_],
#                         gauss1(I_lst[s][ch].unsqueeze(0).unsqueeze(0)).view([h_lst[s] * w_lst[s]])[ind_])
#             else:
#                 Jw_ = F.grid_sample(J_lst[s].unsqueeze(0).unsqueeze(0), (xy_lst[s] + d_).unsqueeze(0),
#                                     padding_mode='reflection').squeeze()  # ,align_corners=True
#                 mi = mine_net(torch.stack([I_lst[s], Jw_], 2), ind_lst[s])
#                 # print('one channel')
#                 Jw_exp = Jw_.unsqueeze(-1)
#                 Jw_exp = torch.transpose(Jw_exp, 0, 2)
#                 Jw_exp = torch.transpose(Jw_exp, 1, 2)
#                 Jw_exp = Jw_exp.unsqueeze(0)
#                 I_lst_exp = I_lst[s].unsqueeze(-1)
#                 I_lst_exp = torch.transpose(I_lst_exp, 0, 2)
#                 I_lst_exp = torch.transpose(I_lst_exp, 1, 2)
#                 I_lst_exp = I_lst_exp.unsqueeze(0)
#
#                 # print(1-ssim(I_lst_exp,Jw_exp).item())
#
#                 loss = loss - (1. / L) * mi + (1 - ssim(I_lst_exp,
#                                                         Jw_exp))  # F.mse_loss(gauss1(Jw_.unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_], gauss1(I_lst[s].unsqueeze(0).unsqueeze(0)).view([h_lst[s]*w_lst[s]])[ind_])
#             # regularization of the displacement vector
#             loss = loss + lam * torch.mean(d_ ** 2)
#
#     return d_, loss
#
#
# # 5
# net = Net().to(device)
# mine_net = MINE().to(device)
# # optimizer = torch.optim.LBFGS(net.parameters(),lr=1e-2) # not much luck with LBFGS
#
# #     learning_rate=0.001,
# # beta_1=0.9,
# # beta_2=0.999,
# # epsilon=1e-07,
# # amsgrad=False,
# # name='Adam',betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
#
# optimizer = optim.Adam(
#     [{'params': mine_net.parameters(), 'lr': 1e-3}, {'params': net.parameters(), 'lr': 5.0 * 1e-4}],
#     amsgrad=False)
#
# mi_list = []
# param_group = optimizer.param_groups
# # Loop
# maxiter = 400
# thresh_min = 0.00011
# prev_loss = 1
#
# iterationnum = maxiter
# for itr in range(iterationnum):
#     optimizer.zero_grad()
#     d_, loss = deformation(net, True)
#
#     # mi_list.append(-loss.item())
#     loss.backward()
#     # clear_output(wait=True)
#     # plt.plot(mi_list)
#     # plt.title("MI")
#     # plt.show()
#     optimizer.step()
#     param_group = optimizer.param_groups
#
#     if itr % 100 == 0:
#         print("learning: ", float(param_group[0]['lr']))
#         print("Itr:", itr, "loss:", -loss.item())
#         curr_loss = loss.item()
#         print('abs :', curr_loss, prev_loss)
#         if curr_loss >= prev_loss and itr != 0:
#             print('break')
#             break
#         prev_loss = loss.item()
#         # if itr%300 == 0 and itr!= 0 :
#         #   oldlr=float(param_group[0]['lr'])
#         #   lr =oldlr* 0.25
#         #   param_group[0]['lr'] = lr
#         #   oldlr=float(param_group[1]['lr'])
#         #   lr =oldlr* 0.25
#         #   param_group[1]['lr'] = lr
#
# d_, loss = deformation(net, False)
# # print(d_)
# if nChannel > 1:
#     # print(nChannel)
#     Jw_ = F.grid_sample(J_lst[0].unsqueeze
#                         (0), (xy_lst[0] + d_).unsqueeze(0), padding_mode='reflection').squeeze()
# else:
#
#     Jw_l = F.grid_sample((torch.Tensor(J_label).to(device)).unsqueeze(0).unsqueeze(0),
#                          (xy_lst[0] + d_).unsqueeze(0), padding_mode='reflection').squeeze()
#     Jw_ = F.grid_sample(J_lst[0].unsqueeze(0).unsqueeze(0), (xy_lst[0] + d_).unsqueeze(0),
#                         padding_mode='reflection').squeeze()
# import time
#
# print("--- %s seconds ---" % (time.time() - start_time))
# # print(d_.size())
# # % matplotlib
# # inline
# fig = plt.figure()
# fig.add_subplot(2, 3, 1)
# if nChannel > 1:
#     plt.imshow((J_lst[0] - I_lst[0]).permute(1, 2, 0).cpu().data)
#     plt.show()
# else:
#     plt.imshow((J_lst[0] - I_lst[0]).cpu().data)
#     plt.show()
#
# fig.add_subplot(2, 3, 2)
# if nChannel > 1:
#     plt.imshow((Jw_ - I_lst[0]).permute(1, 2, 0).cpu().data)
#     plt.show()
# else:
#     # show results
#     result_flag = True
#     print(torch.Tensor(I_label).to(device).shape != Jw_l.shape)
#     if torch.Tensor(I_label).to(device).shape[0] != Jw_l.shape[0]:
#         if torch.Tensor(I_label).to(device).shape[0] == Jw_l.shape[1]:
#             I_label = I_label.T
#             print('here***************:  ', torch.Tensor(I_label).to(device).shape, Jw_l.shape)
#         else:
#             # result_flag = False
#             print('This patient has a problem')
#     # plt.imshow((Jw_-I_lst[0]).cpu().data)
#     # if result_flag:
#     # % matplotlib
#     # inline
#     # fig = plt.figure()
#     # fig.add_subplot(2, 2, 1)
#     # plt.imshow(Jw_l.cpu().data)
#     # plt.title("registered_label")
#     # fig.add_subplot(2, 2, 2)
#     # plt.imshow(J_label)
#     # plt.title("actual label")
#     # fig.add_subplot(2, 2, 3)
#     # plt.imshow(I_label)
#     # plt.title("Fixed label")
#     # fig.add_subplot(2, 2, 4)
#     # plt.imshow((Jw_l - torch.Tensor(I_label).to(device)).cpu().data)
#     # plt.title("difference")
#     # plt.savefig('/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/Images/' + file_name + '_' + str(
#     #         slice_) + '_all.png')
#     # plt.show()
# if result_flag:
#     # save deformation field
#     if Is_ACDC == False:
#         file_name = file_path_img.split('/')[-2]
#         print(file_name)
#         nummoving = movingimg_filename.split('_')[-2][8:]
#         numfixed = fixedimg_filename.split('_')[-2][8:]
#
#         image_name = movingimg_filename.split('_')[0]
#         df_path = "/content/drive/My Drive/MICAI2020" + image_name + '_' + file_name + '_' + nummoving + '_to_' + numfixed
#     else:
#         file_name = file_path_img.split('/')[-2]
#         print(file_name)
#         # nummoving = movingimg_filename.split('_')[-2][8:]
#         # numfixed = fixedimg_filename.split('_')[-2][8:]
#
#         # image_name = movingimg_filename.split('_')[0]
#         df_path = "/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/Images/" + file_name + '_' + str(
#             slice_)
#     fig = plt.figure()
#     fig.add_subplot(2, 2, 1)
#     plt.imshow(Jw_l.cpu().data)
#     plt.title("registered_label")
#     fig.add_subplot(2, 2, 2)
#     plt.imshow(J_label)
#     plt.title("actual label")
#     fig.add_subplot(2, 2, 3)
#     plt.imshow(I_label)
#     plt.title("Fixed label")
#     fig.add_subplot(2, 2, 4)
#     plt.imshow((Jw_l - torch.Tensor(I_label).to(device)).cpu().data)
#     plt.title("difference")
#     plt.savefig(
#         '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/Images/' + file_name + '_' + str(
#             slice_) + '_all.png')
#     plt.show()
#     print(df_path)
#     np.save(df_path, d_.cpu().data.numpy())
#     print(Jw_l.cpu().data.numpy().shape[0], torch.Tensor(I_label).to(device).shape[0])
#     if torch.Tensor(I_label).to(device).shape[0] == Jw_l.cpu().data.numpy().shape[0]:
#         print(Jw_l.cpu().data.numpy().max())
#         print(float(I_label.max()))
#         dice_after = "%.6f" % (np.sum(Jw_l.cpu().data.numpy()[I_label == 1]) * 2.0 / (
#                 np.sum(Jw_l.cpu().data.numpy()) + np.sum(I_label)))
#
#         dice_before = "%.6f" % (np.sum(J_label[I_label == 1]) * 2.0 / (
#                 np.sum(J_label) + np.sum(I_label)))
#
#         print("unregiterde Dice: ", dice_before)
#         print("regiterde Dice: ", dice_after)
#
#         # HD
#         contours_target = skimage.measure.find_contours(I_label, 0.5)
#         cnt = skimage.measure.find_contours(Jw_l.cpu().data.numpy(), 0.5)
#         HD1 = directed_hausdorff(contours_target[0], cnt[0])
#
#         cnt_before = skimage.measure.find_contours(J_label, 0.5)
#         HD_before = directed_hausdorff(contours_target[0], cnt_before[0])
#
#         print("registered HD: ", HD1[0])
#         print("unregistered HD: ", HD_before[0])
#
#         loss_before = F.mse_loss(J_lst[0], I_lst[0])
#         loss_after = F.mse_loss(Jw_, I_lst[0])
#         print(loss_before.item(), loss_after.item())
#
#         if Is_ACDC == False:
#             with open(
#                     '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/resultsfile_ACDC_22.txt',
#                     'a') as the_file:
#                 the_file.write(image_name + '_' + file_name + '_' + nummoving + '_to_' + numfixed + '\n')
#                 the_file.write('Unregistered Dice: ' + str(dice_before) + '  \n')
#                 the_file.write('registered Dice: ' + str(dice_after) + '  \n')
#                 the_file.write('Unregistered HD: ' + str(HD_before[0]) + '  \n')
#                 the_file.write('registered HD: ' + str(HD1[0]) + '  \n')
#                 the_file.write('unregistered Loss: ' + str(loss_before.item()) + '  \n')
#                 the_file.write('registered Loss: ' + str(loss_after.item()) + '  \n')
#                 the_file.write('  \n')
#         else:
#             with open(
#                     '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/resultsfile_ACDC_22.txt',
#                     'a') as the_file:
#                 the_file.write(file_name + '_' + str(slice_) + '\n')
#                 the_file.write('Unregistered Dice: ' + str(dice_before) + '  \n')
#                 the_file.write('registered Dice: ' + str(dice_after) + '  \n')
#                 the_file.write('Unregistered HD: ' + str(HD_before[0]) + '  \n')
#                 the_file.write('registered HD: ' + str(HD1[0]) + '  \n')
#                 the_file.write('unregistered Loss: ' + str(loss_before.item()) + '  \n')
#                 the_file.write('registered Loss: ' + str(loss_after.item()) + '  \n')
#                 the_file.write('  \n')
#
#         down_factor = 0.25
#         h_resize = int(down_factor * h_lst[0])
#         w_resize = int(down_factor * w_lst[0])
#         grid_x = resize(xy_lst[0].cpu()[:, :, 0].squeeze().numpy(), (h_resize, w_resize))
#         grid_y = resize(xy_lst[0].cpu()[:, :, 1].squeeze().numpy(), (h_resize, w_resize))
#         distx = resize((xy_lst[0] + d_).cpu()[:, :, 0].squeeze().detach().numpy(), (h_resize, w_resize))
#         disty = resize((xy_lst[0] + d_).cpu()[:, :, 1].squeeze().detach().numpy(), (h_resize, w_resize))
#
#         fig, ax = plt.subplots()
#         # plt.imshow(I)
#         plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
#         plot_grid(distx, disty, ax=ax, color="C0")
#         if Is_ACDC == False:
#             plt.savefig(
#                 '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/Images/' + file_name + '_' + image_name + '_to_' + numfixed + '.png')
#             plt.show()
#         else:
#             plt.savefig(
#                 '/home/ameneh/UofA/git/Pyramidresolution_Deformable_Registration/Results/MICCAI2020/ACDC/Images/' + file_name + '_' + str(
#                     slice_) + '.png')
#             plt.show()
# exit(0)
# % reset - f except variables(file_path_img, file_path_lb, g, a)
# sys.exit()
# %reset_selective -f Jw_l, Jw_, xy_,xy_lst,I,I_label,J,d,loss,x_,y_,net, J_label,I_lst,J_lst,Jw_l, name_of_frame_max,name_of_frame_min,pyramid_I,pyramid_J, h_resize,w_resize,grid_x,rid_y,distx,disty
