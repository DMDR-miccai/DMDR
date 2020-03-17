


import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import skimage
import math
from skimage.transform import resize
def griddata_1d_python(nx, ny, V):
    m, n = V.shape
    res = np.zeros((nx.shape[0]))

    for i in range(nx.shape[0]):
        Rx = ny[i] - 1.0
        Ry = nx[i] - 1.0
        if (Rx >= (m - 1.0)):
            Rx = m - 1.0
            cRx = int(Rx)
            fRx = int(cRx - 1)
        else:
            if (Rx < 0):
                Rx = 0

            fRx = int(Rx)
            cRx = int(fRx + 1)

        if (Ry >= (n - 1.0)):
            Ry = n - 1.0
            cRy = int(Ry)
            fRy = int(cRy - 1)
        else:
            if (Ry < 0):
                Ry = 0

            fRy = int(Ry)
            cRy = int(fRy + 1)

        res[i] = V[fRx, fRy] * (cRx - Rx) * (cRy - Ry) + V[fRx, cRy] * (cRx - Rx) * (Ry - fRy) + V[cRx, fRy] * (
                    Rx - fRx) * (cRy - Ry) + V[cRx, cRy] * (Rx - fRx) * (Ry - fRy)

    return res


def get_pointcorrespondence_mesh(DM_i, pts, nframes):
    """Get point correspondence using the displacement matrix for all images in a series."""
    DM_fx, DM_fy, bb = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["bb"]

    iuf, ivf = pts[:, 0], pts[:, 1]
    iub, ivb = iuf, ivf
    contour_f, contour_b, contour = [], [], {}

    for k in range(nframes):
        ix = griddata_1d_python((iuf - bb[0]).astype(np.float32), (ivf - bb[2]).astype(np.float32),
                                (DM_fy[k]).astype(np.float32))
        iy = griddata_1d_python((iuf - bb[0]).astype(np.float32), (ivf - bb[2]).astype(np.float32),
                                (DM_fx[k]).astype(np.float32))
        iuf, ivf = iuf + ix, ivf + iy

        contour_f.append(np.vstack([iuf, ivf]))

    for k in range(nframes):
        contour[k] = contour_f[k]

    return contour

def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def get_Sunny_contour(S2_truth):
    # find contour number

    # contour_file = self.get_Sunny_contour_file(contour_foldere)
    # if contour_file == -1: return []
    # # load the data file
    # S2 = pydicom.dcmread(data_file).pixel_array.astype('float32')  # read_file(myFile)

    # S2_truth = mpimg.imread(contour_file)

    # get the endocardial border - 353+
    max_val_ind = np.unravel_index(np.argmax(S2_truth, axis=None), S2_truth.shape)
    max_val = S2_truth[max_val_ind]

    # get a map of just the max value
    S2_endo = np.zeros((S2_truth.shape))

    ind = np.unravel_index(np.where(S2_truth == max_val), S2_endo.shape)
    ind = np.asarray(ind)

    S2_endo[ind[:, 0], ind[:, 1]] = 1

    # get contour of endocardium
    S2_pts = skimage.measure.find_contours(S2_endo, 0)
    if not S2_pts:
        print('there is no GT for this slice')
        return S2_pts  # there is no gt for this slice

    S2_pts = S2_pts[1]
    S2_pts = np.roll(S2_pts, 1, axis=1)

    return S2_pts
def find_boundingbox(msx, msy, imxmax, imymax, bb, tol):
    """Find the bounding box based on a contour. This will allow for computing registration only within the bounding box to save time."""
    if bb == []:
        b1, b2 = np.floor(max(msx.min() - tol, 0)), np.ceil(min(msx.max() + tol, imxmax))
        b3, b4 = np.floor(max(msy.min() - tol, 0)), np.ceil(min(msy.max() + tol, imymax))
        if (b4 - b3) < (b2 - b1):

            temp = (b2 - b1) % 8
            b1 = b1 - temp
            diff = (b2 - b1) - (b4 - b3)
            b4 = b4 + diff
        elif (b4 - b3) > (b2 - b1):
            temp = (b4 - b3) % 8
            b1 = b1 - temp
            diff = - (b2 - b1) + (b4 - b3)
            b4 = b4 + diff
        else:
            diff = 0

    else:
        b1, b2 = np.floor(max(min(msx.min() - tol, bb[0]), 0)), np.ceil(min(max(msx.max() + tol, bb[1]), imxmax))
        b3, b4 = np.floor(max(min(msy.min() - tol, bb[2]), 0)), np.ceil(min(max(msy.max() + tol, bb[3]), imymax))

    return np.array([b3, b4, b1, b2])

def show_grid(I,J,deformation,deformation_r,Jw_l,Iw_l, I_label,xy_lst,h_lst,w_lst, df_p1, df_pj,df_pi,df_p3):
    contours_Warped = skimage.measure.find_contours(Jw_l.cpu().data, 0.5)
    contours_WarpedI = skimage.measure.find_contours(Iw_l.cpu().data, 0.5)
    down_factor = 1
    h_resize = int(down_factor * h_lst[0])
    w_resize = int(down_factor * w_lst[0])
    grid_x = resize(xy_lst[0].cpu()[:, :, 0].squeeze().numpy(), (h_resize, w_resize))
    grid_y = resize(xy_lst[0].cpu()[:, :, 1].squeeze().numpy(), (h_resize, w_resize))
    distx = resize((xy_lst[0] + deformation).cpu()[:, :, 0].squeeze().detach().numpy(), (h_resize, w_resize))
    disty = resize((xy_lst[0] + deformation).cpu()[:, :, 1].squeeze().detach().numpy(), (h_resize, w_resize))
    imgresize = resize(J, (h_resize, w_resize))
    distx = 0.5 * (distx + 1) * distx.shape[1]
    disty = 0.5 * (disty + 1) * disty.shape[0]
    contour_points = get_Sunny_contour(Jw_l.cpu().data)

    if len(contour_points) != 0:

    # define the bounding box
        bb = find_boundingbox(contour_points[:, 0], contour_points[:, 1], Jw_l.cpu().data.shape[1], Jw_l.cpu().data.shape[0], [],
                          30)  # 87x88
    rect = np.array(bb)
    rect = rect.astype('int')
    rect = np.round(rect)
    print(bb)
    distx = distx[rect[0]:rect[1], rect[2]-15:rect[3]]
    disty = disty[rect[0]:rect[1], rect[2]-15:rect[3]]
    # plt.imshow(imgresize, cmap='gray')
    # plt.plot(contours_Warped[0][:, 1], contours_Warped[0][:, 0], linewidth=2)
    # plt.axis('off')
    # plt.axis('equal')
    # plt.savefig(df_p2)
    # grid_x = 0.5*(distx[::2,::2]+1)*distx.shape[1]
    # disty = 0.5*(disty[::2,::2]+1)*disty.shape[0]
    # distx = distx[15:70,:]
    # disty = disty[15:70,:]
    print(distx.shape, disty.shape, grid_x.shape)
    fig, ax = plt.subplots()
    plt.imshow(imgresize, cmap='gray')
    # plt.plot(contours_Warped[0][:, 1], contours_Warped[0][:, 0], linewidth=2)
    # plot_grid(grid_x,grid_y, ax=ax,  color="lightgrey")
    plot_grid(distx[::3, ::3], disty[::3, ::3], ax=ax, color="r")

    # plot_grid(0.5*(distx[::2,::2]+1)*distx.shape[1], 0.5*(disty[::2,::2]+1)*disty.shape[0], ax=ax, color="r")
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(df_p1)
    plt.close('all')
    plt.imshow(imgresize, cmap='gray')
    plt.plot(contours_Warped[0][:, 1], contours_Warped[0][:, 0], linewidth=2,color="r")
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(df_pj)
    plt.close('all')
    distx = resize((xy_lst[0] + deformation_r).cpu()[:, :, 0].squeeze().detach().numpy(), (h_resize, w_resize))
    disty = resize((xy_lst[0] + deformation_r).cpu()[:, :, 1].squeeze().detach().numpy(), (h_resize, w_resize))
    imgresize = resize(I, (h_resize, w_resize))
    distx = 0.5 * (distx + 1) * distx.shape[1]
    disty = 0.5 * (disty + 1) * disty.shape[0]
    #TODO
    #Find the grondtruth to set sizeof grid
    contour_points = get_Sunny_contour(Iw_l.cpu().data)

    if len(contour_points) != 0:
    # define the bounding box
        bb = find_boundingbox(contour_points[:, 0], contour_points[:, 1], Iw_l.cpu().data.shape[1], Iw_l.cpu().data.shape[0], [],
                          30)  # 87x88
    distx = distx[rect[0]:rect[1], rect[2]-15:rect[3]]
    disty = disty[rect[0]:rect[1], rect[2]-15:rect[3]]
    print(distx.shape, disty.shape, grid_x.shape)
    fig, ax = plt.subplots()
    plt.imshow(imgresize, cmap='gray')

    plot_grid(distx[::3, ::3], disty[::3, ::3], ax=ax , color="r")

    plt.axis('off')
    plt.axis('equal')
    plt.savefig(df_p3)
    plt.close('all')
    plt.imshow(imgresize, cmap='gray')
    plt.plot(contours_WarpedI[0][:, 1], contours_WarpedI[0][:, 0], linewidth=2,color="r")
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(df_pi)