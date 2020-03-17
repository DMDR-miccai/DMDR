import os , re, glob
import numpy as np
import matplotlib.image as mpimg
import pydicom

import matplotlib.pyplot as plt
def get_Sunny_contour_file(contour_folder,cardiacnumofimages,sliceNum):
    gt_num = -1

    gt_bool = False
    os.chdir(contour_folder)
    files = glob.glob("*.png")
    files.sort(key=lambda x: x.split('-')[2])
    if len(files) > 0:
        for myFile in files:
            b = re.findall(r"\d+", myFile)
            numberimg = int(b[1])

            if numberimg > sliceNum * cardiacnumofimages and numberimg < (
                    (sliceNum + 1) * cardiacnumofimages) + 1 and gt_bool != True:
                if 'icontour' in myFile:
                    if gt_num < numberimg:
                        temp = gt_num
                        gt_num = numberimg
                        if temp < gt_num and temp != -1:
                            gt_path = os.path.join(contour_folder, myFile)
                            gt_bool = True
        if gt_bool != True:
            print('there is no ground trouth for this slice')
            return -1
        return gt_path
    return -1

def get_Sunny_contour(self,data_file, contour_foldere):
    #find contour number

    contour_file = self.get_Sunny_contour_file(contour_foldere)
    if contour_file ==-1:return []
    # load the data file
    S2 = pydicom.dcmread(data_file).pixel_array.astype('float32')  # read_file(myFile)


    S2_truth = mpimg.imread(contour_file)

    # get the endocardial border - 353+
    max_val_ind = np.unravel_index(np.argmax(S2_truth, axis=None), S2_truth.shape)
    max_val = S2_truth[max_val_ind]

    # get a map of just the max value
    S2_endo = np.zeros((S2.shape))

    ind = np.unravel_index(np.where(S2_truth == max_val), S2_endo.shape)
    ind = np.asarray(ind)

    S2_endo[ind[:, 0], ind[:, 1]] = 1

    # get contour of endocardium
    S2_pts = find_contours(S2_endo, 0)
    if not S2_pts:
        print('there is no GT for this slice')
        return S2_pts # there is no gt for this slice


    S2_pts = S2_pts[1]
    S2_pts = np.roll(S2_pts, 1, axis=1)

    return S2_pts



def get_data(Suuny_PATH,folder_name, slice_,GT_Sunny_Path):
    name_of_frame_min = 0
    name_of_frame_max = 0

    images = []
    print(os.path.join(GT_Sunny_Path, folder_name))
    os.chdir(os.path.join(GT_Sunny_Path, folder_name))
    files = glob.glob("*.png")
    files.sort(key=lambda x: x.split('-')[2])
    cardiacnumofimages = 20  # len(files) #20
    firstgt_num = -1
    secondgt_num = 1000
    find_gt_bool = False
    if len(files) > 0:
        maxpix = 0
        minpix = 100000
        for myFile in files:
            b = re.findall(r"\d+", myFile)
            numberimg = int(b[1])

            if numberimg > slice_ * cardiacnumofimages and numberimg < (
                    (slice_ + 1) * cardiacnumofimages) + 1 and find_gt_bool != True:
                if 'icontour' in myFile:
                    p = mpimg.imread(myFile).astype(np.float32)
                    numberofpix = np.sum(p== 1.0)
                    if numberofpix > maxpix:
                        maxpix = numberofpix
                        name_of_frame_max = numberimg
                        I_label = p


                    if numberofpix < minpix:
                        minpix = numberofpix
                        name_of_frame_min = numberimg
                        J_label = p



    else:
        return -1


    os.chdir(os.path.join(Suuny_PATH, folder_name))
    files_img = glob.glob("*.dcm")

    temp_img = files_img[0]
    b = re.findall(r"\d+", temp_img)
    if name_of_frame_min != 0 and name_of_frame_max != 0 and os.path.exists(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_min).zfill(4))))and os.path.exists(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_max).zfill(4)))):
        no_gt = False
        # J_name = pydicom.dcmread(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_min).zfill(4))))
        ds0 = pydicom.read_file(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_min).zfill(4))))
        # ds1 = pydicom.read_file(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_max).zfill(4))))

        pixdim = ds0.PixelSpacing

        J_dicom = pydicom.dcmread(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_min).zfill(4))))
        I_dicom = pydicom.dcmread(temp_img.replace(b[1], '{0}'.format(str(name_of_frame_max).zfill(4))))
        # plt.imshow(J_dicom.pixel_array.astype('float32'))
        # plt.show()
    else:
        no_gt = True
        pixdim=[1,1]
        J_label=I_label= I_dicom =J_dicom =  pydicom.dcmread(temp_img)


    return I_label,J_label,I_dicom.pixel_array.astype('float32'), J_dicom.pixel_array.astype('float32'),name_of_frame_max,name_of_frame_min,no_gt,pixdim