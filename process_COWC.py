# coding: utf-8

import argparse
import cv2 as cv
import numpy as np
import os
import math
from utils import decode_mask2bbox, scale_img_bbox

# process_directories = []
# # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Potsdam_ISPRS")
# # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Selwyn_LINZ")
# # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Toronto_ISPRS")
# # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Utah_AGRC")
#
# save_directory = ""
#
# directories_list = "process_COWC_dirs.txt"
#
# dir_lists = [dir_.strip() for dir_ in open(directories_list)]
# for dir_ in dir_lists:
#     record = dir_.split(",")
#     if record[0] == "process" : process_directories.append(record[1])
#     elif record[0] == "save": save_directory = record[1]
#     else:
#         try:
#             raise ValueError("invalid directory specification")
#         except ValueError as e:
#             print(e)
#
# print("process dirs:")
# print(process_directories)
#
# output_size = 300
# windowsize = 50
# adjust_margin = False
# margin = 100
# mask_margin_test = True
# use_edge = False
# scale = 0.5 # set None when not using, 0.5 -> halve the resolution
# rotate = True
#
# train_img_number = 0
# test_img_number = 0
# all_img_number_unique = 0
# train_imglist_text = os.path.join(save_directory,"list","train.txt")
# test_imglist_text = os.path.join(save_directory,"list","validation.txt")
#
# if not os.path.isdir(os.path.join(save_directory,"train")):
#     os.makedirs(os.path.join(save_directory,"train"))
# if not os.path.isdir(os.path.join(save_directory, "validation")):
#     os.makedirs(os.path.join(save_directory, "validation"))
# if not os.path.isdir(os.path.join(save_directory, "list")):
#     os.makedirs(os.path.join(save_directory, "list"))


def rotateBbox(bbox_, bboxsize, degree, imgshape): #imgshape(np.array):(W,H)
    center = imgshape/2
    bbox = []
    rad = np.radians(-degree)  #axis y is reversed
    rMat = np.matrix([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    for b in bbox_:
        b_center = np.array(((b[0] + b[2])/2,(b[1] + b[3])/2))
        #b_size_half = (b[2] - b[0]+1 + b[3] - b[1]+1)/4
        b_size_half = bboxsize/2
        b_center = b_center - center
        b_center = np.array((rMat*(b_center[np.newaxis,:].T)).T)[0] + center
        p_min = np.round(b_center - b_size_half)
        if (p_min >= imgshape).any(): continue
        p_min[p_min <= 0] = 1
        p_max = np.round(b_center + b_size_half)
        if (p_max <= 1).any(): continue
        mask = p_max > imgshape
        p_max[mask] = imgshape[mask]
        b_ = np.hstack((p_min,p_max)).astype(np.int32)
        bbox.append(b_.tolist())
    return bbox

def mask_margin(img, bbox, margin,windowsize):
    img_ = img.copy()
    H, W, C = img.shape
    mask_img = np.ones(img.shape[0:2],dtype=bool)
    mask_img[margin:-margin,margin:-margin] = False
    img_[mask_img] = 0
    bbox_ = []
    for b in bbox:
        if b[0] == 1: center_x = b[2] - windowsize / 2
        elif b[2] == W: center_x = b[0] + windowsize / 2
        else: center_x = (b[0] + b[2]) / 2
        if b[1] == 1: center_y = b[3] - windowsize / 2
        elif b[3] == H : center_y = b[1] + windowsize / 2
        else: center_y = (b[1] + b[3]) / 2
        if center_x <= margin or center_y <= margin  or center_x > W - margin or center_y > H - margin: continue
        xmin = b[0] if b[0] > margin else margin + 1
        ymin = b[1] if b[1] > margin else margin + 1
        xmax = b[2] if b[2] <= W - margin else W - margin
        ymax = b[3] if b[3] <= H - margin else H - margin
        bbox_.append([xmin,ymin,xmax,ymax])
    return img_, bbox_

def make_img_cutouts(image_path,image_mask_path,save_directory,cutout_size,size_output,windowsize,rotate,mask_margin_test,maketestdata=True):
    image = cv.imread(image_path)
    image_mask = cv.imread(image_mask_path)
    height, width, channel = image.shape
    pointer = [0,0] #W,H
    #cutout_ = np.empty((cutout_size,cutout_size,3))
    #cutout_mask_ = np.empty((cutout_size, cutout_size, 3))
    write_flag = False
    global train_img_number
    global test_img_number
    global all_img_number_unique
    global train_imglist_text
    global test_imglist_text
    global adjust_margin
    global margin
    n_angles = 4 if rotate else 1
    if not adjust_margin:
        W_slot = int((width-margin)/(cutout_size-margin))
        H_slot = int((height-margin)/(cutout_size-margin))
        for H in range(H_slot+1):
            if H != H_slot or use_edge:
                for W in range(W_slot+1):
                    if H == H_slot:
                        if use_edge:
                            if W == W_slot:
                                cutout_ = image[-cutout_size:,-cutout_size:,:]
                                cutout_mask_ = image_mask[-cutout_size:, -cutout_size:, :]
                                write_flag = True
                            else:
                                cutout_ = image[-cutout_size:, (cutout_size-margin)*W:(cutout_size-margin)*(W+1)+margin, :]
                                cutout_mask_ = image_mask[-cutout_size:, (cutout_size-margin)*W:(cutout_size-margin)*(W+1)+margin, :]
                                write_flag = True
                    else:
                        if W == W_slot:
                            if use_edge:
                                cutout_ = image[(cutout_size-margin)*H:(cutout_size-margin)*(H+1)+margin, -cutout_size:, :]
                                cutout_mask_ = image_mask[(cutout_size-margin)*H:(cutout_size-margin)*(H+1)+margin, -cutout_size:, :]
                                write_flag = True
                        else:
                            cutout_ = image[(cutout_size-margin)*H:(cutout_size-margin)*(H+1)+margin,  (cutout_size-margin)*W:(cutout_size-margin)*(W+1)+margin, :]
                            cutout_mask_ = image_mask[(cutout_size-margin)*H:(cutout_size-margin)*(H+1)+margin,  (cutout_size-margin)*W:(cutout_size-margin)*(W+1)+margin, :]
                            write_flag = True
                    if write_flag:
                        write_flag = False
                        bbox_ = decode_mask2bbox(cutout_mask_,windowsize)
                        if len(bbox_) > 0:  #omit if there is no car
                            all_img_number_unique += 1
                            if not maketestdata or all_img_number_unique % 4 != 0:
                                usage = "train"
                                train_img_number += n_angles
                                img_number = train_img_number
                            else:
                                usage = "validation"
                                test_img_number += n_angles
                                img_number = test_img_number

                            center = (int(cutout_size/2),int(cutout_size/2))

                            for i in range(img_number-n_angles+1,img_number+1):
                                filename = '{0:010d}.png'.format(i)
                                filename_annotation = '{0:010d}.txt'.format(i)
                                angle = (i+n_angles-1-img_number)*90.0
                                if angle != 0:
                                    rmat = cv.getRotationMatrix2D(center, angle, 1.0)
                                    cutout = cv.warpAffine(cutout_, rmat, (cutout_size,cutout_size))
                                    #cutout_mask = cv.warpAffine(cutout_mask_, rmat, (cutout_size, cutout_size))
                                    #bbox = decode_mask2bbox(cutout_mask, windowsize)
                                    bbox = rotateBbox(bbox_,windowsize,angle,np.roll(cutout.shape[0:2],1))
                                else:
                                    cutout = cutout_.copy()
                                    bbox = bbox_.copy()

                                if usage == "validation":
                                    if margin > 0 and mask_margin_test:
                                        cutout, bbox = mask_margin(cutout, bbox, margin,windowsize)
                                    if len(bbox) == 0: continue

                                if cutout_size != size_output:
                                    cutout, bbox = scale_img_bbox(cutout, bbox, size_output)

                                cv.imwrite(os.path.join(save_directory,usage,filename),cutout)
                                with open(os.path.join(save_directory,usage,filename_annotation), 'w') as bbox_text:
                                    for b in bbox:
                                        bbox_text.write(",".join(map(str, b))+"\n")
                                if usage == "train":
                                    with open(train_imglist_text,'a') as train_list:
                                        train_list.write('{0:010d}'.format(i)+"\n")
                                else:
                                    with open(test_imglist_text,'a') as test_list:
                                        test_list.write('{0:010d}'.format(i)+"\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--directories_list', type=str, default="process_COWC_dirs.txt")

    #Images to be processed and save directory is specified by a text file of the following format:
    #
    # process,E:/work/COWCdataset/ground_truth_sets/Potsdam_ISPRS
    # process,E:/work/COWCdataset/ground_truth_sets/Selwyn_LINZ
    # process,E:/work/COWCdataset/ground_truth_sets/Toronto_ISPRS
    # process,E:/work/COWCdataset/ground_truth_sets/Utah_AGRC
    # save,E:/work/vehicle_detection_dataset/cowc_300px_0.3_daug_nmargin
    #
    parser.add_argument('--output_size', type=int, default=300)
    parser.add_argument('--windowsize', type=int, default=50) # This window size will be scaled by the variable "scale" below
    parser.add_argument('--adjust_margin', action='store_true')
    parser.add_argument('--margin', type=int, default=0)
    parser.add_argument('--mask_margin_test', action='store_true')
    parser.add_argument('--use_edge', action='store_true')
    parser.add_argument('--scale', type=float, default=0.5) # set None when not using, 0.5 -> halve the resolution
    parser.add_argument('--norotate', action='store_true')
    parser.add_argument('--trainonly', action='store_true') #False to make only training data
    parser.add_argument('--mydata', action='store_true') #True to process my own format, False to process COWC dataset
    parser.add_argument('--train_start_num', type=int, default=1)

    args = parser.parse_args()

    process_directories = []
    # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Potsdam_ISPRS")
    # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Selwyn_LINZ")
    # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Toronto_ISPRS")
    # process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Utah_AGRC")

    save_directory = ""

    directories_list = args.directories_list #"process_COWC_dirs.txt"

    dir_lists = [dir_.strip() for dir_ in open(directories_list)]
    for dir_ in dir_lists:
        record = dir_.split(",")
        if record[0] == "process" : process_directories.append(record[1])
        elif record[0] == "save": save_directory = record[1]
        else:
            try:
                raise ValueError("invalid directory specification")
            except ValueError as e:
                print(e)

    print("process dirs:")
    print(process_directories)

    output_size = args.output_size #300
    windowsize = args.windowsize #50 # This window size will be scaled by the variable "scale" below
    adjust_margin = args.adjust_margin #False
    margin = args.margin #0
    mask_margin_test = args.mask_margin_test #False
    use_edge = args.use_edge #False
    scale = args.scale #0.5 # set None when not using, 0.5 -> halve the resolution
    rotate = not(args.norotate) #True
    maketestdata = not(args.trainonly) #True #False to make only training data
    mydata_mode = args.mydata #False  #True to process my own format, False to process COWC dataset

    train_img_number = args.train_start_num - 1
    test_img_number = 0
    all_img_number_unique = 0
    train_imglist_text = os.path.join(save_directory,"list","train.txt")
    test_imglist_text = os.path.join(save_directory,"list","validation.txt")

    if not os.path.isdir(os.path.join(save_directory,"train")):
        os.makedirs(os.path.join(save_directory,"train"))
    if maketestdata:
        if not os.path.isdir(os.path.join(save_directory, "validation")):
            os.makedirs(os.path.join(save_directory, "validation"))
    if not os.path.isdir(os.path.join(save_directory, "list")):
        os.makedirs(os.path.join(save_directory, "list"))

    cutout_size = math.floor(output_size / scale) if scale != None else output_size

    for directory in process_directories:
        print("current directory:"+directory)
        filelist = os.listdir(directory)
        filelist.sort()
        image_list = []
        image_mask_list = []
        if mydata_mode:
            for file in filelist:
                root, ext = os.path.splitext(file)
                if ext == ".tif" and root.find("_gtmap") == -1:
                    image_list.append(os.path.join(directory, file))
                    image_mask_list.append(os.path.join(directory, root + "_gtmap.tif"))
        else:
            for file in filelist:
                root, ext = os.path.splitext(file)
                if ext == ".png" and root.find("Annotated") == -1:
                    image_list.append(os.path.join(directory, file))
                    image_mask_list.append(os.path.join(directory, root + "_Annotated_Cars.png"))

        for image_path, image_mask_path in zip(image_list,image_mask_list):
            print("processing image:" + image_path)
            make_img_cutouts(image_path, image_mask_path, save_directory, cutout_size, output_size,windowsize,rotate,mask_margin_test,maketestdata)

