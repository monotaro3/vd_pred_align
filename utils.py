# coding: utf-8

import numpy as np
#from osgeo import gdal, osr
from chainer import serializers
import os
from chainercv import utils as utils_chainercv
import chainer
from chainer import Variable
import chainer.functions as F
from chainercv.links.model.ssd import VGG16Extractor300
import pickle
import cv2 as cv
import math
#from process_COWC import scale_img_bbox
from COWC_dataset_processed import vehicle_classes
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd, defaultbox_size_300, defaultbox_size_512

def draw_rect(image, bbox, match,mode="TEST"):
    for i in range(bbox.shape[0]):
        if mode == "TEST":
            color = (0, 0, 255) if match[i] == 1 else (255, 0, 0)
        if mode == "GT":
            color = (0, 255, 0)
        cv.rectangle(image, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color)

def initSSD(modelname,resolution,path=None,SA=False):
    if modelname == "ssd300":
        attn_args = {}
        model = SSD300_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet', defaultbox_size=defaultbox_size_300[resolution],SA=SA,**attn_args)
    elif modelname == "ssd512":
        model = SSD512_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet', defaultbox_size=defaultbox_size_512[resolution])
    if path != None:
        serializers.load_npz(path, model)
    return model

def gen_dms_time_str(time_sec):
    sec = time_sec % 60
    min = (int)(time_sec / 60) % 60
    hour = (int)(time_sec / 60 / 60) % 24
    day = (int)(time_sec / 60 / 60 / 24)
    time_str = ""
    if day!=0: time_str = str(day) + " day(s)"
    if time_str != "" or hour != 0: time_str = time_str + " " + str(hour) + " hour(s)"
    if time_str != "" or min != 0: time_str = time_str + " " + str(min) + " min(s)"
    time_str = time_str + (" " if time_str != "" else "") + str(sec) + " sec(s)"
    return time_str

def make_bboxeslist_chainercv(gt_file):
    gt_txt = open(gt_file,"r")
    bboxes = []
    line = gt_txt.readline()
    while (line):
        xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([ymin-1, xmin-1, ymax-1, xmax-1])
        line = gt_txt.readline()
    bboxes = np.stack(bboxes).astype(np.float32)
    return bboxes

def convertDepth(input,output):
    from osgeo import gdal, osr
    gdal.AllRegister()
    src = gdal.Open(input)
    SpaRef = src.GetProjection()
    geoTransform = src.GetGeoTransform()

    cols = src.RasterXSize
    rows = src.RasterYSize
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output, cols, rows, 3, gdal.GDT_Byte)
    dst.SetGeoTransform(geoTransform)
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromWkt(SpaRef)
    dst.SetProjection(dstSRS.ExportToWkt())
    for i in range(1,4):
        src_data_ = src.GetRasterBand(i).ReadAsArray()
        src_data = src_data_.copy()
        src_data[src_data >= 1024] = 1023
        src_data = (src_data/4).astype(np.uint8)
        dst.GetRasterBand(i).WriteArray(src_data)
    dst = None

def decode_mask2bbox(maskimg,windowsize):
    rp = np.where(maskimg == 255)
    h,w,c = maskimg.shape
    dots_ = []
    dot_num = rp[0].size
    for i in range(dot_num):
        dots_.append([rp[0][i]+1, rp[1][i]+1])
    bbox = []
    window_half = int(windowsize / 2)
    for d in dots_:
        xmin = d[1] - window_half
        ymin = d[0] - window_half
        xmax = d[1] + window_half
        ymax = d[0] + window_half
        if windowsize % 2 == 0:
            xmin +=1
            ymin +=1
        if xmin < 1: xmin = 1
        if ymin < 1: ymin = 1
        if xmax > w: xmax = w
        if ymax > h: ymax = h
        bbox.append([xmin,ymin,xmax,ymax])
    return bbox

def convertImgs2fmaps(imgsdir, savedir, model_path, model_class = "extractor", res = 0.3, dirname="train"):
    num_example = 0
    files = os.listdir(imgsdir)
    imgs = []
    for f in files:
        root, ext = os.path.splitext(f)
        if ext in [".png", ".tif"]:
            imgs.append(os.path.join(imgsdir,f))
    if model_class == "extractor":
        extractor = VGG16Extractor300()
        serializers.load_npz(model_path, extractor)
        extractor.to_gpu()
    else:
        if model_class in ["ssd300","ssd512"]:
            ssd_model = initSSD(model_class,res,model_path)
            extractor = ssd_model.extractor
            extractor.to_gpu()
        else:
            print("invalid model class name")
            return
    fmaplist = os.path.join(savedir,"list",dirname+".txt")
    if not os.path.isdir(os.path.join(savedir,dirname)): os.makedirs(os.path.join(savedir,dirname))
    if not os.path.isdir(os.path.join(savedir, "list")):os.makedirs(os.path.join(savedir, "list"))
    for imgpath in imgs:
        num_example += 1
        if num_example % 100 == 1: print("processing {0}th file...".format(num_example))
        img = utils_chainercv.read_image(imgpath, color=True)
        img = img[np.newaxis, :]
        img = chainer.cuda.to_gpu(img)
        with chainer.no_backprop_mode():
            fmap_ = extractor(img)
        fmap = []
        for i in range(6):
            fmap.append(chainer.cuda.to_cpu(fmap_[i][0].data))
        dir, file = os.path.split(imgpath)
        root, ext = os.path.splitext(file)
        with open(fmaplist, 'a') as listfile:
            listfile.write(root + "\n")
        with open(os.path.join(savedir,dirname,root+".fmp"),"wb") as f:
            pickle.dump(fmap,f)

def make_img_cutout(imgdir,savedir,imgsize_out,scale = 1, margin = 0,useEdge=False, rotate=False):
    num_example = 0
    savedir_img = "train"
    savedir_list = "list"
    if scale != 1:
        imgsize = int(imgsize_out / scale)
    else:
        imgsize = imgsize_out
    if not os.path.isdir(os.path.join(savedir,savedir_img)):os.makedirs(os.path.join(savedir,savedir_img))
    if not os.path.isdir(os.path.join(savedir, savedir_list)): os.makedirs(os.path.join(savedir, savedir_list))
    listfile_path = os.path.join(savedir,savedir_list,savedir_img+".txt")
    files = os.listdir(imgdir)
    imgs = []
    for f in files:
        root, ext = os.path.splitext(f)
        if ext in [".tif",".png","jpg"]:
            imgs.append(os.path.join(imgdir,f))
    for imgpath in imgs:
        img = cv.imread(imgpath)
        H, W, c = img.shape
        W_slot = int(math.ceil((W-margin)/(imgsize-margin)))
        H_slot = int(math.ceil((H-margin)/(imgsize-margin)))
        root, ext = os.path.splitext(imgpath)
        for h in range(H_slot):
            if h == H_slot-1:
                if H == (h+1) * (imgsize-margin) + margin or useEdge:
                    h_start, h_end = -imgsize, None
                else: continue
            else:
                h_start, h_end = h * (imgsize-margin), h * (imgsize-margin) + imgsize
            for w in range(W_slot):
                if w == W_slot - 1:
                    if W == (w + 1) * (imgsize - margin) + margin or useEdge:
                        w_start, w_end = -imgsize, None
                    else:
                        continue
                else:
                    w_start, w_end = w * (imgsize - margin), w * (imgsize - margin) + imgsize
                cutout = img[h_start:h_end,w_start:w_end,:]
                cutout_images = [cutout]
                if rotate:
                    center = (int(imgsize / 2), int(imgsize / 2))
                    for i in range(3):
                        angle = (i+1) * 90.0
                        rmat = cv.getRotationMatrix2D(center, angle, 1.0)
                        cutout_images.append(cv.warpAffine(cutout, rmat, (imgsize, imgsize)))
                for outimg in cutout_images:
                    num_example += 1
                    if num_example % 100 == 0: print("{0}th file has been output.".format(num_example))
                    rootname = "{0:010d}".format(num_example)
                    if imgsize_out != imgsize:
                        outimg = cv.resize(outimg,(imgsize_out, imgsize_out))
                    cv.imwrite(os.path.join(savedir,savedir_img,rootname+ext),outimg)
                    with open(listfile_path, 'a') as list:
                        list.write(rootname + "\n")

def scale_img_bbox(img, bbox, output_size_h):
    h,w,c = img.shape
    scale = output_size_h / h
    img_ = cv.resize(img, (int(w*scale), output_size_h))
    bbox_ = []
    for b in bbox:
        xmin = math.floor(b[0] * scale)
        ymin = math.floor(b[1] * scale)
        xmax = math.floor(b[2] * scale)
        ymax = math.floor(b[3] * scale)
        b_ = [xmin,ymin,xmax,ymax]
        for i in range(len(b_)):
            if b_[i] == 0: b_[i] = 1
            if b_[i] > output_size_h: b_[i] = output_size_h
        bbox_.append(b_)
    return img_, bbox_

def scaleConvert(srcdir,dstdir,scale):
    if srcdir == dstdir :
        print("srcdir and dstdir must not be the same")
        return
    if not os.path.isdir(dstdir): os.makedirs(dstdir)
    files = os.listdir(srcdir)
    uniques = []
    for f in files:
        root, ext = os.path.splitext(f)
        if not root in uniques:
            uniques.append(root)
    for u in uniques:
        srcimg = cv.imread(os.path.join(srcdir,u + ".tif"))
        h, w, c = srcimg.shape
        outpusize_h = int(h * scale)
        srcbbox = []
        with open(os.path.join(srcdir,u + ".txt"),"r") as annotations:
            line = annotations.readline()
            while (line):
                xmin, ymin, xmax, ymax = line.split(",")
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                srcbbox.append([xmin, ymin, xmax, ymax])
                line = annotations.readline()
        dstimg, dstbbox = scale_img_bbox(srcimg,srcbbox,outpusize_h)
        cv.imwrite(os.path.join(dstdir,u + ".tif"),dstimg)
        with open(os.path.join(dstdir, u + ".txt"), 'w') as bbox_text:
            for b in dstbbox:
                bbox_text.write(",".join(map(str, b)) + "\n")


def bbox_visualize(imgfile,gtfile):
    #assume voc coordinate
    img = cv.imread(imgfile)
    # bboxes = []
    # # labels =[]
    # with open(gtfile, "r") as annotations:
    #     line = annotations.readline()
    #     while (line):
    #         xmin, ymin, xmax, ymax = line.split(",")
    #         xmin = int(xmin)
    #         ymin = int(ymin)
    #         xmax = int(xmax)
    #         ymax = int(ymax)
    #         bboxes.append([xmin, ymin, xmax, ymax])
    #         # labels.append(vehicle_classes.index("car"))
    #         line = annotations.readline()
    # from SSD_test import draw_rect
    bboxes = make_bboxeslist_chainercv(gtfile)
    draw_rect(img, bboxes, np.array((0,) * bboxes.shape[0], dtype=np.int8), mode="GT")
    name, ext = os.path.splitext(imgfile)
    cv.imwrite(name + "_visualize"+ext, img)

def prepare_gt_annotations(bboxes_, labels_, ssdmodel):

    xp = ssdmodel.xp
    n_class = ssdmodel.multibox.n_class
    # replace loc data of bg with gaussian noise
    # labels_dup = xp.expand_dims(labels_,axis=-1)
    # labels_dup = xp.dstack((labels_dup,) * 4)
    fg_indexes = xp.where(labels_ != 0)
    bboxes = xp.random.randn(*(bboxes_.shape))
    bboxes[fg_indexes] = bboxes_[fg_indexes]
    # bboxes_ = bboxes

    l_shape = labels_.shape
    labels_ = labels_.flatten()
    labels = np.zeros(labels_.shape + (n_class,))

    labels[(np.arange(len(labels_)), labels_)] = 1

    # labels_ = labels_.reshape(l_shape)  # for debug
    labels = labels.reshape(l_shape + (n_class,))

    return bboxes, labels

def concat_gt_annotations(bboxes, labels, ssdmodel):

    xp = ssdmodel.xp

    if not isinstance(bboxes, Variable): bboxes = Variable(xp.asarray(bboxes).astype("f"))
    if not isinstance(labels, Variable): labels = Variable(xp.asarray(labels).astype("f"))

    aspect_ratios = ssdmodel.multibox.aspect_ratios
    # n_class = ssdmodel.multibox.n_class
    grids = ssdmodel.extractor.grids

    loc_num_boxes = [0]
    # conf_num_boxes = [0]

    for i, ar in enumerate(aspect_ratios):
        n = (len(ar) + 1) * 2
        loc_num_boxes.append(grids[i] ** 2 * n)

    # print(loc_num_boxes)
    loc_num_boxes = np.cumsum(np.array(loc_num_boxes))
    # print(loc_num_boxes)

    bboxes_splits = []
    labels_splits = []

    for i in range(len(loc_num_boxes) - 1):
        bboxes_splits.append(bboxes[:, loc_num_boxes[i]:loc_num_boxes[i + 1], ...])
        labels_splits.append(labels[:, loc_num_boxes[i]:loc_num_boxes[i + 1], ...])

    # for b in bboxes_splits:
    #     print(b.shape[1])

    for i, (b, l, ar) in enumerate(zip(bboxes_splits, labels_splits, aspect_ratios)):
        n = (len(ar) + 1) * 2
        bboxes_splits[i] = b.reshape(b.shape[0], grids[i], grids[i], b.shape[-1] * n).transpose((0, 3, 1, 2))
        labels_splits[i] = l.reshape(l.shape[0], grids[i], grids[i], l.shape[-1] * n).transpose((0, 3, 1, 2))

    for i in range(1, len(grids)):

        dst_size = grids[0]
        size = grids[i]
        k_size = int(math.ceil(dst_size / size))
        # bs = len(bboxes_splits[i])
        # ch = bboxes_splits[i].shape[1]

        # print("k_size:{}".format(k_size))

        # a = np.ones((bs, ch, size, size))
        index = int(math.ceil(k_size ** 2 / 2)) - 1
        if k_size % 2 == 0: index -= k_size / 2
        indexes_b = np.empty(bboxes_splits[i].shape).astype("i")
        indexes_b[...] = index
        indexes_l = np.empty(labels_splits[i].shape).astype("i")
        indexes_l[...] = index

        print("index:{}".format(index))

        bboxes_splits[i] = F.upsampling_2d(bboxes_splits[i], indexes_b, k_size, k_size, outsize=(dst_size, dst_size))
        labels_splits[i] = F.upsampling_2d(labels_splits[i], indexes_l, k_size, k_size, outsize=(dst_size, dst_size))
        # print(b.shape)
        # print(b[1, 1, :])
        # print(np.where(b.array[1, 1, :] == 1))
    import chainer
    # bboxes_splits[0] = chainer.Variable(bboxes_splits[0])
    # labels_splits[0] = chainer.Variable(labels_splits[0])

    # bboxes_map = F.concat(bboxes_splits, axis=1)
    # labels_map = F.concat(labels_splits, axis=1)

    return F.concat(bboxes_splits + labels_splits, axis=1)

if __name__ == "__main__":
    #convertDepth('c:/work/spacenet\\raw\\RGB-PanSharpen_AOI_5_Khartoum_img1.tif',"c:/work/test4.tif")
    # trainer = None
    # serializers.load_npz("model/snapshot_iter_30000", trainer)
    # pass
    #convertImgs2fmaps("E:/work/vehicle_detection_dataset/cowc_300px_0.3/train","E:/work/vehicle_detection_dataset/cowc_300px_0.3_fmap","model/vgg_300_0.3_30000")
    # make_img_cutout("E:/work/dataset/experiments/vehicle_detection_dataset/da_target_raw","E:/work/dataset/experiments/vehicle_detection_dataset/target_nolabels_full",300,0.16/0.3,rotate=False)#0.16/0.3)
    #scaleConvert("../DA_images/NTT2","../DA_images/NTT2_scale0.3",0.16/0.3)
    bbox_visualize("E:/work/dataset/experiments/vehicle_detection_dataset/target_labels/train/0000000009.png","E:/work/dataset/experiments/vehicle_detection_dataset/target_labels/train/0000000009.txt")
    # bbox_visualize("E:/work/dataset/experiments/vehicle_detection_dataset/cowc_300px_0.3_daug_nmargin/train/0000000017.png",
    #                "E:/work/dataset/experiments/vehicle_detection_dataset/cowc_300px_0.3_daug_nmargin/train/0000000017.txt")