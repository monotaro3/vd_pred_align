# coding: utf-8

import matplotlib.pyplot as plot
import chainer
from chainer import serializers
from chainer import reporter
from chainercv import utils
import math
import os
import numpy as np
from chainercv.visualizations import vis_bbox
from eval_detection_voc_custom import eval_detection_voc_custom
import cv2 as cv
import csv

#--custom
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from COWC_dataset_processed import vehicle_classes
from SSD_for_vehicle_detection import defaultbox_size_300, defaultbox_size_512
from utils import make_bboxeslist_chainercv, draw_rect

def ssd_predict(model, image, margin,nms_thresh = 0.45,calc_entropy=False):
    size = model.insize
    c, H, W = image.shape

    stride = size-margin
    H_slot = math.ceil((H - margin) / stride)
    W_slot = math.ceil((W - margin) / stride)

    bbox = list()
    label = list()
    score = list()
    if calc_entropy:
        entropy_list = []

    #to be fixed: crashes when image is smaller than cutout size

    for h in range(H_slot):
        offset_H = stride * h if h < H_slot-1 else H - size
        for w in range(W_slot):
            offset_W = stride * w if w < W_slot-1 else W - size
            cutout = image[:,offset_H:offset_H+size,offset_W:offset_W+size]
            if calc_entropy:
                bboxes, labels, scores, entropy = model.predict([cutout],calc_entropy=calc_entropy)
            else:
                bboxes, labels, scores = model.predict([cutout])
            bbox_, label_, score_ = bboxes[0], labels[0], scores[0]
            bbox_[:,(0,2)] += offset_H # bbox_: (y_min, x_min, y_max, x_max)
            bbox_[:, (1, 3)] += offset_W
            bbox.append(bbox_)
            label.append(label_)
            score.append(score_)
            if calc_entropy:
                entropy_list.append(entropy)
    bbox = np.vstack(bbox).astype(np.float32)
    label = np.hstack(label).astype(np.int32)
    score = np.hstack(score).astype(np.float32)

    bbox_nms = list()
    label_nms = list()
    score_nms = list()

    #label-wise nms
    for l in range(len(vehicle_classes)):
        mask_l = label == l
        bbox_l = bbox[mask_l]
        score_l = score[mask_l]
        indices = utils.non_maximum_suppression(
            bbox_l, nms_thresh, score_l)
        bbox_l = bbox_l[indices]
        score_l = score_l[indices]
        bbox_nms.append(bbox_l)
        label_nms.append(np.array((l,) * len(bbox_l)))
        score_nms.append(score_l)
    bbox = np.vstack(bbox_nms).astype(np.float32)
    label = np.hstack(label_nms).astype(np.int32)
    score = np.hstack(score_nms).astype(np.float32)

    if calc_entropy:
        return bbox, label, score, entropy_list

    return bbox, label, score

def mask_trim(maskimg,bbox, label,score):
    bbox_center_y = ((bbox[:,0] + bbox[:,2])/2).astype(np.int)
    bbox_center_x = ((bbox[:,1] + bbox[:,3])/2).astype(np.int)
    center_values = maskimg[bbox_center_y,bbox_center_x,2]
    inner_mask = center_values == 255
    return bbox[inner_mask], label[inner_mask], score[inner_mask]

def ssd_test(ssd_model, imagepath, modelsize="ssd300", resolution=0.16, procDir=False, testonly = False, resultdir ="result", evalonly=False, mask=False, gpu = 0, iou_threshold = 0.4, mbox_another=None,calc_entropy=False,score_threshold=0.6,score_thre_step=None):
    margin = 50
    # gpu = 0

    images = []
    gt_files = []

    # if (not evalonly) and (not os.path.isdir(resultdir)):
    if not os.path.isdir(resultdir):
        os.makedirs(resultdir)

    if procDir:
        if not os.path.isdir(imagepath):
            try:
                raise(ValueError("invalid image directory path"))
            except ValueError as e:
                print(e)
                return
        files = os.listdir(imagepath)
        for f in files:
            root, ext = os.path.splitext(f)
            if ext in [".tif", ".png", ".jpg"]:
                images.append(os.path.join(imagepath,f))
                if not testonly:
                    gt_files.append(os.path.join(imagepath,root+".txt"))
    else:
        images.append(imagepath)
        if not testonly:
            root, ext = os.path.splitext(imagepath)
            gt_files.append(root + ".txt")

    if not isinstance(ssd_model,chainer.link.Link):
        if modelsize == 'ssd300':
            model = SSD300_vd(
                n_fg_class=len(vehicle_classes),
                defaultbox_size=defaultbox_size_300[resolution])
        else:
            model = SSD512_vd(
                n_fg_class=len(vehicle_classes),
                defaultbox_size=defaultbox_size_512[resolution])

        serializers.load_npz(ssd_model, model)
    else:
        model = ssd_model

    model.use_preset("evaluate")

    if mbox_another:
        serializers.load_npz(mbox_another, model.multibox)

    if gpu >= 0 and model.xp == np: model.to_gpu()
    #if model.xp == np: model.to_gpu()

    #predict
    bboxes = []
    labels = []
    scores = []
    bboxes_evaluate = []
    labels_evaluate = []
    scores_evaluate = []
    gt_bboxes = []
    gt_labels = []
    result_stat = []
    if calc_entropy:
        entropy_list = []
    for i in range(len(images)):
        image = utils.read_image(images[i], color=True)
        if calc_entropy:
            bbox, label, score, entropy = ssd_predict(model, image, margin,calc_entropy=calc_entropy)
        else:
            bbox, label, score = ssd_predict(model,image,margin)
        if mask:
            root, ext = os.path.splitext(images[i])
            mask_file = root + "_mask.png"
            if os.path.isfile(mask_file):
                maskimg = cv.imread(mask_file)
                bbox, label, score = mask_trim(maskimg,bbox,label,score)
            else:
                print("No mask file:" + mask_file)

        bboxes_evaluate.append(bbox)
        labels_evaluate.append(label)
        scores_evaluate.append(score)

        # bboxes.append(bbox)
        # labels.append(label)
        # scores.append(score)

        mask_det = score >= score_threshold

        bboxes.append(bbox[mask_det])
        labels.append(label[mask_det])
        scores.append(score[mask_det])

        if calc_entropy: entropy_list.extend(entropy)
        if testonly:
            dirpath, fname = os.path.split(images[i])
            result_stat.append([fname, "number of detected cars",len(bbox)])
            result_stat.append([])
        else:
            gt_bbox = make_bboxeslist_chainercv(gt_files[i])
            gt_bboxes.append(gt_bbox)
            # labels are without background, i.e. class_labels.index(class). So in this case 0 means cars
            gt_labels.append(np.stack([0]*len(gt_bbox)).astype(np.int32))
            result_i, _, _, _ = eval_detection_voc_custom([bbox], [label], [score], [gt_bbox], [
                np.stack([0] * len(gt_bbox)).astype(np.int32)], iou_thresh=iou_threshold)
            _, stats_i, matches_i, selec_i = eval_detection_voc_custom([bbox[mask_det]], [label[mask_det]], [score[mask_det]], [gt_bbox], [
                np.stack([0] * len(gt_bbox)).astype(np.int32)], iou_thresh=iou_threshold)
            # result_i, stats_i, matches_i, selec_i = eval_detection_voc_custom([bbox],[label],[score],[gt_bbox],[np.stack([0]*len(gt_bbox)).astype(np.int32)],iou_thresh=iou_threshold)
            dirpath,fname = os.path.split(images[i])
            root, ext = os.path.splitext(os.path.join(resultdir,fname))
            result_stat.append([fname,"map",result_i["map"]])
            result_stat.append(['', "ap", float(result_i["ap"])])
            result_stat.append(['', "PR", stats_i[0]["PR"]])
            result_stat.append(['', "RR", stats_i[0]["RR"]])
            result_stat.append(['', "FAR", stats_i[0]["FAR"]])
            result_stat.append(['', "F1", stats_i[0]["F1"]])
            result_stat.append(['', "mean_ap_F1",  (result_i['map'] + (stats_i[0]['F1'] if stats_i[0]['F1'] != None else 0)) / 2])
            result_stat.append(['', "number of detected cars",len(bbox)])
            result_stat.append(['', "number of TP", np.sum(matches_i[0]==1)])
            result_stat.append(['', "number of FP", np.sum(matches_i[0] == 0)])
            result_stat.append(['', "number of gt", len(gt_bbox)])
            result_stat.append(['', "number of FN(undected gt)", np.sum(selec_i[0]==False)])
            # result_stat.append(['', "number of FN(undected gt)(calculated)", len(gt_bbox)-np.sum(matches_i[0]==1)])
            result_stat.append([])
            # resulttxt_i = root + "_result.txt"
            # with open(resulttxt_i, mode="w") as f:
            #     f.write(str(result_i) + "\n" + str(stats_i))

    if not testonly:
        # result, stats, matches, selec_list = eval_detection_voc_custom(bboxes,labels,scores,gt_bboxes,gt_labels,iou_thresh=iou_threshold)
        result, _, _, _ = eval_detection_voc_custom(bboxes_evaluate, labels_evaluate, scores_evaluate, gt_bboxes, gt_labels,
                                                                       iou_thresh=iou_threshold)
        _, stats, matches, selec_list = eval_detection_voc_custom(bboxes, labels, scores, gt_bboxes, gt_labels,
                                                                       iou_thresh=iou_threshold)
        mean_ap_f1 = (result['map'] + (stats[0]['F1'] if stats[0]['F1'] != None else 0)) / 2

        if score_thre_step:
            thre_steps = []
            stats_steps = []
            matches_steps = []
            selec_list_steps = []
            assert isinstance(score_thre_step, list) and len(score_thre_step) == 3
            for thre in np.linspace(score_thre_step[0],score_thre_step[1], num=int((score_thre_step[1]-score_thre_step[0])/score_thre_step[2])+1):
                bboxes_step = [bboxes_evaluate[i][scores_evaluate[i] >= thre] for i in range(len(scores_evaluate))]
                labels_step = [labels_evaluate[i][scores_evaluate[i] >= thre] for i in range(len(scores_evaluate))]
                scores_step = [scores_evaluate[i][scores_evaluate[i] >= thre] for i in range(len(scores_evaluate))]
                _, stats_, matches_, selec_list_ = eval_detection_voc_custom(bboxes_step, labels_step, scores_step, gt_bboxes, gt_labels,
                                                                          iou_thresh=iou_threshold)
                thre_steps.append(thre)
                stats_steps.append(stats_)
                matches_steps.append(matches_)
                selec_list_steps.append(selec_list_)


    if not evalonly:
        #visualizations
        for imagepath , bbox, label, score in zip(images,bboxes,labels,scores):
            dir, imagename = os.path.split(imagepath)
            result_name, ext = os.path.splitext(imagename)
            image = utils.read_image(imagepath, color=True)
            if mask:
                mask_inverted = np.ones(maskimg.shape,dtype=bool)
                mask_indices = np.where(maskimg==255)
                mask_inverted[mask_indices[0],mask_indices[1],:] = False
                mask_inverted = mask_inverted.transpose((2,0,1))
                image[mask_inverted] = (image[mask_inverted]/2).astype(np.int32)
            vis_bbox(
                image, bbox, label, score, label_names=vehicle_classes)
            #plot.show()
            plot.savefig(os.path.join(resultdir,result_name+ "_vis1.png"))

            #result
            image_ = image.copy()
            image_ = image_.transpose(1,2,0)
            image_ = cv.cvtColor(image_, cv.COLOR_RGB2BGR)
            if testonly:
                draw_rect(image_, bbox, np.array((1,) * bbox.shape[0], dtype=np.int8))
            else:
                draw_rect(image_,bbox,matches[images.index(imagepath)])
            cv.imwrite(os.path.join(resultdir,result_name+ "_vis2.png"),image_)
            gt_bbox = gt_bboxes[images.index(imagepath)]
            undetected_gt = gt_bbox[selec_list[images.index(imagepath)]==False]
            draw_rect(image_, undetected_gt, np.array((0,) * undetected_gt.shape[0], dtype=np.int8), mode="GT")
            cv.imwrite(os.path.join(resultdir, result_name + "_vis3.png"), image_)

            #gt visualization
            if not testonly:
                image_ = image.copy()
                image_ = image_.transpose(1, 2, 0)
                image_ = cv.cvtColor(image_, cv.COLOR_RGB2BGR)
                gt_bbox = gt_bboxes[images.index(imagepath)]
                draw_rect(image_, gt_bbox, np.array((0,) * gt_bbox.shape[0],dtype=np.int8),mode="GT")
                cv.imwrite(os.path.join(resultdir,result_name+ "_vis_gt.png"), image_)
    if not testonly:
        result_txt = os.path.join(resultdir, "result.txt")
        with open(result_txt,mode="w") as f:
            f.write(str(result) +"\n" + str(stats) + '\nmean_ap_F1: ' + str(mean_ap_f1))
        # result_stat.append(["summary", "map", result["map"]])
        # result_stat.append(['', "ap", float(result["ap"])])
        # result_stat.append(['', "PR", stats[0]["PR"]])
        # result_stat.append(['', "RR", stats[0]["RR"]])
        # result_stat.append(['', "FAR", stats[0]["FAR"]])
        # result_stat.append(['', "F1", stats[0]["F1"]])
        # result_stat.append(
        #     ['', "mean_ap_F1", mean_ap_f1])
        if score_thre_step:
            result_stat_ladder = []
            result_stat_ladder.extend([["evaluation in ladder of score threshold"]])
            result_stat_ladder.append(["threshold","PR", "RR", "FAR", "F1"])
            for i in range(len(thre_steps)):
                result_stat_ladder.append([thre_steps[i],stats_steps[i][0]["PR"],stats_steps[i][0]["RR"],stats_steps[i][0]["FAR"],stats_steps[i][0]["F1"]])
            # result_stat.append([])

        columns = ["summary", "map", "ap", "PR", "RR", "FAR", "F1","mean_ap_F1"]
        values = ["",result["map"],float(result["ap"]),stats[0]["PR"],stats[0]["RR"],stats[0]["FAR"],stats[0]["F1"],mean_ap_f1]
        if calc_entropy:
            entropy_all = np.array([en[0] for en in entropy_list])
            entropy_pos = np.array([en[1] for en in entropy_list])
            entropy_neg = np.array([en[2] for en in entropy_list])
            columns.extend(["avr. entropy(all)","avr. entropy(pos)","avr. entropy(neg)"])
            # values.append(np.array(entropy_list).mean())
            values.extend([entropy_all.mean(),np.nanmean(entropy_pos),entropy_neg.mean()])
        result_stat.extend([columns,values])

    with open(os.path.join(resultdir,"result_stat.csv"),"w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(result_stat)

    if score_thre_step:
        with open(os.path.join(resultdir, "result_stat_ladder.csv"), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(result_stat_ladder)

    if not testonly:
        print(result)
        print(stats)
        print("mean_ap_F1:{0}".format(mean_ap_f1))

        return result, stats

class ssd_evaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, img_dir, target,updater, savedir, n_ranking=5, resolution=0.3,modelsize="ssd300",evalonly=True, label_names=None,save_bottom = 0.6, gpu = 0,suffix=None, save_snapshot = True):
        super(ssd_evaluator, self).__init__(
            {}, target)
        self.img_dir = img_dir
        self.resolution = resolution
        self.modelsize = modelsize
        self.label_names = label_names
        self.evalonly = evalonly
        self.save_bottom = save_bottom
        self.n_ranking = n_ranking
        self.rank_map =  []
        self.rank_map_data = np.full((self.n_ranking,3),-1.,dtype=np.float32)
        self.rank_F1 = []
        self.rank_F1_data = np.full((self.n_ranking, 3), -1., dtype=np.float32)
        self.rank_mean = []
        self.rank_mean_data = np.full((self.n_ranking, 4), -1., dtype=np.float32)
        self.updater = updater
        self.savedir = savedir
        self.gpu = gpu
        if suffix:
            self.default_name += suffix
        self.save_snapshot = save_snapshot

    def evaluate(self):
        target = self._targets['main']
        result, stats = ssd_test(target, self.img_dir, procDir=True, resolution=self.resolution,
                 modelsize=self.modelsize,evalonly=self.evalonly,resultdir=self.savedir, gpu=self.gpu)

        report = {'map': result['map']}
        for i in range(len(stats)):
            classname = self.label_names[i] if self.label_names is not None else "class"+str(i)
            report['PR/{:s}'.format(classname)] = stats[i]['PR']
            report['RR/{:s}'.format(classname)] = stats[i]['RR']
            report['FAR/{:s}'.format(classname)] = stats[i]['FAR']
            report['F1/{:s}'.format(classname)] = stats[i]['F1']

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        current_iteration = self.updater.iteration
        save_flag = False
        del_iter = []
        mean_F1 = 0
        for i in range(len(stats)):
            if stats[i]['F1'] != None:
                mean_F1 += stats[i]['F1']
        mean_F1 /= len(stats)
        mean_map_mF1 = (result['map'] + mean_F1) / 2
        if len(self.rank_map) == 0:
            if result['map'] > self.save_bottom:
                save_flag = True
                self.rank_map.append([current_iteration, result['map'],mean_F1])
        elif result['map'] > self.rank_map[-1][1]:
            save_flag = True
            if len(self.rank_map) ==self.n_ranking:
                iter = self.rank_map.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_map.append([current_iteration, result['map'],mean_F1])
            self.rank_map.sort(key=lambda x: x[1],reverse=True)
        if len(self.rank_F1) == 0:
            if mean_F1 > self.save_bottom:
                save_flag = True
                self.rank_F1.append([current_iteration, result['map'],mean_F1])
        elif mean_F1 > self.rank_F1[-1][2]:
            save_flag = True
            if len(self.rank_F1) == self.n_ranking:
                iter = self.rank_F1.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_F1.append([current_iteration, result['map'],mean_F1])
            self.rank_F1.sort(key=lambda x: x[2],reverse=True)
        if len(self.rank_mean) == 0:
            if mean_map_mF1 > self.save_bottom:
                save_flag = True
                self.rank_mean.append([current_iteration, result['map'],mean_F1,mean_map_mF1])
        elif mean_map_mF1 > self.rank_mean[-1][3]:
            save_flag = True
            if len(self.rank_mean) ==self.n_ranking:
                iter = self.rank_mean.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_mean.append([current_iteration, result['map'],mean_F1,mean_map_mF1])
            self.rank_mean.sort(key=lambda x: x[3],reverse=True)
        if save_flag and self.save_snapshot:
            serializers.save_npz(os.path.join(self.savedir,target.__class__.__name__ + "_{0}.npz".format(current_iteration)),target)
        for iter in del_iter:
            if not iter in [i[0] for i in self.rank_map + self.rank_F1 + self.rank_mean]:
                if os.path.isfile(os.path.join(self.savedir,target.__class__.__name__ + "_{0}.npz".format(iter))):
                    os.remove(os.path.join(self.savedir,target.__class__.__name__ + "_{0}.npz".format(iter)))

        self.encode(self.rank_map,self.rank_map_data)
        self.encode(self.rank_F1, self.rank_F1_data)
        self.encode(self.rank_mean, self.rank_mean_data)

        ranking_summary = [["best map"],["iter","map","F1"]]
        ranking_summary.extend(self.rank_map)
        ranking_summary.extend([[],["best F1"], ["iter", "map", "F1"]])
        ranking_summary.extend(self.rank_F1)
        ranking_summary.extend([[], ["best mean"], ["iter", "map", "F1","mean"]])
        ranking_summary.extend(self.rank_mean)
        with open(os.path.join(self.savedir,"ranking.csv"),"w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(ranking_summary)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation

    def encode(self,ranking_list,data):
        n_rank = len(ranking_list)
        if n_rank > 0:
            data[:n_rank] = np.array(ranking_list)

    def decode(self, data):
        ranking_list = []
        for i in range(len(data)):
            if data[i][0] == -1:
                break
            else:
                ranking_list.append([data[i][j] for j in range(data.shape[1])])
                ranking_list[i][0] = int(ranking_list[i][0]) #iter number must be int
        return ranking_list

    def serialize(self, serializer):
        self.rank_map_data = serializer('rank_map', self.rank_map_data)
        self.rank_F1_data = serializer('rank_F1', self.rank_F1_data)
        self.rank_mean_data = serializer('rank_mean', self.rank_mean_data)
        if isinstance(serializer, chainer.serializer.Deserializer):
            self.rank_map = self.decode(self.rank_map_data)
            self.rank_F1 = self.decode(self.rank_F1_data)
            self.rank_mean = self.decode(self.rank_mean_data)

if __name__ == "__main__":
    imagepath = "E:/work/dataset/raw/DA_images/NTT_scale0.3/2_6" #c:/work/DA_images/kashiwa_lalaport/0.3"#"#"E:/work/vehicle_detection_dataset/cowc_processed/train/0000000001.png"
    # imagepath = "E:/work/dataset/experiments/vehicle_detection_dataset/source_val"
    modelpath = "E:/work/experiments/trained_models/ssd_adv_emin1_gamma0.3/ssdmodel_iter_30000" #"model/DA/NTT_buf_alt_100_nalign_DA4_nmargin/SSD300_vd_7000.npz" #"model/DA/CORAL/ft_patch_w100000000_nmargin/SSD300_vd_33750.npz"
    # modelpath = "E:/work/DA_vehicle_detection/model/DA/m_thesis/additional/x13_nmargin_nobias/model_iter_50000"
    # result_dir = "../chainer-cyclegan/experiment_data/results/NTT_fake_GT_L1_l10/"
    result_dir = 'E:/work/experiments/results/ssd_adv_emin1_gamma0.3'
    ssd_test(modelpath,imagepath,procDir=True,resultdir=result_dir,resolution=0.3,modelsize="ssd300", iou_threshold=0.4,calc_entropy=True)
    # for i in range(1,10):
    #     # modelpath = modelpath.format(i)
    #     #
    #     # result_dir = result_dir + "{}".format(i)
    #     ssd_test(modelpath.format(i), imagepath, procDir=True, resultdir=result_dir.format(i), resolution=0.3, modelsize="ssd300",
    #              iou_threshold=0.4)


