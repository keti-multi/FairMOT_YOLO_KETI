from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
from models.model import create_model, load_model
from datasets.dataset.jde import DetEvalDataset, collate_fn
# from datasets.dataset.jde_attribute import AttDetDataset, collate_fn


from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process


def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def test_det(
        opt,
        iou_thres=0.5,
        print_interval=40,
):
    nC = 1

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)

    gts=open('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt.txt','r')
    # dets=open('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/Edge Device Result/288_160_16_yolov5n_linear_31/conf_thres0.4/result_0.4.txt','r')
    dets=open('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/정량평가/result_detect.txt','r')

    gt_dict = {}
    det_dict = {}
    for i in range(100):
        gt_dict[str(i + 1)] = []
        det_dict[str(i + 1)] = []

    while True:
        line = gts.readline()
        if not line: break
        f_num, f_id, x, y, w, h = line.split(" ")
        gt_dict[str(f_num)].append([int(f_id), float(x), float(y), float(w), float(h[:-1])])
    gts.close()
    while True:
        line = dets.readline()
        if not line: break
        f_num, f_id, x, y, w, h,conf,_,_,_,_ = line.split(",")
        det_dict[str(f_num)].append([int(f_id), float(x), float(y), float(w), float(h[:-1]),float(conf)])
    dets.close()

    path_ = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/out"
    imgs = os.listdir(path_)
    imgs.sort()
    for i in range(100):
        t = time.time()
        path = os.path.join(path_, imgs[i])
        img0 = cv2.imread(path)
        # id x y w h => cx cy w h
        targets_f=np.array(gt_dict[str(i+1)])
        # id x1 y1 w h conf
        dets=np.array(det_dict[str(i+1)])[:,1:6]
        dets[:,0] += dets[:,2]/2
        dets[:,1] += dets[:,3]/2
        # Compute average precision for each sample
        labels = targets_f
        labels=labels[:,1:5]

        # x y w h
        seen += 1
        width=1920
        height=1080
        # print("labels : ",labels)
        if dets is None:
            # If there are labels but no detections mark as zero AP
            if labels.size(0) != 0:
                mAPs.append(0), mR.append(0), mP.append(0)
            continue

        # If no labels add number of detections as incorrect
        correct = []
        if labels.__len__() == 0:
            # correct.extend([0 for _ in range(len(detections))])
            mAPs.append(0), mR.append(0), mP.append(0)
            continue
        else:
            target_cls = 0
            # Extract target boxes as (x1, y1, x2, y2)
            # xywh -> x1 y1 x2 y2
            target_boxes = xywh2xyxy(labels)
            target_boxes[:, 0] *= width
            target_boxes[:, 2] *= width
            target_boxes[:, 1] *= height
            target_boxes[:, 3] *= height
            dets = xywh2xyxy(dets)
            if os.path.exists(os.path.join(opt.save_dir,'gt',path.split('/')[-3])):
                pass
            else :
                os.makedirs(os.path.join(opt.save_dir,'gt',path.split('/')[-3]))
            # print("target_boxes : ",target_boxes)
            # path = paths[si]
            # print("path : ",path.split('/')[-3])
            # img0 = cv2.imread(path)
            # img1 = cv2.imread(path)
            # for t in range(len(target_boxes)):
            #     x1 = int(target_boxes[t, 0])
            #     y1 = int(target_boxes[t, 1])
            #     x2 = int(target_boxes[t, 2])
            #     y2 = int(target_boxes[t, 3])
                # cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
            # if os.path.exists(os.path.join(opt.save_dir,'gt',path.split('/')[-3])):
            #     pass
            # else :
            #     os.makedirs(os.path.join(opt.save_dir,'gt',path.split('/')[-3]))
            # cv2.imwrite(os.path.join(opt.save_dir,'gt',path.split('/')[-3],path.split('/')[-1].split('.')[0]+'.jpg'), img0)
            # for t in range(len(dets)):
            #     x1 = int(float(dets[t, 0]))
            #     y1 = int(float(dets[t, 1]))
            #     x2 = int(float(dets[t, 2]))
            #     y2 = int(float(dets[t, 3]))
                # cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
            # if os.path.exists(os.path.join(opt.save_dir, 'pred', path.split('/')[-3])):
            #     pass
            # else:
            #     os.makedirs(os.path.join(opt.save_dir, 'pred', path.split('/')[-3]))
            # # cv2.imwrite(os.path.join(opt.save_dir,'pred',path.split('/')[-3],path.split('/')[-1].split('.')[0]+'.jpg'), img1)
            # #abc = ace
            detected = []

            for *pred_bbox,conf in dets:
                obj_pred = 0
                pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, torch.Tensor(target_boxes), x1y1x2y2=True)[0]
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > iou_thres and obj_pred == 0 and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)
        # Compute Average Precision (AP) per class
        AP, AP_class, R, P = ap_per_class(tp=correct,
                                          conf=dets[:, 4],
                                          pred_cls=np.zeros_like(dets[:, 4]),  # detections[:, 6]
                                          target_cls=np.zeros_like(dets[:, 4]))
        # Accumulate AP per class
        AP_accum_count += np.bincount(AP_class, minlength=nC)
        AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

        # Compute mean AP across all classes in this image, and append to image list
        mAPs.append(AP.mean())
        mR.append(R.mean())
        mP.append(P.mean())

        # Means of all images
        mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
        mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
        mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        cv2.imwrite(
            os.path.join(opt.save_dir, 'gt', path.split('/')[-3],
                         path.split('/')[-1].split('.')[0] + '.jpg'), img0)
        print(('%11s%11s' + '%11.3g' * 4 + 's') %
              (seen, 100, mean_P, mean_R, mean_mAP, time.time() - t))
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        map = test_det(opt)
