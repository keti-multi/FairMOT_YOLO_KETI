import motmetrics as mm
import numpy as np

'''
Following function loads the ground truth and tracker object files, processes them and produces a set of metrices.
'''

import cv2
import matplotlib.pyplot as plt

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color



import os

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 3
lineType               = 2

import numpy as np

def visualization(img_frame,gt_boxes, tar_boxes,out):
    # TODO 20230914 SYH
    dirr= "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220715_keti/images"
    img_src = os.path.join(dirr,'%d.jpg'%img_frame)
    img_origin = cv2.imread(img_src)
    img_eval= img_origin.copy()
    for dets in gt_boxes:
        dets[1] = dets[1] * 1920
        dets[2] = dets[2] * 1080
        dets[3] = dets[3] * 1920
        dets[4] = dets[4] * 1080

    # print("gt_dets : ", gt_dets)
    # bbox ==[cx,cy,w,h]
    for i in range(0, gt_boxes.shape[0]):

        bbox = gt_boxes[i,1:]
        bbox=[int(inn) for inn in bbox]
        gt_boxes[i, 1:]

        # bbox=int(bbox)
        # bbox == [cx,cy]
        # cv2.rectangle(img_eval, (bbox[0], bbox[1]),
        #               (bbox[0]+bbox[2],  bbox[1]+bbox[3]),
        #               (0, 255, 0), 2)
        cv2.putText(img_eval, str(int(gt_boxes[i,0])),
                    (bbox[0]-int(bbox[2]/2), bbox[1]-int(bbox[3]/2)-10),
                    font,
                    fontScale,
                    (0, 255, 0),
                    thickness,
                    lineType)
        bbox[0]=bbox[0]-int(bbox[2]/2)
        bbox[1]=bbox[1] - int(bbox[3] / 2)
        cv2.rectangle(img_eval, (bbox[0], bbox[1]),
                      (bbox[0]+int(bbox[2]), bbox[1]+int(bbox[3])),
                      (0, 255, 0), 2)
        gt_boxes[i, 1:] = bbox
        ### confirmed

        # bbox ==[x1,y1,w,h]
    for i in range(0, tar_boxes.shape[0]):
        color = get_color(tar_boxes[i][0])
        bbox = tar_boxes[i,1:]
        bbox=[int(inn) for inn in bbox]
        # cv2.rectangle(img_eval, (bbox[0], bbox[1]),
        #               (bbox[2], bbox[3]),
        #               color, 2)
        cv2.rectangle(img_eval, (bbox[0], bbox[1]),
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                      color, 2)
        cv2.putText(img_eval, str(int(tar_boxes[i,0])),
                    (bbox[0], bbox[1]-10),
                    font,
                    fontScale,
                    color,
                    thickness,
                    lineType)
    out.write(img_eval)
    return gt_boxes,tar_boxes
    # cv2.imshow('dets', img_eval)
    # cv2.waitKey(0)

# import glob

# img_array = []
# for filename in glob.glob('C:/New folder/Images/*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
#
# out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()



def motMetricsEnhancedCalculator(gtSource, tSource):
  # load ground truth
  gt = np.loadtxt(gtSource, delimiter=' ')
  # load tracking output
  t = np.loadtxt(tSource, delimiter=',')
  # Create an accumulator that will be updated during each frame
  acc = mm.MOTAccumulator(auto_id=True)
  # Max frame number maybe different for gt and t files
  out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (1920,1080))

  for frame in range(int(gt[:,0].max())):
    # print("frame : ",frame)
    frame += 1 # detection and frame numbers begin at 1
    if frame%1000 == 0:
        print("frame : ", frame)
    # select id, x, y, width, height for current frame
    # required format for distance calculation is X, Y, Width, Height \
    # We already have this format
    gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
    # print("gt_dets : ",gt_dets)
    # for dets in gt_dets:
    #     dets[1] = dets[1] * 1920
    #     dets[2] = dets[2] * 1080
    #     dets[3] = dets[3] * 1920
    #     dets[4] = dets[4] * 1800
    # print("gt_dets : ", gt_dets)
    t_dets = t[t[:,0]==frame,1:6] # select all detections in t
    # print("t_dets : ",t_dets)
    gt_dets,t_dets = visualization(frame, gt_dets, t_dets,out)

    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=0.5) # format: gt, t


    # print(" C : ",C)
    # Call update once for per frame.
    # format: gt object ids, t object ids, distance
    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)
  mh = mm.metrics.create()
  out.release()
  summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], \
                      name='acc')
  strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
  )
  print(strsummary)
'''

Run the function by pointing to the ground truth and your generated results as follows

'''
#python
# Calculate for pre-trained Yolov3 + SORT

##/media/syh/hdd/data/infoworks_220725_cctv/output
# motMetricsEnhancedCalculator('gt/groundtruth.txt', \
#                              'to/trackeroutput.txt')
#
# motMetricsEnhancedCalculator('/media/syh/hdd/data/infoworks_220725_cctv/images/train/infoworks_220725_cctv/gt/gt.txt', \
#                               '/media/syh/hdd/data/infoworks_220725_cctv/images/results/MUF_train_yolov5_288_160/infoworks_220725_cctv.txt')

# import cv2
# gts = open('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt.txt', 'r')
# dets = open(
#     '/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/Edge Device Result/288_160_16_yolov5n_linear_31/conf_thres0.2/result.txt',
#     'r')

# motMetricsEnhancedCalculator('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt.txt', \
#                               '/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/Edge Device Result/288_160_16_yolov5n_linear_31/conf_thres0.1/result.txt')
#

# motMetricsEnhancedCalculator('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220715_keti/datas/cam-001/20220715130000_mp4/keti_220715_gt.txt', \
#                               '/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/Edge_Device Result/result_mota.txt')
# TODO 20230914 SYH
motMetricsEnhancedCalculator('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220715_keti/datas/cam-001/20220715130000_mp4/keti_220715_gt.txt', \
                              '/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/정량평가/result_mota.txt')

# motMetricsEnhancedCalculator('/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220707_jointree/datas/cam-001/20220707130000_mp4/jointree_220707_gt.txt', \
#                               '/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/Edge_Device Result/result_mota_AREA_220707.txt')
"""
   num_frames  IDF1       IDP       IDR      Rcll      Prcn   GT  MT  PT  ML  FP  FN  IDsw  FM      MOTA      MOTP
acc         150  0.75  0.857143  0.666667  0.743295  0.955665  261   0   2   0   9  67     1  12  0.704981  0.244387
"""
