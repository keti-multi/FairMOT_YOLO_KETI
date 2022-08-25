import itertools
import os
import os.path as osp
import time
from collections import deque
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process

from tracker import matching

from .basetrack import BaseTrack, TrackState

#from rknn.api import RKNN
import onnx
import onnxruntime as ort
import json
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
torch.set_printoptions(profile="full")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        # print(temp_feat)
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks): ##
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id): ##
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False): ##
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True): ##
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self): ##
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self): ##
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh): ##
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr): ##
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()


        if opt.make_onnx != '':
            self.Convert_ONNX(self.model)
            exit(-1)

        if opt.run_onnx != '':
            self.session = ort.InferenceSession(opt.run_onnx)
            print('done')

        if opt.make_rknn != '':
            self.rknn = RKNN()

            print('--> Config model')
            self.rknn.config(#mean_values=[[0, 0, 0]], # [0.408, 0.447, 0.470] [104.04, 113.985, 119.85]
                            #std_values=[[255, 255, 255]], # [0.289, 0.274, 0.278] [73.695, 69.87, 70.89]
                            # reorder_channel='0 1 2',
                            target_platform = 'rk1808',
                            optimization_level=3,
                            quantized_dtype="dynamic_fixed_point-i16", # asymmetric_quantized-u8 dynamic_fixed_point-i8 dynamic_fixed_point-i16
                            quantized_algorithm="normal")  # normal mmse kl_divergence

            # Load ONNX model
            print('--> Loading model')
            load_onnx = opt.make_rknn.replace('rknn', 'onnx')
            ret = self.rknn.load_onnx(model=load_onnx)#, inputs=['modelInput'], input_size_list=[[3, 608, 1088]], outputs=['hm', 'wh', 'id', 'reg'])
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')

            # Build model
            print('--> Building model')
            # ret = self.rknn.build(do_quantization=False)
            ret = self.rknn.build(do_quantization=True, dataset='/home/hjlee/FairMOT/Dataset/dataset.txt')
            if ret != 0:
                print('Build model failed!')
                exit(ret)
            print('done')

            # Export RKNN model
            print('--> Export RKNN model')
            ret = self.rknn.export_rknn(opt.make_rknn)
            if ret != 0:
                print('Export rknn failed!')
                exit(ret)
            print('done')


            # self.rknn.accuracy_analysis(inputs='/home/hjlee/FairMOT/Dataset/dataset_acc.txt', 
            #                         output_dir='./normal_quantization_analysis',
            #                         target='rk1808',
            #                         device_id='13171fbb0ecb2508')

            self.rknn.release()
            exit(-1)

        if opt.make_hybrid1 != '':
            self.rknn = RKNN()

            print('--> Config model')
            self.rknn.config(#mean_values=[[0.408, 0.447, 0.470]], # [0.408, 0.447, 0.470] [104.04, 113.985, 119.85]
                            #std_values=[[0.289, 0.274, 0.278]], # [0.289, 0.274, 0.278] [73.695, 69.87, 70.89]
                            # reorder_channel='0 1 2',
                            target_platform = 'rk1808',
                            optimization_level=3,
                            quantized_dtype="dynamic_fixed_point-i16", # asymmetric_quantized-u8 dynamic_fixed_point-i8 dynamic_fixed_point-i16
                            quantized_algorithm="kl_divergence")  # normal mmse kl_divergence

            # Load ONNX model
            print('--> Loading model')
            load_onnx = opt.make_hybrid1.replace('rknn', 'onnx')
            ret = self.rknn.load_onnx(model=load_onnx)#, inputs=['modelInput'], input_size_list=[[3, 608, 1088]], outputs=['hm', 'wh', 'id', 'reg'])
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')

            # Hybrid quantization step1
            print('--> hybrid_quantization_step1')
            ret = self.rknn.hybrid_quantization_step1(dataset='/home/hjlee/FairMOT/Dataset/dataset.txt')
            if ret != 0:
                print('hybrid_quantization_step1 failed!')
                exit(ret)
            print('done')

            self.rknn.release()
            exit(-1)

        if opt.make_hybrid2 != '':
            self.rknn = RKNN()

            print('--> Config model')
            self.rknn.config(#mean_values=[[0.408, 0.447, 0.470]], # [0.408, 0.447, 0.470] [104.04, 113.985, 119.85]
                            #std_values=[[0.289, 0.274, 0.278]], # [0.289, 0.274, 0.278] [73.695, 69.87, 70.89]
                            # reorder_channel='0 1 2',
                            target_platform = 'rk1808',
                            optimization_level=3,
                            quantized_dtype="dynamic_fixed_point-i16", # asymmetric_quantized-u8 dynamic_fixed_point-i8 dynamic_fixed_point-i16
                            quantized_algorithm="kl_divergence")  # normal mmse kl_divergence

            # Hybrid quantization step2
            print('--> hybrid_quantization_step2')
            ret = self.rknn.hybrid_quantization_step2(model_input='./torchjitexport.json',
                                                data_input='./torchjitexport.data',
                                                model_quantization_cfg='./torchjitexport.quantization.cfg',
                                                dataset='/home/hjlee/FairMOT/Dataset/dataset.txt')
            if ret != 0:
                print('hybrid_quantization_step2 failed!')
                exit(ret)
            print('done')

            # Export RKNN model
            print('--> Export RKNN model')
            ret = self.rknn.export_rknn(opt.make_hybrid2)
            if ret != 0:
                print('Export rknn failed!')
                exit(ret)
            print('done')

            self.rknn.release()
            exit(-1)

        if opt.run_rknn != '':
            self.rknn = RKNN()

            # Load RKNN model
            print('--> Loading model')
            ret = self.rknn.load_rknn(path=opt.run_rknn)
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')

            #init runtime environment
            print('--> Init runtime environment')
            # #ret = self.rknn.init_runtime() # PC Simulator
            ret = self.rknn.init_runtime(target='rk1808',
                                         device_id='13171fbb0ecb2508',
                                         perf_debug=False) # Edge Device
            
            #오류 메세지 출력
            if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
            print('done')

            # self.rknn.eval_perf(loop_cnt=100)
            # self.rknn.release()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def Convert_ONNX(self, model): 
        # set the model to inference mode 
        model.eval() 

        # Let's create a dummy input tensor  
        # dummy_input = torch.randn(1, 3, 608, 1088)
        # dummy_input = torch.randn(1, 3, 480, 864)
        # dummy_input = torch.randn(1, 3, 320, 576)
        # dummy_input = torch.randn(1, 3, 160, 288)
        w = self.opt.img_size[0]
        h = self.opt.img_size[1]
        dummy_input = torch.randn(1, 3, h, w)

        # Export the model   
        torch.onnx.export(model,                        # model being run 
            dummy_input,                                # model input (or a tuple for multiple inputs) 
            self.opt.make_onnx,                         # where to save the model  
            export_params=True,                         # store the trained parameter weights inside the model file 
            opset_version=11,                           # the ONNX version to export the model to 
            #do_constant_folding=True,                  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],               # the model's input names 
            output_names = ['hm', 'wh', 'id', 'reg'])   # the model's output names 
            #keep_initializers_as_inputs=True) 
        print(" ") 
        print('Model has been converted to ONNX') 

    def post_process(self, dets, meta): ##
        torch.set_printoptions(profile="full")

        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)

        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)

        return dets[0]

    def merge_outputs(self, detections): ##
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1

        # print(self.model)
        # from torchsummary import summary
        from pytorch_model_summary import summary


        ## 220824 KETI_MUF syh check model param storage
        # print(im_blob.shape)
        #
        # summary(self.model, im_blob, batch_size=-1, show_input=False, show_hierarchical=False, print_summary=True,
        #         max_depth=1, show_parent_layers=False)
        # summary(,,input_size=(3,288,160))
        # raise KeyboardInterrupt

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            if self.opt.run_onnx != '':
                ort_inputs = {self.session.get_inputs()[0].name: to_numpy(im_blob)}
                outputs = self.session.run(None, ort_inputs)

                hm = torch.Tensor(outputs[0]).sigmoid_()
                wh = torch.Tensor(outputs[1])
                id_feature = torch.Tensor(outputs[2])
                id_feature = F.normalize(id_feature, dim=1)
                reg = torch.Tensor(outputs[3]) if self.opt.reg_offset else None

            elif self.opt.run_rknn != '':
                # st = time.time()
                # im_blob = im_blob.permute(0, 2, 3, 1) # from NCHW to NHWC
                time3 = time.time()

                outputs = self.rknn.inference(inputs=[to_numpy(im_blob)], data_type='float32', data_format='nchw')
                time4 = time.time()
                # print("inference time: ", time4-time3)

                if 0:
                    fp = open('python_value_inference_220530.txt', 'w')
                    np.set_printoptions(threshold=sys.maxsize)
                    
                    for j in range(40):
                        for k in range(72):
                            fp.write(str(outputs[0][0][0][j][k]))
                            fp.write("\n")
                    for j in range(4):
                        for k in range(40):
                            for l in range(72):
                                fp.write(str(outputs[1][0][j][k][l]))
                                fp.write("\n")
                    for j in range(64):
                        for k in range(40):
                            for l in range(72):
                                fp.write(str(outputs[2][0][j][k][l]))
                                fp.write("\n")
                    for j in range(2):
                        for k in range(40):
                            for l in range(72):
                                fp.write(str(outputs[3][0][j][k][l]))
                                fp.write("\n")
                    # [fp.write(np.array_str(output)) for output in outputs]
                    fp.close()

                if 0:
                    print("cpp value test")
                    img_width_qt = int(self.opt.img_size[0] / 4)
                    img_height_qt = int(self.opt.img_size[1] / 4)
                    
                    hm_size = 1 * 1 * img_height_qt * img_width_qt
                    wh_size = 1 * 4 * img_height_qt * img_width_qt
                    id_size = 1 * 64 * img_height_qt * img_width_qt
                    reg_size = 1 * 2 * img_height_qt * img_width_qt

                    file_path = '/home/hjlee/cpp_value/mot16_ep10_{}{}_cpp_value.txt'.format(self.opt.img_size[0], self.opt.img_size[1])
                    fp = open(file_path, 'r')
                    lines_org = fp.readlines()
                    print(len(lines_org))

                    lines = []

                    for line in lines_org:
                        line = float(line.strip())
                        lines.append(line)

                    hm_list = lines[:hm_size]
                    wm_list = lines[hm_size:hm_size + wh_size]
                    id_list = lines[hm_size + wh_size: hm_size + wh_size + id_size]
                    reg_list = lines[hm_size + wh_size + id_size:]

                    hm_array = np.array(hm_list, dtype=np.float32)
                    hm_result = hm_array.reshape(1, 1, img_height_qt, img_width_qt)

                    wm_array = np.array(wm_list, dtype=np.float32)
                    wm_result = wm_array.reshape(1, 4, img_height_qt, img_width_qt)

                    id_array = np.array(id_list, dtype=np.float32)
                    id_result = id_array.reshape(1, 64, img_height_qt, img_width_qt)

                    reg_array = np.array(reg_list, dtype=np.float32)
                    reg_result = reg_array.reshape(1, 2, img_height_qt, img_width_qt)

                    outputs = [0, 0, 0, 0]
                    outputs[0] = hm_result
                    outputs[1] = wm_result
                    outputs[2] = id_result
                    outputs[3] = reg_result

                hm = torch.Tensor(outputs[0]).sigmoid_()
                np.set_printoptions(threshold=sys.maxsize)
                wh = torch.Tensor(outputs[1])
                id_feature = torch.Tensor(outputs[2])
                id_feature = F.normalize(id_feature, dim=1)
                reg = torch.Tensor(outputs[3]) if self.opt.reg_offset else None

            else:
                output = self.model(im_blob)[-1]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                id_feature = output['id']
                id_feature = F.normalize(id_feature, dim=1)
                reg = output['reg'] if self.opt.reg_offset else None

            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        # self.rknn.eval_perf()
        # self.rknn.release()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis0

        # for i in range(0, dets.shape[0]):
        #     bbox = dets[i][0:4]
        #     cv2.rectangle(img0, (bbox[0], bbox[1]),
        #                   (bbox[2], bbox[3]),
        #                   (0, 255, 0), 2)
        # cv2.imshow('dets', img0)
        # cv2.waitKey(0)
        #id0 = id0-1



        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
            
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb): ##
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb): ##
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb): ##
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
