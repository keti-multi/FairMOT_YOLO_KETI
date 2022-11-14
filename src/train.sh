## 220720 데이터 형식 변경후 재 트레인 kiosk, cctv

#python -W ignore ./train.py mot --gpus 0 --exp_id keti_220715_kiosk_yolov5s_160_288_re --data_cfg './lib/cfg/keti_220715_kiosk.json' --image-width 160 --image-height 288 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_kiosk_yolov5s_160_288_re

#python -W ignore ./train.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_288_160_re --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_288_160_re

#python -W ignore ./train.py mot --gpus 0 --exp_id keti_220715_kiosk_yolov5s_160_288_re --data_cfg './lib/cfg/keti_220715_kiosk.json' --image-width 160 --image-height 288 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_kiosk_yolov5s_160_288_test


#python -W ignore ./train.py mot --gpus 0 --exp_id keti_220715_gate_yolov5s_288_160 --data_cfg './lib/cfg/keti_220715_gate.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_gate_yolov5s_288_160


## 220723 test 추가

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_gate_yolov5s_288_160_test


#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_500ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_500ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_last.pth

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_15ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_15ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_15.pth

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_20ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_20ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_20.pth
#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_25ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_25ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_25.pth

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_10ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_10ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_10.pth

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_5ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_5ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth

#python -W ignore ./train_test.py mot --gpus 0 --exp_id keti_220715_gate_yolov5s_5ep_288_160_with_test --data_cfg './lib/cfg/keti_220715_gate_with_test.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_gate_yolov5s_5ep_288_160_test \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth


## 220803 infoworks cctv train
#python -W ignore ./train_test.py mot --gpus 0 --exp_id infoworks_220725_cctv_yolov5s_5ep_288_160_with_test --data_cfg './lib/cfg/infoworks_220725_cctv.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220725_cctv_yolov5s_5ep_288_160_with_testc \
# --load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \

## 220812 infoworks robot train

#python -W ignore ./train_test.py mot --gpus 0 --exp_id infoworks_220808_robot_yolov5s_5ep_288_160_with_test --data_cfg './lib/cfg/infoworks_220808_robot.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test \
# --load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth

### 220823 infoworks gate train
# python -W ignore ./train_test.py mot --gpus 0 --exp_id infoworks_220808_robot_yolov5s_5ep_288_160_with_test --data_cfg './lib/cfg/infoworks_220808_robot.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test \
# --load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth


## 220901 Add Attributes inference test
#EXP_NAME=infoworks_220808_robot_yolov5s_att_5ep_288_160_with_test
#python -W ignore ./train_test.py mot_att --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/keti_220715_cctv.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde_attribute  \
#--num_att 6

## 220920 Add Attributes inference
#EXP_NAME=keti_220715_cctv_yolov5s_att_5ep_288_160_with_test
#python -W ignore ./train_test.py mot_att --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/keti_220715_cctv.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde_attribute  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
###########
###########

#
#EXP_NAME=keti_220715_cctv_fold2_yolov5s_att_5ep_288_160_with_test
#python -W ignore ./train_test.py mot_att --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/keti_220715_cctv_fold2.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde_attribute  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
###########
###########

## 220928 jointree
#EXP_NAME=jointree_220707_cctv_yolov5s_att_5ep_160_288_with_tests
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/jointree_220707_kiosk_fold1.json' \
#--image-width 160 --image-height 288 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
###########
###########

#EXP_NAME=jointree_220707_kiosk_keti_kisok_yolov5s_att_5ep_160_288_with_tests
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/jointree_220707_kiosk_fold1_ketiplus.json' \
#--image-width 160 --image-height 288 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
############
###########
#  220929 jointree cctv + keti

#EXP_NAME=jointree_220707_cctv_keti_kisok_yolov5s_att_5ep_288_160_with_tests
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  #'./lib/cfg/jointree_220707_kiosk_fold1_ketiplus.json' \
#--image-width 160 --image-height 288 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
############

#EXP_NAME=keti_220715_cctv_fold1_yolov5s_att_5ep_288_160_with_test
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/keti_220715_cctv.json.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
#
#EXP_NAME=keti_220715_cctv_fold1_infoadd_yolov5s_att_5ep_288_160_with_test
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/keti_220715_cctv_fold3.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# EXP_NAME=jointree_220707_cctv_1_keti_plus_yolov5s_att_5ep_288_160_with_tests
# python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/jointree_220707_cctv_fold1_ketiplus.json' \
# --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
# --save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
# --dataset jde  \
# --num_att1 32 \
# --num_att2 17 \
# --num_att3 7 \
# --num_att4 17 \
# --num_att5 7 \
# --num_att6 17 \
# --num_att 6


#EXP_NAME=jointree_221004_gate_1_keti_plus_yolov5s_att_5ep_288_160_with_tests
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/jointree_221004_gate_keti_plus.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# 221020 size-wise
# 221024 train done
#WIDTH=576
#HEIGHT=320
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

#  File "/home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/common.py", line 283, in forward
#    return torch.cat(x, self.d)
# RuntimeError: Sizes of tensors must match except in dimension 2. Got 25 and 26 (The offending index is 0)
# model squeeze
## 221024 train FAIL
#WIDTH=720
#HEIGHT=400
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## 221024 train FAIL
#WIDTH=430
#HEIGHT=240
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6


# 221024 train done
#WIDTH=576
#HEIGHT=320
#EXP_NAME=MUF_cctv_all_MOT_plus_yolov5s_5ep_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## 221024 train FAIL
#WIDTH=720
#HEIGHT=400
#EXP_NAME=MUF_cctv_all_MOT_plus_yolov5s_5ep_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
### 221024 train FAIL
#WIDTH=430
#HEIGHT=240
#EXP_NAME=MUF_cctv_all_MOT_plus_yolov5s_5ep_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# 221024 retrain edited
## don't need edge spec
#WIDTH=864
#HEIGHT=480
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6


# 221024 retrain edited
# layer depth 비교군 학습
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## reid dimension to 32
## done
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_32_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 32 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
# reid dimension to 16
# done
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# reiddim 16 w576 h320
#WIDTH=576
#HEIGHT=320
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# reiddim 32 w576 h320
#WIDTH=576
#HEIGHT=320
#BACKBONE=yolov5n
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_32_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 32 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## reiddim 32 w576 h320
## reid dimension to 16
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_gate_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_gate.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
#

### 221108 anonymous data fitting
## reiddim 32 w576 h320
## reid dimension to 16
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_anonymous_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_anonymous_four.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
#
### 221108 anonymous data fitting power fitting
## reiddim 32 w576 h320
## reid dimension to 16
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests
#EPOCH=100
#
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_anonymous_four_only.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde  \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## 221108 anonymous data fitting power fitting
# reiddim 32 w576 h320
# reid dimension to 16
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EXP_NAME=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_reiddim_16_with_tests_retraining
#EPOCH=100
#ETC=_reiddim_${REID_DIM}
#LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_anonymous_four_only_only.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 1 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 16 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_4.pth \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6


# 221112 KETI 상에서 성능 안좋은 것 개선
# KETI 영상에 fitting
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#EXP_NAME=MUF_demo_data_all_for_keti_demo_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
#LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_for_keti_demo.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_4.pth \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

## 221112 576 320
#WIDTH=576
#HEIGHT=320
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#EXP_NAME=MUF_demo_data_all_for_keti_demo_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
##LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_for_keti_demo.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6

# 1088 608
### 221114-1
#WIDTH=1088
#HEIGHT=608
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#EXP_NAME=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
##LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_for_keti_demo_without_error.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6
##

#221114-2
#WIDTH=576
#HEIGHT=320
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#EXP_NAME=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
##LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_for_keti_demo_without_error.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
#--num_att5 7 \
#--num_att6 17 \
#--num_att 6


#221114-3
#WIDTH=288
#HEIGHT=160
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#EXP_NAME=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
##LOAD_EXP=MUF_cctv_all_MOT_plus_yolov5n_288_160_reiddim_16_with_tests
#
#python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all_for_keti_demo_without_error.json' \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
#--dataset jde \
#--num_att1 32 \
#--num_att2 17 \
#--num_att3 7 \
#--num_att4 17 \
--num_att5 7 \
--num_att6 17 \
--num_att 6