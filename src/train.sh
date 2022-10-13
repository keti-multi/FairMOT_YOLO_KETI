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


EXP_NAME=jointree_221004_gate_1_keti_plus_yolov5s_att_5ep_288_160_with_tests
python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/jointree_221004_gate_keti_plus.json' \
--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
--load_model /media/syh/ssd2/SynologyDrive/03_FairMOT/FairMOT-master/exp/mot/all_yolov5s_MOT16_288160_bs16_ep500/model_5.pth \
--dataset jde  \
--num_att1 32 \
--num_att2 17 \
--num_att3 7 \
--num_att4 17 \
--num_att5 7 \
--num_att6 17 \
--num_att 6