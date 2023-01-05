##220723 test 코드 테스트

#python -W ignore ./test_det.py mot --gpus 0 --exp_id keti_220715_cctv_yolov5s_288_160_test \
#--data_cfg './lib/cfg/keti_220715_cctv.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
##--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_288_160_test # --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_288_160_re/model_5.pth


##220804 info test 코드 테스트
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id infoworks_220725_cctv_yolov5s_288_160_test \
#--data_cfg './lib/cfg/infoworks_220725_cctv.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220725_cctv_yolov5s_288_160_test  --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220725_cctv_yolov5s_5ep_288_160_with_testc/model_best_44.pth\
# --det_thres 0.3


## 220816 detection test 확인
#python -W ignore ./test_det.py mot --gpus 0 --exp_id infoworks_220725_cctv_yolov5s_288_160_test \
#--data_cfg './lib/cfg/infoworks_220808_robot.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test  --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test/model_best_201.pth\
# --det_thres 0.3



## 220906 detection test 확인
#python -W ignore ./test_det.py mot_att --gpus 0 --exp_id infoworks_220725_cctv_yolov5s_288_160_test \
#--data_cfg './lib/cfg/infoworks_220808_robot.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test  --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/infoworks_220808_robot_yolov5s_5ep_288_160_with_test/model_best_201.pth\
# --det_thres 0.3

## 220921
#python -W ignore ./test_det.py mot_att --gpus 0 --exp_id keti_220715_cctv_att_yolov5s_288_160_only_test \
#--data_cfg './lib/cfg/keti_220715_cctv.json' \
#--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_att_5ep_288_160_with_test_only \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_att_5ep_288_160_with_test/model_best_1.pth\
# --det_thres 0.3


# ## 221021
#WIDTH=576
#HEIGHT=320
#EXP_NAME=jointree_221004_cctv_all_keti_plus_yolov5s_att_5ep_${WIDTH}_${HEIGHT}_only_test
#LOAD_EXP=jointree_221004_cctv_all_keti_plus_yolov5s_att_5ep_${WIDTH}_${HEIGHT}_with_tests
#TEST_EPOCH=12
#DATASET=./lib/cfg/MUF_demo_data_all_test.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3
#
#WIDTH=576
#HEIGHT=320
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#LOAD_EXP=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_only_test
#
#TEST_EPOCH=21
#DATASET=./lib/cfg/MUF_demo_data_all_test.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3

#WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#LOAD_EXP=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_only_test
#
#TEST_EPOCH=1
#DATASET=./lib/cfg/MUF_demo_data_all_test.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3

# WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#LOAD_EXP=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_only_test_only_mot16
#
#TEST_EPOCH=7
#DATASET=./lib/cfg/mot16.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3
#
# WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#LOAD_EXP=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_only_test_only_mot16
#
#TEST_EPOCH=4
#DATASET=./lib/cfg/mot16.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3
#

#  WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#LOAD_EXP=MUF_gate_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_only_test_only_all_gate_
#
#TEST_EPOCH=10
#DATASET=./lib/cfg/MUF_demo_data_all_gate_data_test.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3

#WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#LOAD_EXP=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
#EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_only_test_only_jointree_220707
#
#TEST_EPOCH=4
#DATASET=./lib/cfg/MUF_demo_data_all_test_jointree_cctv_only.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_best_${TEST_EPOCH}.pth\
# --det_thres 0.3


## 221108 det test
# WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
##MUF_anonymous_only_accred_MOT_plus_yolov5n_288_160_reiddim_16_with_tests_retraining
#LOAD_EXP=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests_retraining
#EXP_NAME=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_retraining_test_only
#
#TEST_EPOCH=1
#DATASET=./lib/cfg/MUF_demo_data_all_anonymous_four.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_last.pth\
# --det_thres 0.3

## 2211114 트레이닝 결과 테스트
#
#WIDTH=288
#HEIGHT=160
### WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
### yolo == yolov5s, yolov5n==yolov5n
#BACKBONE=yolov5n
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
##MUF_anonymous_only_accred_MOT_plus_yolov5n_288_160_reiddim_16_with_tests_retraining
#LOAD_EXP=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests_retraining
#EXP_NAME=MUF_anonymous_only_accred_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_retraining_test_only
#
#TEST_EPOCH=1
#DATASET=./lib/cfg/MUF_demo_data_all_anonymous_four.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_last.pth\
# --det_thres 0.3

### 221114 추가트레이닝 학습된거 테스트
#WIDTH=576
#HEIGHT=320
#BACKBONE=yolov5n
#EPOCH=1000
#REID_DIM=16
#ETC=_reiddim_${REID_DIM}
#LOAD_EXP=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
#EXP_NAME=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_test_only
#
#TEST_EPOCH=15
#DATASET=./lib/cfg/MUF_demo_data_all_for_keti_demo_without_error_just_test.json
#
#python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
#--data_cfg ${DATASET} \
#--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
#--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
# --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_last.pth\
# --det_thres 0.3

#230102 feature 추출

WIDTH=576
HEIGHT=320
BACKBONE=yolov5n
EPOCH=1000
REID_DIM=16
ETC=_reiddim_${REID_DIM}
LOAD_EXP=MUF_demo_data_all_for_keti_demo_without_error_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_with_tests
EXP_NAME=230102_MUF_demo_data_all_for_keti_demo_without_error_feature_check_${BACKBONE}_${WIDTH}_${HEIGHT}${ETC}_test_only

TEST_EPOCH=15
DATASET=./lib/cfg/MUF_demo_data_all_for_keti_demo_without_error_just_test_jointree.json

python -W ignore ./test_det.py mot --gpus 0 --exp_id ${EXP_NAME} \
--data_cfg ${DATASET} \
--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim ${REID_DIM} \
--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
 --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${LOAD_EXP}/model_last.pth\
 --det_thres 0.3
