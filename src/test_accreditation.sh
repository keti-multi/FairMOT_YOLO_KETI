#
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

# 1.
## 추가학습해서 평가 -> 추가학습완료 3번과 함께


# 2.
python test_det_result.py mot

# 3.

# 4. MOTA 분석 코드
python eval_mot.py

