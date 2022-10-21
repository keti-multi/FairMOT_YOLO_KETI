WIDTH=576
HEIGHT=320
## WARNING /home/syh/workspace/multi/FairMOT_YOLO_KETI/src/lib/models/yolo.py 참조하여 config 변경
## yolo == yolov5s, yolov5n==yolov5n
BACKBONE=yolov5n
EXP_NAME=MUF_cctv_all_MOT_plus_${BACKBONE}_${WIDTH}_${HEIGHT}_with_tests
EPOCH=100

python -W ignore ./train_test.py mot --gpus 0 --exp_id ${EXP_NAME} --data_cfg  './lib/cfg/MUF_demo_data_all.json' \
--image-width ${WIDTH} --image-height ${HEIGHT} --lr 5e-4 --batch_size 16 --num_epochs ${EPOCH} --wh_weight 0.5 --multi_loss 'fix' --arch ${BACKBONE} --reid_dim 64 \
--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/${EXP_NAME} \
--dataset jde  \
--num_att1 32 \
--num_att2 17 \
--num_att3 7 \
--num_att4 17 \
--num_att5 7 \
--num_att6 17 \
--num_att 6
