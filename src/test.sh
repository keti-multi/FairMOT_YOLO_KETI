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
python -W ignore ./test_det.py mot_att --gpus 0 --exp_id keti_220715_cctv_att_yolov5s_288_160_only_test \
--data_cfg './lib/cfg/keti_220715_cctv.json' \
--image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 \
--save_dir /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_att_5ep_288_160_with_test_only \
 --load_model /media/syh/hdd/checkpoints/FairMOT_YOLO_KETI/exp/mot/keti_220715_cctv_yolov5s_att_5ep_288_160_with_test/model_best_1.pth\
 --det_thres 0.3
