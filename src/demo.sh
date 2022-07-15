#python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4
#python demo.py keti --load_model ../models/hrnetv2_w32_imagenet_pretrained.pth  --arch hrnet_18 --reid_dim 128 --conf_thres 0.4 --input-video '/ssd/syh/Multi/20211118/101_1/20211118_20211118132238_20211118132845_132136.mp4' --output-root '../results/results_keti' 

#python demo.py mot --load_model ../models/fairmot_dla34.pth  --conf_thres 0.4 --input-video '/ssd/syh/Multi/20211118/101_1/20211118_20211118132238_20211118132845_132136.mp4' --output-root '../results/results_keti'


#python demo.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --input-video '/ssd/syh/Multi/20211118/101_1/20211118_20211118132238_20211118132845_132136.mp4' --img_size  (2688, 1520)
#python demo.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --input-video '/ssd/syh/Multi/20211118/101_1/20211118-20211118132238-20211118132845-132136_e4dFCo8y.mp4'

## It can be done!!!!!!!!!!! 211126
#python track.py mot --test_mot16 True --arch hrnet_32 --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.5

#python track.py mot --exp_id MOT16_val_all_hrnet_32 --val_mot16 True --arch hrnet_32 --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.5

#211201
#python track.py mot --exp_id MOT16_val_all_hrnet_32 --val_mot16 True --arch hrnet_32 --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.5
#python track.py mot_512_288 --exp_id mot16_hrnet18_w512_h288_test --val_mot16 True --arch 'hrnet_18' --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot_512_288/mot16_hrnet18_w512_h288/model_last.pth  --input_w 512 --input_h 288 --reid_dim 128 --gpus 1

#211203
#python track.py mot --exp_id mot16_hrnet18_w512_h288_test --val_mot16 True --arch 'hrnet_18' --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot_512_288/mot16_hrnet18_w512_h288/model_last.pth  --reid_dim 128 --gpus 1

#211206
#python track_proj.py mot_512_288 --exp_id mot16_hrnet18_w512_h288_test --arch 'hrnet_18' --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot_512_288/mot16_hrnet18_w512_h288/model_last.pth  --reid_dim 128 --gpus 1 --input_w 512 --input_h 288

#python track_proj.py mot --exp_id keti_hrnet18_test --arch 'hrnet_18' --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet18_w1088_h608_reid_dim_128/model_50.pth  --reid_dim 128 --gpus 1 --input_w 1088 --input_h 608

# python track.py mot --exp_id MOT16_val_all_hrnet_32 --val_mot16 True --arch hrnet_32 --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.5



#211207

# 공인인증시험 N1_다중휴먼 인식수
# python demo.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.01 \
# --arch hrnet_32 --input-format 'images' --input-image-dir '/ssd/syh/Multi/mot/MOT16/train/MOT16-04/img1' --output-root '/ssd/syh/Multi/mot/MOT16/train/MOT16-04/outputs'
  # keep
# python test_det.py mot --load_model /home/keti/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.2 --det_thres 0.2 \
# --arch hrnet_32 --input-format 'ImagesAndLabels' \
# --data_cfg '../src/lib/cfg/mot16_keti.json' --K 256

python demo.py mot --load_model /home/keti/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.01 \
--arch hrnet_32 --input-format 'images' --input-image-dir '/home/keti/FairMOT/MOT16-04/img1' --output-root '/home/keti/FairMOT/MOT16-04/outputs'

# 공인인증시험 N2_다중휴먼 검출 정밀도
  # fail
# python track_proj.py mot --exp_id keti_hrnet18_test --arch 'hrnet_18' --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet18_w1088_h608_reid_dim_128/model_50.pth  --reid_dim 128 --gpus 1 --input_w 1088 --input_h 608
#python test_det_prec.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.2 --det_thres 0.2 \
# --arch hrnet_32 --input-format 'ImagesAndLabels' \
# --data_cfg '../src/lib/cfg/keti.json' --K 256
  # fail 11일자 gt 바꿔야함
#python test_det_prec.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/keti_hrnet32/model_last.pth --conf_thres 0.2 --det_thres 0.2 \
# --arch hrnet_32 --input-format 'ImagesAndLabels' \
# --data_cfg '../src/lib/cfg/keti.json' --K 256\
# --gpus '0'

#python test_det_prec.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot_keti_11_2ep/model_last.pth --conf_thres 0.2 --det_thres 0.2 \
# --arch hrnet_32 --input-format 'ImagesAndLabels' \
# --data_cfg '../src/lib/cfg/keti_18.json' --K 256\
# --gpus '0'

#/home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot_keti_11_1ep
# 공인인증시험 N3_다중휴먼 검출 속도
# python demo.py mot --load_model /home/syh/Desktop/work/multi/FairMOT-master/exp/mot/mot16_hrnet32/model_last.pth --conf_thres 0.01 \
# --arch hrnet_32 --input-format 'images' --input-image-dir '/ssd/syh/Multi/mot/MOT16/train/MOT16-04/img1' --output-root '/ssd/syh/Multi/mot/MOT16/train/MOT16-04/outputs'

# 공인인증시험 N4_다중휴먼 추적 정확도







# hrnet
#python demo.py mot --load_model /home/keti/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.01 --arch hrnet_32 --input-format 'images' --input-image-dir '/home/keti/FairMOT/MOT16-02/img1' --output-root '/home/keti/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.01 --arch hrnet_32 --input-format 'images' --input-image-dir '/home/keti/FairMOT/FairMOT/MOT16-02/img1' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --input-format 'images' --input-image-dir '/home/keti/FairMOT/FairMOT/MOT16-02/img1' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --input-video '/home/keti/FairMOT/FairMOT/videos/MOT16-03.mp4' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/src/lib/models/mot16_hrnet32/model_last.pth --conf_thres 0.4 --arch hrnet_32 --input-video '/home/keti/FairMOT/FairMOT/videos/test.mp4' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'

# dla
# python demo.py mot --load_model /home/keti/FairMOT/FairMOT/models/model_dla.rknn --conf_thres 0.4 --input-video '/home/keti/FairMOT/FairMOT/videos/MOT16-03.mp4' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'

#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/models/model_dla.rknn --conf_thres 0.4 --input-video '/home/keti/FairMOT/FairMOT/videos/MOT16-03.mp4' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'
#python demo.py mot --load_model /home/keti/FairMOT/FairMOT/models/model_dla.pth --conf_thres 0.4 --input-video '/home/keti/FairMOT/FairMOT/videos/MOT16-03.mp4' --output-root '/home/keti/FairMOT/FairMOT/MOT16-02/outputs'

# yolov5
#python demo.py mot --load_model /home/keti/FairMOT/FairMOTif/models/fairmot_lite.pth --conf_thres 0.4 --arch yolo --input-video '/home/keti/FairMOT/FairMOTif/videos/MOT16-03.mp4' --output-root '/home/keti/FairMOT/FairMOTif/MOT16-02/yolov5'



######################### # FairMOT 모델 변환
# hrnet_w18
# run demo
python demo.py mot --load_model ../models/fairmot_hrnet_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4

# make onnx
python demo.py mot --load_model ../models/fairmot_hrnet_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4 --gpus -1 --make-onnx ../models/hrnet_w18.onnx
# run onnx
python demo.py mot --load_model ../models/fairmot_hrnet_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4 --gpus -1 --run-onnx ../models/hrnet_w18.onnx
# make rknn
python demo.py mot --load_model ../models/fairmot_hrnet_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4 --gpus -1 --make-rknn ../models/hrnet_w18.rknn
# run rknn
python demo.py mot --load_model ../models/fairmot_hrnet_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4 --gpus -1 --run-rknn ../models/hrnet_w18.rknn

# yolov5s
# run demo
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4
python demo.py mot --load_model ../models/model_last.pth --arch yolo --reid_dim 64 --conf_thres 0.4
# make onnx  // pip install torch==1.8.0 torchvision==0.9.0 / pip install torch==1.7.1 torchvision==0.8.2
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 544 --image-height 320 --make-onnx ../models/yolov5.onnx
# run onnx
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --run-onnx ../models/yolov5.onnx
# make rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --make-rknn ../models/yolov5.rknn
# run rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --run-rknn ../models/yolov5.rknn
# run rknn input image
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --input-format 'images' --input-image-dir '/data/Dataset/MOT16/images/test/MOT16-03/img1' --output-root '/home/hjlee/FairMOT/demos' --run-rknn ../models/yolov5.rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --input-format 'images' --input-image-dir '/data/Dataset/Test' --output-root '/data/Dataset/Test' --run-rknn ../models/yolov5.rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 1088 --image-height 608 --input-format 'images' --input-image-dir '/data/Dataset/Test' --output-root '/data/Dataset/Test1088608' --run-rknn ../models/yolov5.rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 864 --image-height 480 --input-format 'images' --input-image-dir '/data/Dataset/Test' --output-root '/data/Dataset/Test864480' --run-rknn ../models/yolov5.rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 576 --image-height 320 --input-format 'images' --input-image-dir '/data/Dataset/Test' --output-root '/data/Dataset/Test576320' --run-rknn ../models/yolov5.rknn
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --input-format 'images' --input-image-dir '/data/Dataset/Test' --output-root '/data/Dataset/Test288160' --run-rknn ../models/yolov5.rknn


# make hybrid1
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --make-hybrid1 ../models/yolov5.rknn
# make hybrid2
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --make-hybrid2 ../models/yolov5.rknn



# dla_34
# run demo
python demo.py mot --load_model ../models/model_dla.pth --arch dla_34 --reid_dim 64 --conf_thres 0.4




1088, 608 -> 864, 480 -> 576, 320 -> 288, 160 -> 256, 128
train.py33 jde.py354 opts.py242 multitracker.py361


# yolov5 tracking evaluation
python track.py mot --test_mot16 True --load_model ../models/fairmot_yolov5s.pth --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus -1 --run-rknn ../models/yolov5.rknn




python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1
python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --run-rknn ../models/yolov5.rknn


# 220504
python demo.py mot --load_model ../models/fairmot_yolov5s.pth --arch yolo --reid_dim 64 --conf_thres 0.4 --gpus -1 --image-width 288 --image-height 160 --input-format 'images' --input-image-dir '/data/Dataset/MOT16/images/train/MOT16-04/img1' --output-root '/home/hjlee/old/FairMOT/demos' --run-rknn ../models/yolov5.rknn