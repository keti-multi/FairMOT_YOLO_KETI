python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --image-width 288 --image-height 160 --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --seqs 1 --run-rknn ../models/yolov5.rknn

python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --image-width 288 --image-height 160 --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --seqs 0 --run-rknn ../models/yolov5.rknn

python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --image-width 1088 --image-height 608 --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --seqs 0 --run-rknn ../models/yolov5.rknn

python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --image-width 576 --image-height 320 --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --seqs 0 --run-rknn ../models/yolov5.rknn

python track.py mot --val_mot16 True --load_model ../models/fairmot_yolov5s.pth --image-width 864 --image-height 480 --conf_thres 0.4 --arch yolo --reid_dim 64 --gpus 1 --seqs 0 --run-rknn ../models/yolov5.rknn
