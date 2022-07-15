# cd src

# MIX
python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/data_all.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 64 --num_epochs 20 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64


# MOT16 Train dataset
python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16.json' --image-width 544 --image-height 288 --lr 5e-4 --batch_size 8 --num_epochs 10 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 500 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 32 --num_epochs 100 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 64 --num_epochs 100 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64


tensorboard --logdir=/home/keti/FairMOT/exp/mot/all_yolov5s/logs_2022-02-09-15-22/
tensorboard --logdir=/home/hjlee/FairMOT/exp/mot/all_yolov5s/logs_2022-02-24-16-33/


# MOT16-02
python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16-02.json' --image-width 1088 --image-height 608 --lr 5e-4 --batch_size 8 --num_epochs 20 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16-02.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 10 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

# MOT16-04
python train.py mot --gpus 1 --exp_id all_yolov5s --data_cfg '../src/lib/cfg/mot16-04.json' --image-width 288 --image-height 160 --lr 5e-4 --batch_size 16 --num_epochs 1000 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64

