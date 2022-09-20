# !OPENCV_VIDEOIO_DEBUG=1
import cv2
import os
dir_cam = '/media/syh/hdd/data/20220715_keti_with_att'
device_id="cctv"
cam_id="1"
db_proj_name=dir_cam.split('/')[-1]#+"_"+device_id+"_"+cam_id
#vidcap = cv2.VideoCapture(video_loc)
cwds='/media/syh/ssd2/data/keti_220715_cctv_1_att'
db_proj_name=dir_cam.split('/')[-1]#+"_"+device_id+"_"+cam_id

import pathlib
pathlib.Path(os.path.join(cwds,"labels_with_ids/train",db_proj_name,"img1")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(cwds,"labels_with_ids/train",db_proj_name,"img1").replace("labels_with_ids","images")).mkdir(parents=True, exist_ok=True)
label_loc=os.path.join("./labels_with_ids/train/",db_proj_name,"img1")
img_loc=os.path.join("./labels_with_ids/train/",db_proj_name,"img1").replace("labels_with_ids","images")


dirr=os.path.join(dir_cam,'datas/cam-001')
videos=os.listdir(dirr)
print("videos : ",videos)
videos[0]=videos[0].replace('_','.')
dirr=os.path.join(dirr,os.listdir(dirr)[0])

video_loc = os.path.join(dirr,videos[0])

vidcap = cv2.VideoCapture("/media/syh/hdd/data/20220715_keti_with_att/datas/cam-001/20220715150000_mp4/20220715150000.mp4")

count = 0
pwd_=cwds
with open("keti_220715_cctv_1_att.all", "w") as f:
    success=True
    while success:
        success,image = vidcap.read()
        if success ==False:
            # vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print("false")
            success,image = vidcap.read()
        strr=os.path.join(img_loc,"%06d.jpg"%(count+1))
        if os.path.exists(strr.replace('jpg','txt').replace('images','labels_with_ids')) :
            print(strr.replace('jpg','txt').replace('images','labels_with_ids'))
            print(os.path.exists(strr.replace('jpg','txt').replace('images','labels_with_ids')))
            cv2.imwrite(strr, image)   
            f.write(os.path.join(os.path.join(pwd_,img_loc[2:]),"%06d.jpg\n" % (count+1)))
        count += 1