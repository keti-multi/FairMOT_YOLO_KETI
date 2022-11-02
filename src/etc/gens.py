import os
import cv2
from PIL import Image, ImageDraw
test_img = "003539.jpg"

dir1="./"
dir2="./"
dir3="./"
dir4="./"

dir_name_1="jointree_221004"
dir_name_2="jointree_220707"
dir_name_3="keti_220118"
dir_name_4="infoworks_220808"

dir_name_test="jointree_220707_cctv_1"

dir_src = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/"
dir_labels = "labels_with_ids/train/jointree_220707_cctv_1/img1"

labels_dir1= os.path.join(dir_src,dir_name_test,"labels_with_ids/train",dir_name_test,"img1")
imgs_dir1= os.path.join(dir_src,dir_name_test,"images/train",dir_name_test,"img1")


test_len=1 # 100 으로 될 자리

# 검증용
box_color_RGBA  = (0,255,0,255)
fill_color_RGBA = (0,255,0,50)

for i in range(test_len):
    img_1 =  Image.open(os.path.join(imgs_dir1,test_img)).resize((960,540))
    img_2 =  Image.open(os.path.join(imgs_dir1,test_img)).resize((960,540))
    img_3 =  Image.open(os.path.join(imgs_dir1,test_img)).resize((960,540))
    img_4 =  Image.open(os.path.join(imgs_dir1,test_img)).resize((960,540))
    new_image = Image.new('RGB',(1920, 1080), (250,250,250))
    new_image.paste(img_1,(0,0))
    new_image.paste(img_2,(960,0))
    new_image.paste(img_3,(0,540))
    new_image.paste(img_4,(960,540))
    label1 = os.path.join(labels_dir1,test_img.replace("jpg","txt"))
    label2 = os.path.join(labels_dir1, test_img.replace("jpg", "txt"))
    label3 = os.path.join(labels_dir1, test_img.replace("jpg", "txt"))
    label4 = os.path.join(labels_dir1, test_img.replace("jpg", "txt"))

    file = open(label1, "r")
    while True:
        line = file.readline()
        if not line:
            break
        _,f_id,x,y,w,h,_,_,_,_,_,_=line.split(" ")
        f_id=int(f_id)
        x=float(x)
        y=float(y)
        w=float(w)
        h=float(h)
        top=int(1080*(y-h/2))
        left=int(1920*(x-w/2))
        bott=int(1080*(y+h/2))
        right=int(1920*(x+w/2))
        draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
        draw.rectangle((left/2,top/2,right/2,bott/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)

    file.close()
    file = open(label2, "r")
    while True:
        line = file.readline()
        if not line:
            break
        _,f_id,x,y,w,h,_,_,_,_,_,_=line.split(" ")
        f_id=int(f_id)
        x=float(x)+1
        y=float(y)
        w=float(w)
        h=float(h)
        top=int(1080*(y-h/2))
        left=int(1920*(x-w/2))
        bott=int(1080*(y+h/2))
        right=int(1920*(x+w/2))
        draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
        draw.rectangle((left/2,top/2,right/2,bott/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)

    file.close()
    file = open(label3, "r")
    while True:
        line = file.readline()
        if not line:
            break
        _,f_id,x,y,w,h,_,_,_,_,_,_=line.split(" ")
        f_id=int(f_id)
        x=float(x)
        y=float(y)+1
        w=float(w)
        h=float(h)
        top=int(1080*(y-h/2))
        left=int(1920*(x-w/2))
        bott=int(1080*(y+h/2))
        right=int(1920*(x+w/2))
        draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
        draw.rectangle((left/2,top/2,right/2,bott/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)

    file.close()

    file = open(label4, "r")
    while True:
        line = file.readline()
        if not line:
            break
        _,f_id,x,y,w,h,_,_,_,_,_,_=line.split(" ")
        f_id=int(f_id)
        x=float(x)+1
        y=float(y)+1
        w=float(w)
        h=float(h)
        top=int(1080*(y-h/2))
        left=int(1920*(x-w/2))
        bott=int(1080*(y+h/2))
        right=int(1920*(x+w/2))
        draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
        draw.rectangle((left/2,top/2,right/2,bott/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
    file.close()
    new_image.save("merged_image.jpg","JPEG")
    new_image.show()