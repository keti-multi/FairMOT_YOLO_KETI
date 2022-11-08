import os
import cv2
from PIL import Image, ImageDraw

dir1="./"



dir_name_1="20220118_keti"

labels_dir1= "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt"
imgs_dir1= "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/out"

test_len=100 # 100 으로 될 자리

# 검증용
box_color_RGBA  = (0,255,0,255)
fill_color_RGBA = (0,255,0,50)

for i in range(test_len):
    # out_gt = open(os.path.join("/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt","%06d.txt"%(i+1)), "w")
    #test_img="%06d.jpg" % (i+1)
    test_img=i+1
    test_img="%06d.jpg"% (int(test_img))

    img_1 = Image.open(os.path.join(imgs_dir1,test_img))
    test_img = "%06d.jpg" % (i+1)
    label1 = os.path.join(labels_dir1,test_img.replace("jpg","txt"))
    file = open(label1, "r")
    while True:
        line = file.readline()
        if not line:
            break
        _,f_id,x,y,w,h=line.split(" ")
        f_id=int(f_id)
        # x=float(x)
        # y=float(y)
        # w=float(w)
        # h=float(h)
        # top=int(1080*(y-h/2))
        # left=int(1920*(x-w/2))
        # bott=int(1080*(y+h/2))
        # right=int(1920*(x+w/2))
        x=float(x)
        y=float(y)
        w=float(w)
        h=float(h)
        top=int(1080*(y-h/2))
        left=int(1920*(x-w/2))
        bottom=int(1080*(y+h/2))
        right=int(1920*(x+w/2))

        draw = ImageDraw.Draw(img_1, 'RGBA') # RGBA
        draw.rectangle((left,top,right,bottom), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
        # draw.rectangle((1080*y/2,1920*x/2,1080*h/2,1920*w/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)

    file.close()
    # new_image.save(os.path.join(out_dir,"%06d.jpg" % (i+1)),"JPEG")
    img_1.show()