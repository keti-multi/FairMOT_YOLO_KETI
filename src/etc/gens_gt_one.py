import os
import cv2
from PIL import Image, ImageDraw
test_img = "003539.jpg"

dir1="./"
dir2="./"
dir3="./"
dir4="./"


# dir_name_3="20220715_keti"
dir_name_3="20220707_jointree"

dir_src = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터"

#/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220715_keti/datas/cam-001/20220715130000_mp4/labels_with_ids
# labels_dir3= os.path.join(dir_src,dir_name_3,"datas/cam-001/20220715130000_mp4/labels_with_ids")
labels_dir3= os.path.join(dir_src,dir_name_3,"datas/cam-001/20220707130000_mp4/labels_with_ids")

imgs_dir3= os.path.join(dir_src,dir_name_3,"images")


out_dir = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/out_gt"

test_len=100 # 100 으로 될 자리

# 검증용
box_color_RGBA  = (0,255,0,255)
fill_color_RGBA = (0,255,0,50)
out_gt_sum = open("/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/jointree_220707_gt.txt", "w")
for i in range(test_len):
    # out_gt = open(os.path.join("/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt","%06d.txt"%(i+1)), "w")
    #test_img="%06d.jpg" % (i+1)
    test_img=i+1
    test_img_1 = "%04d.jpg" % (int(test_img))
    test_img="%d.jpg"% (int(test_img))

    new_image = Image.open(os.path.join(imgs_dir3,test_img))

    test_img = "%06d.jpg" % (i)
    label3 = os.path.join(labels_dir3, test_img.replace("jpg", "txt"))


    try:
        file = open(label3, "r")
        while True:
            line = file.readline()
            if not line:
                break
            _,f_id,x,y,w,h=line.split(" ")
            f_id=int(f_id)-100
            x=float(x)
            y=float(y)
            w=float(w)
            h=float(h)
            # top=int(1080*(y-h/2))
            # left=int(1920*(x-w/2))
            # bott=int(1080*(y+h/2))
            # right=int(1920*(x+w/2))
            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2,top/2,right/2,bott/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
            top = y *1080
            left = x * 1920
            bottom = h * 1080
            right = w* 1920
            cx = (left+right)/2
            cx /=1920
            cy = (top+bottom)/2
            cy /=1080
            ww = right-left
            ww /=1920
            hh = bottom-top
            hh /=1080
            # out_gt.write("0 %s %f %f %f %f\n"%(str(f_id),cx,cy,ww,hh))
            out_gt_sum.write("%d %s %f %f %f %f\n"%(i+1,str(f_id),cx,cy,ww,hh))


            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2,top/2+540,right/2,bottom/2+540), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
        file.close()
    except:
        pass

    # out_gt.close()
    # new_image.ave(os.path.join(out_dir,"%06d.jpg" % (i+1)),"JPEG")
    # new_image.show()