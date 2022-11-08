import os
import cv2
from PIL import Image, ImageDraw
test_img = "003539.jpg"

dir1="./"
dir2="./"
dir3="./"
dir4="./"


dir_name_1="20220118_keti"
dir_name_2="20220707_jointree"
dir_name_3="20220715_keti"
dir_name_4="20221005_jointree"


dir_name_test="jointree_220707_cctv_1"

dir_src = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터"

# labels_dir1= os.path.join(dir_src,dir_name_test,"labels_with_ids/train",dir_name_test,"img1")
# imgs_dir1= os.path.join(dir_src,dir_name_test,"images/train",dir_name_test,"img1")
#/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220118_keti/datas/cam-001/20220118130000_mp4
labels_dir1= os.path.join(dir_src,dir_name_1,"datas/cam-001/20220118130000_mp4/labels_with_ids")
#/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220707_jointree/datas/cam-001/20220707130000_mp4/labels_with_ids
labels_dir2= os.path.join(dir_src,dir_name_2,"datas/cam-001/20220707130000_mp4/labels_with_ids")
#/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20220715_keti/datas/cam-001/20220715130000_mp4/labels_with_ids
labels_dir3= os.path.join(dir_src,dir_name_3,"datas/cam-001/20220715130000_mp4/labels_with_ids")
#/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/20221005_jointree/datas/cam-001/20221005130000_mp4/labels_with_ids
labels_dir4= os.path.join(dir_src,dir_name_4,"datas/cam-001/20221005130000_mp4/labels_with_ids")

imgs_dir1= os.path.join(dir_src,dir_name_1,"images")
imgs_dir2= os.path.join(dir_src,dir_name_2,"images")
imgs_dir3= os.path.join(dir_src,dir_name_3,"images")
imgs_dir4= os.path.join(dir_src,dir_name_4,"images")


out_dir = "/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/out"

test_len=100 # 100 으로 될 자리

# 검증용
box_color_RGBA  = (0,255,0,255)
fill_color_RGBA = (0,255,0,50)

for i in range(test_len):
    out_gt = open(os.path.join("/media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/gt","%06d.txt"%(i+1)), "w")
    #test_img="%06d.jpg" % (i+1)
    test_img=i+1
    test_img_1 = "%04d.jpg" % (int(test_img))
    test_img="%d.jpg"% (int(test_img))

    img_1 =  Image.open(os.path.join(imgs_dir1,test_img_1)).resize((960,540))
    img_2 =  Image.open(os.path.join(imgs_dir2,test_img)).resize((960,540))
    img_3 =  Image.open(os.path.join(imgs_dir3,test_img)).resize((960,540))
    img_4 =  Image.open(os.path.join(imgs_dir4,test_img)).resize((960,540))
    new_image = Image.new('RGB',(1920, 1080), (250,250,250))
    new_image.paste(img_1,(0,0))
    new_image.paste(img_2,(960,0))
    new_image.paste(img_3,(0,540))
    new_image.paste(img_4,(960,540))
    test_img = "%06d.jpg" % (i)
    label1 = os.path.join(labels_dir1,test_img.replace("jpg","txt"))
    label2 = os.path.join(labels_dir2, test_img.replace("jpg", "txt"))
    label3 = os.path.join(labels_dir3, test_img.replace("jpg", "txt"))
    label4 = os.path.join(labels_dir4, test_img.replace("jpg", "txt"))
    try:
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
            bott=int(1080*(y+h/2))
            right=int(1920*(x+w/2))

            top = y *1080
            left = x * 1920
            bottom = h * 1080
            right = w* 1920
            left,top,right,bottom = left/2,top/2,right/2,bottom/2
            cx = (left+right)/2
            cx /=1920
            cy = (top+bottom)/2
            cy /=1080
            ww = right-left
            ww /=1920
            hh = top-bottom
            hh /=1080
            out_gt.write("0 %s %s %s %s %s\n"%(str(f_id),str(cx),str(cy),str(ww),str(hh)))


            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2,top/2,right/2,bottom/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
            # draw.rectangle((1080*y/2,1920*x/2,1080*h/2,1920*w/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)

        file.close()
    except Exception as e:
        print(e)
        pass
    try:
        file = open(label2, "r")
        while True:
            line = file.readline()
            if not line:
                break
            _,f_id,x,y,w,h=line.split(" ")
            f_id=int(f_id)-41
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
            left,top,right,bottom = left/2 + 960,top/2,right/2 + 960,bottom/2
            cx = (left+right)/2
            cx /=1920
            cy = (top+bottom)/2
            cy /=1080
            ww = right-left
            ww /=1920
            hh = top-bottom
            hh /=1080
            out_gt.write("0 %s %s %s %s %s\n"%(str(f_id),str(cx),str(cy),str(ww),str(hh)))

            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2 + 960,top/2,right/2 + 960,bottom/2), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
        file.close()
    except Exception as e:
        print(e)
        pass
    try:
        file = open(label3, "r")
        while True:
            line = file.readline()
            if not line:
                break
            _,f_id,x,y,w,h=line.split(" ")
            f_id=int(f_id)-78
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
            left,top,right,bottom = left/2,top/2+540,right/2,bottom/2+540
            cx = (left+right)/2
            cx /=1920
            cy = (top+bottom)/2
            cy /=1080
            ww = right-left
            ww /=1920
            hh = top-bottom
            hh /=1080
            out_gt.write("0 %s %s %s %s %s\n"%(str(f_id),str(cx),str(cy),str(ww),str(hh)))

            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2,top/2+540,right/2,bottom/2+540), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
        file.close()
    except:
        pass
    try:
        file = open(label4, "r")
        while True:
            line = file.readline()
            if not line:
                break
            _,f_id,x,y,w,h=line.split(" ")
            f_id=int(f_id)-121
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
            left,top,right,bottom = left/2+960,top/2+540,right/2+960,bottom/2+540
            cx = (left+right)/2
            cx /=1920
            cy = (top+bottom)/2
            cy /=1080
            ww = right-left
            ww /=1920
            hh = top-bottom
            hh /=1080
            out_gt.write("0 %s %s %s %s %s\n"%(str(f_id),str(cx),str(cy),str(ww),str(hh)))
            # draw = ImageDraw.Draw(new_image, 'RGBA') # RGBA
            # draw.rectangle((left/2+960,top/2+540,right/2+960,bottom/2+540), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
        file.close()
    except:
        pass
    out_gt.close()
    new_image.save(os.path.join(out_dir,"%06d.jpg" % (i+1)),"JPEG")
    # new_image.show()