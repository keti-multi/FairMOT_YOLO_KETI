import cv2

vidcap = cv2.VideoCapture('/media/syh/ssd2/data/keti_220715_gate/20220721105235.mp4')
count = 0

with open("../../data/keti_220715_gate.train", "w") as f:
    success=True
    while success:
        success,image = vidcap.read()
        cv2.imwrite("/media/syh/ssd2/data/keti_220715_gate/images/train/keti_220715_gate/img1/%06d.jpg"%(count+1), image)
        f.write("/media/syh/ssd2/data/keti_220715_gate/images/train/keti_220715_gate/img1/%06d.jpg\n" % (count+1))
        count += 1