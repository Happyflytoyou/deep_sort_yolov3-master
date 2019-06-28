import cv2
import os

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

print(2)
vc=cv2.VideoCapture("visDrone/DJI_0431.MOV")
c=1
dir = "/media/ubuntu/45860a09-77fc-4f27-8cf3-5739e384e61d/huangw/MOT/deep_sort_yolov3-master/videos/images/DJI_0431"
if os.path.exists(dir) is False:
    os.mkdir(dir)
else:
    del_file(dir)
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    if c%4==0:
        cv2.imwrite(dir+'/%07d.jpg'%(c/4),frame)
        cv2.waitKey(1)
    c=c+1
vc.release()