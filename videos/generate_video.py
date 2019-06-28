import cv2
import os
import glob

def imgs2video(imgs_dir, save_name):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (2688, 1512))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.jpg'))

    for i in range(len(imgs)):
        imgname = os.path.join(imgs_dir, '{:07d}.jpg'.format(i))
        frame = cv2.imread(imgname)
        video_writer.write(frame)

    video_writer.release()

if __name__=='__main__':
    imgs_dir = '/media/ubuntu/45860a09-77fc-4f27-8cf3-5739e384e61d/huangw/MOT/datasets/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000137_00458_v'
    # imgs_dir = 'images/DJI_0001'
    save_name = 'visDrone/uav0000137_00458_v.avi'
    imgs2video(imgs_dir, save_name)