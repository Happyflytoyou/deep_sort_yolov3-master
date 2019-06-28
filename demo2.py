#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image,resize_image
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(model):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    #video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('videos/soccer_01.mp4')
    # video_capture = cv2.VideoCapture('videos/visDrone/short/DJI_0003_01.mov')
    video_capture = cv2.VideoCapture('videos/visDrone/num8_2.mov')
    # video_capture = cv2.VideoCapture('videos/M1304.avi')
    # video_capture = cv2.VideoCapture('videos/uav123_car6.avi')
    #video_capture = cv2.VideoCapture('videos/car/car_11.mp4')


    labels_to_names = {0: 'regions', 1: 'pedestrain', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck',
                   7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('num8_2_v6.mp4', fourcc, 30, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    i=0;
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()
        # i+=1
        # if i%2!=1:
            # continue;
        # image = Image.fromarray(frame)
        image = preprocess_image(frame)
        # boxs, out_classes = yolo.detect_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image,axis=0))
        i = 0
        box_tmp = []
        for box,label,score in zip(boxes[0], labels[0], scores[0]):
            if np.array(score) < 0.5:
                break
            # b = box.astype(int)
            box_tmp.append((box[0:2]).tolist()+(box[2:4]-box[0:2]).tolist())
            i += 1
        # box_tmp = boxes[0][0:i]

        # box_tmp = box_tmp.tolist()
       # print("box_num",len(boxs))
        features = encoder(frame,box_tmp)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(box_tmp, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(np.array(box_tmp), nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print("indices:",indices)
        # print("detection: ",detections[indices[0]]);
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(fps))  

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)


        for j,det in enumerate(detections):
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            # cv2.putText(frame,str(labels_to_names[j]),(int(bbox[0]),int(bbox[1])-35),0,5e-3*200,(143,17,86),2)
        cv2.namedWindow("track result", 0)
        cv2.resizeWindow("track result", 1280, 720)  
        cv2.imshow('track result', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxes) != 0:
                for i in range(0,len(boxes)):
                    list_file.write(str(boxes[i][0]) + ' '+str(boxes[i][1]) + ' '+str(boxes[i][2]) + ' '+str(boxes[i][3]) + ' ')
            list_file.write('\n')
            
       
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'model_data/uav_out15_v5.h5'
    model = models.load_model(model_path,backbone_name='resnet50')
    main(model)

