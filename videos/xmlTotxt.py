import xml.etree.ElementTree as ET
import pickle
import os
from collections import defaultdict
# import listdir, getcwd from os.path
# import join
classes = ["person","car"]#红绿灯检测
name_box_id = defaultdict(list)
def convert_annotation(img_path,sub_path):
    in_file = open(sub_path)
    # out_file = open('/home/******/darknet/scripts/VOCdevkit/voc/label/%s.txt'%(image_id),'w')#生成txt格式文件
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes :
            continue
        if cls == 'person':
            cls = 'people'
        xmlbox = obj.find('bndbox')
        box_info =",%d,%d,%d,%d,%s"%(int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text), str(cls))
        name_box_id[img_path].append(box_info)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')
def main(base_img_dir):

    img_list = os.listdir(base_img_dir)
    img_list.sort()
    for img in img_list:
        sub_path=os.path.join(base_img_dir,img)
        filepath, tmpfilename = os.path.split(sub_path)
        shotname, extension = os.path.splitext(tmpfilename)
        if extension == '.xml':
            img_name = shotname+'.jpg'
            img_path = os.path.join(filepath,img_name)
            convert_annotation(img_path, sub_path)
            # name_box_id[sub_path].append(box_info)

if __name__=="__main__":
    base_img_dir = "/media/ubuntu/45860a09-77fc-4f27-8cf3-5739e384e61d/huangw/MOT/deep_sort_yolov3-master/videos/images_label"
    output_path = '/media/ubuntu/45860a09-77fc-4f27-8cf3-5739e384e61d/huangw/MOT/deep_sort_yolov3-master/videos/images_label/label.txt'
    if os.path.exists(output_path):
        os.remove(output_path)
    base_img_dir_list = os.listdir(base_img_dir)
    base_img_dir_list.sort()
    for sub_list in base_img_dir_list:
        if sub_list !='uav_DJI_0004':
            sub_path = os.path.join(base_img_dir, sub_list)
            main(sub_path)
    f = open(output_path, 'w')
    for key in name_box_id.keys():
        box_info_key = name_box_id[key]
        for info in box_info_key:
            f.write(key)
            f.write(info)
            f.write('\n')
    f.close()

