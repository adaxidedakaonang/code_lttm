import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import shutil

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

root_path = r"C:\ADAXI\Replay_Data"
dst_path = r"C:\ADAXI\Replay_Data_for_train/threshold_0.1"
# lbl_tag = r"label_sdr_10-10_non_reach_0.2"
lbl_tag = r"RECALL/labels"
# img_tag = "image"
img_tag = r"RECALL/images"
random.seed(123)
img_num = 500
class_path = os.listdir(root_path)
for cls_path in class_path[10:]:
    print(cls_path)
    labels_path = os.path.join( os.path.join( cls_path, lbl_tag ) )
    imgs_path = os.path.join( os.path.join( cls_path, img_tag ) )
    labels = glob.glob( os.path.join(root_path, labels_path+"/*.png") )
    random.shuffle(labels)
    for lbl in tqdm(labels[:img_num]):
        lbl_dst_path = os.path.join(dst_path, cls_path + "/label")
        img_dst_path = os.path.join(dst_path, cls_path + "/image")
        file_name = os.path.split(lbl)[1]
        lbl_dst_file = os.path.join(lbl_dst_path, file_name)
        img_dst_file = os.path.join(img_dst_path, file_name[:-3] + "jpg")
        create_folder(lbl_dst_path)
        create_folder(img_dst_path)
        shutil.copy(lbl,lbl_dst_file)
        shutil.copy( os.path.join(root_path, imgs_path+"/"+file_name[:-3]+"jpg"), img_dst_file )
        
        
        

    
    