import numpy as np
from PIL import Image
import os
import shutil
import math
import time
import random

src_dir = r"C:\ADAXI\Replay_Data_for_train"
cls_list = os.listdir(src_dir)
############################# Be careful !!!!!!!!!!!!!!!!
delete_img = True
random_shuffle = True
if_down_sample = True
bias = 0
#################################

for cls in cls_list[17:18]:
    img_path = os.path.join( src_dir, cls + "/image" )
    dst_path = os.path.join( src_dir, cls + "/label" ) 
    # img_path = os.path.join( src_dir, cls + "/RECALL/images" )
    # dst_path = os.path.join( src_dir, cls + "/RECALL/labels" ) 
    print("#"*20 + f" {cls}")


    def psnr(img1, img2,down_sample=False):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        if down_sample:
            r,c,_ = img1.shape
            img1 = img1[:r//4, :,:]
            img2 = img2[:r//4, :,:]
        mse = np.mean((img1-img2) ** 2)+0.001
        score = 20 * math.log10(255.0/math.sqrt(mse))
        return score

    imgs = os.listdir(dst_path)
    if random_shuffle:
        random.shuffle(imgs)
    print(f"length: {len(imgs)}")
    start_point = 0
    end_point = len(imgs)-1
    curr_point = start_point + bias
    del_num = 0
    t1 = time.time()
    while curr_point < end_point:
        img_file = imgs[curr_point]
        img1 = Image.open( os.path.join(img_path,img_file[:-3]+"jpg") )
        img1_np = np.array(img1)
        next_point = curr_point+1
        print(f"{img_file}, {len(imgs)-curr_point} images left")
        deleted_imgs = []
        while not next_point > end_point:
            img_name = imgs[next_point][:-3] + "jpg"
            img2 = Image.open( os.path.join(img_path, img_name) )
            img2_np = np.array(img2)
            r1, c1,_ = img1_np.shape
            r2, c2,_ = img2_np.shape
            if not (r1==r2 and c1==c2):
                next_point+=1
                continue
            elif psnr(img1_np, img2_np, down_sample=if_down_sample)>35:
                deleted_imgs.append(img_name)
                # print(f"delete {img_name}")
                del_num+=1
                if delete_img:
                    os.remove( os.path.join(img_path, img_name) )
                    os.remove( os.path.join(dst_path,imgs[next_point]) )
                del imgs[next_point]
                end_point-=1
            else:
                next_point+=1
                continue
        curr_point+=1
        if len(deleted_imgs):
            print(f"{cls}: deleted images: {deleted_imgs}")
    print(f"time: {time.time()-t1}")
    print(f"delete number {del_num}")