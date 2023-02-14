import numpy as np
from PIL import Image
import os
import shutil
import math
import time
import random
from tqdm import tqdm

def psnr(img1, img2, down_sample=False):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _shape1 = img1.shape
    _shape2 = img2.shape
    if not len(_shape1)==(len(_shape2)):
        return 0
    if down_sample:
        r, c = _shape1[0], _shape1[1]
        if len(_shape1)>2:
            img1 = img1[:r//4, :,:1]
            img2 = img2[:r//4, :,:1]
        else:
            img1 = img1[:r//4, :,]
            img2 = img2[:r//4, :,]
    mse = np.mean((img1-img2) ** 2)+0.001
    score = 20 * math.log10(255.0/math.sqrt(mse))
    return score

if __name__ == "__main__":

    src_dir = r"C:\ADAXI\Replay_Data_tfrecord"
    # cls_list = os.listdir(src_dir)
    cls_list = [str(i).zfill(2)+"_out" for i in range(1,10)]
    cls_list += [str(i)+"_out" for i in range(10,21)]
    # cls_list = ["04_out"]
    cls_list = cls_list[17:18]

    print(cls_list)
    ############################# Be careful !!!!!!!!!!!!!!!!
    delete_img = True
    random_shuffle = False
    #################################
    for cls in cls_list:
        img_path = os.path.join( src_dir, cls + "/image" )
        # dst_path = os.path.join( src_dir, cls + "/label" ) 
        # img_path = os.path.join( src_dir, cls + "/RECALL/images" )
        # dst_path = os.path.join( src_dir, cls + "/RECALL/labels" ) 
        print("#"*20 + f" {cls}")

        imgs = os.listdir(img_path)
        if random_shuffle:
            random.shuffle(imgs)
        print(f"length: {len(imgs)}")
        imgs_dict = {}
        for _ in tqdm(imgs):
            img = np.array(Image.open( os.path.join(img_path, _) ) )
            img_shape = img.shape
            r,c = img_shape[0], img_shape[1]
            if len(img_shape)==3:
                img = img[:,:,0]
            imgs_dict[_] = [[r,c], img[r//2,:c//2]]
        print("finish loading images")
        start_point = 0
        end_point = len(imgs)-1
        curr_point = start_point
        del_num = 0
        t1 = time.time()
        while curr_point < end_point:
            img_file = imgs[curr_point]
            r1, c1 = imgs_dict[img_file][0]
            content_1 = imgs_dict[img_file][1]
            # r1, c1,_ = img1_np.shape
            next_point = curr_point+1
            # print(f" {cls}  {img_file}, {len(imgs)-curr_point} images left")
            deleted_imgs = []
            while not next_point > end_point:
                img_name = imgs[next_point]
                r2, c2 = imgs_dict[img_name][0]
                # r2, c2,_ = img2_np.shape
                if not (r1==r2 and c1==c2):
                    next_point+=1
                    continue
                elif psnr(content_1, 
                        imgs_dict[img_name][1], 
                        down_sample=False)>50:
                    deleted_imgs.append(img_name)
                    # print(f"delete {img_name}")
                    del_num+=1
                    if delete_img:
                        os.remove( os.path.join(img_path, img_name) )
                    _ = imgs_dict.pop(imgs[next_point])
                    del imgs[next_point]
                    end_point-=1
                else:
                    next_point+=1
                    continue

            # curr_point+=1
            del imgs[curr_point]
            end_point-=1
            if len(deleted_imgs):
                print(f"{cls}: {img_file} deleted images: {deleted_imgs}")
        print(f"time: {time.time()-t1}")
        print(f"delete number {del_num}")