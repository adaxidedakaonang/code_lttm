import numpy as np
from PIL import Image
import os
import shutil
import math
import time
import random
from tqdm import tqdm

def psnr(img1, img2,r, c, down_sample=False):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _shape1 = img1.shape
    _shape2 = img2.shape
    if not len(_shape1)==(len(_shape2)):
        return 0
    if down_sample:
        if len(_shape1)>2:
            img1 = img1[:r//4, :,:1]
            img2 = img2[:r//4, :,:1]
        else:
            img1 = img1[:r//4, :,]
            img2 = img2[:r//4, :,]
    mse = np.mean((img1-img2) ** 2)+0.001
    score = 20 * math.log10(255.0/math.sqrt(mse))
    return score


src_dir = r"C:\ADAXI\Download_flickr\potted plant"
dst_dir = r"C:\ADAXI\Replay_Data_tfrecord\16_out\image"

############################# Be careful !!!!!!!!!!!!!!!!
if_down_sample = True
#################################

src_img_path = src_dir
dst_img_path = dst_dir

src_imgs = os.listdir(src_img_path)
dst_imgs = os.listdir(dst_img_path)

# src_imgs_dict = {}
dst_imgs_dict = {}
print("Loading images.")
# for _ in tqdm(src_imgs):
#     img = Image.open( os.path.join(src_img_path, _) )
#     r,c,cnl = np.array(img).shape
#     src_imgs_dict[_] = [r,c]

for _ in tqdm(dst_imgs):
    img = Image.open( os.path.join(dst_img_path, _) )
    img_shape = np.array(img).shape
    r, c = img_shape[0], img_shape[1]
    dst_imgs_dict[_] = [r,c]
print("Finish loading images.")

start_point = 0
end_point = len(src_imgs)-1
curr_point = start_point
add_num = 0
t1 = time.time()
if_delete = False
while curr_point <= end_point:
    img_file = src_imgs[curr_point]
    img1 = Image.open( os.path.join(src_img_path,img_file) )
    img1_np = np.array(img1)
    img_shape = img1_np.shape
    r1 = img_shape[0]
    c1 = img_shape[1]
    # r1, c1 = src_imgs_dict[img_file]
    # r1, c1,_ = img1_np.shape
    print(f"{img_file}, {len(src_imgs)-curr_point} images left")
    added_imgs = []
    dst_start_point = 0
    dst_imgs = os.listdir(dst_img_path)
    dst_end_point = len(dst_imgs)
    dst_curr_point = dst_start_point
    while not dst_curr_point > dst_end_point:

        if dst_end_point>9999:
            break

        if dst_curr_point==dst_end_point-1:
            added_imgs.append(img_file)
            # print(f"delete {img_name}")
            add_num+=1
            new_img_name = str(dst_end_point+1).zfill(5)+".jpg"
            
            new_file = os.path.join(dst_img_path, new_img_name)
            shutil.move(os.path.join(src_img_path,img_file), new_file )
            dst_imgs_dict[new_img_name]=[r1,c1]
            del src_imgs[curr_point]
            dst_end_point+=1
            end_point-=1
            if_delete = True
            break

        dst_img_name = dst_imgs[dst_curr_point]
        r2, c2 = dst_imgs_dict[dst_img_name]
        # r2, c2,_ = img2_np.shape
        if not (r1==r2 and c1==c2):
            dst_curr_point+=1

        elif psnr(img1_np, 
                    np.array(Image.open( os.path.join(dst_img_path, dst_img_name)) ), 
                    r1, c1, down_sample=if_down_sample)>35:
            break

        else:
            dst_curr_point+=1

    if dst_end_point>10000:
        break
    if if_delete:
        if_delete = False
    else:
        curr_point+=1
        if_delete = False
    if len(added_imgs):
        print(f"Added images: {added_imgs} {src_dir}")
print(f"time: {time.time()-t1}")
print(f"Add number {add_num}")