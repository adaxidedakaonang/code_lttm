import numpy as np
from PIL import Image
import os
import shutil
import math
import time
import random
from tqdm import tqdm
import flickrapi
import urllib.request
import sys

def psnr(img1, img2, down_sample=False):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _shape1 = img1.shape
    _shape2 = img2.shape
    if not len(_shape1)==(len(_shape2)):
        print("In-correct shape.")
        return 0
    if down_sample:
        r, c = _shape1
        if len(_shape1)>2:
            img1 = img1[:r//4, :,:1]
            img2 = img2[:r//4, :,:1]
        else:
            img1 = img1[:r//4, :,]
            img2 = img2[:r//4, :,]
    mse = np.mean((img1-img2) ** 2)+0.001
    score = 20 * math.log10(255.0/math.sqrt(mse))
    return score

def get_dict(path_name, class_name, img_dir_name):
    full_path = os.path.join(path_name, class_name+".npy")
    dst_imgs_dict = {}
    dst_imgs = os.listdir(img_dir_name)
    if not os.path.isfile(full_path):
        print(f"Dict not exists, creating it.")
        for _ in tqdm(dst_imgs):
            img = np.array(Image.open( os.path.join(img_dir_name, _) ) )
            img_shape = img.shape
            r, c = img_shape[0], img_shape[1]
            if len(img_shape) == 2:
                small_content = img[r//2, :]
            elif len(img_shape)==3:
                small_content = img[r//2, :, 0]
            dst_imgs_dict[_] = [[r,c], np.array(small_content)]
        np.save( full_path , dst_imgs_dict)
    else:
        
        dst_imgs_dict = np.load(full_path, allow_pickle=True).item()
        len_dict = len(dst_imgs_dict.keys())
        len_img = len(dst_imgs)
        assert len_dict==len_img, f"Wrong dict !!! {len_dict} and {len_img}"

        print("Dict exists, loading it.")
    return dst_imgs_dict

# src_dir = r"C:\ADAXI\Download_flickr\airplane"
dst_dir = r"C:\ADAXI\Replay_Data_tfrecord\20_out\image"
dst_dir = sys.argv[2]
# dst_dir = r"C:\ADAXI\demo_dataset\src"
############################# Be careful !!!!!!!!!!!!!!!!
if_down_sample = False
#################################
api_key = "e5aa9cf217fbfd9edbf4f23f430a2116"
secret_key = "732222cdbb82a108"

# Flickr api access key
flickr = flickrapi.FlickrAPI(api_key, secret_key, cache=True)

# Names of the classes, used to search images in Flicks
# keywords = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
#             "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", 
#             "potted plant", "sheep", "sofa", "train", "tvmonitor"]
keywords = "tv monitor"
keywords = sys.argv[1]
keywords = keywords.replace("_", " ")


# src_img_path = src_dir
dst_img_path = dst_dir
tmp_dir = r"./tmp"
# dst_imgs = os.listdir(dst_img_path)

print(f"Loading  {keywords} images.")

dst_imgs_dict = get_dict(path_name=tmp_dir, class_name=keywords, 
                         img_dir_name=dst_img_path)

if_delete = False
start_point = 0

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
curr_point = start_point
add_num = 0
t1 = time.time()
for photo in flickr.walk(tag_mode='any',
                             text=keywords,
                             extras='url_c',
                             sort='relevance'):

    if len(os.listdir(dst_img_path))>9999:
        break

    url = photo.get('url_c')
    if url is not None:
        # try block needed since some urls give error and throw an exception
        try:
            save_path = tmp_dir + "/" + keywords +".jpg"
            urllib.request.urlretrieve(url, save_path)
            img1 = Image.open( save_path )
            img1_np = np.array(img1)
            img_shape = img1_np.shape
            r1 = img_shape[0]
            c1 = img_shape[1]
            if len(img_shape) == 2:
                small_content = img1_np[r1//2, :]
            elif len(img_shape)==3:
                small_content = img1_np[r1//2, :, 0]
            # r1, c1 = src_imgs_dict[img_file]
            # r1, c1,_ = img1_np.shape
            dst_start_point = 0
            dst_imgs = os.listdir(dst_img_path)
            dst_end_point = len(dst_imgs)
            dst_curr_point = dst_start_point
            while not dst_curr_point > dst_end_point:

                if dst_end_point>9999:
                    os.remove(os.path.join(tmp_dir, keywords+".npy"))
                    break

                if dst_curr_point==dst_end_point:
                    add_num+=1
                    new_img_name = str(dst_end_point+1).zfill(5)+".jpg"
                    new_file = os.path.join(dst_img_path, new_img_name)
                    shutil.move(save_path, new_file )
                    dst_imgs_dict[new_img_name]=[[r1,c1], small_content]
                    # if dst_end_point%5==0:
                    np.save( os.path.join(tmp_dir, keywords+".npy"), dst_imgs_dict)
                        # print("Save dict.")
                    dst_end_point+=1
                    if_delete = True
                    print(f"{dst_end_point} {keywords}")
                    break

                dst_img_name = dst_imgs[dst_curr_point]
                r2, c2 = dst_imgs_dict[dst_img_name][0]
                # r2, c2,_ = img2_np.shape
                if not (r1==r2 and c1==c2):
                    dst_curr_point+=1

                elif psnr(small_content, 
                            dst_imgs_dict[dst_img_name][1], 
                            down_sample=if_down_sample)>50:
                    # print(f"--------- {dst_end_point} {keywords}")
                    break

                else:
                    dst_curr_point+=1

            if dst_end_point>10000:
                os.remove(os.path.join(tmp_dir, keywords+".npy"))
                break
            if if_delete:
                if_delete = False
            else:
                curr_point+=1
                if_delete = False
        except:
            print("Error at url: ", url)
    