import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

src_path = r"D:\ADAXI\SDR-master\tmp\10-10\step_1_src\predict_label"
maximum_val = 11
imgs_path = glob( os.path.join(src_path,"*.png") )

for item in tqdm(imgs_path):
    img = Image.open(item)
    img_np = np.array(img)
    classes = np.unique(img_np)
    if sum(classes>= maximum_val):
        print(item)
print("done")

