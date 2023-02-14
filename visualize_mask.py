import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()


        # print(os.path.join(full_path, img_name) )


# image = Image.open('VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg')
# target = np.array(Image.open('VOCdevkit/VOC2012/SegmentationClass/2007_000129.png'))[:, :, np.newaxis]
# cmap = color_map()[:, np.newaxis, :]
# new_im = np.dot(target == 0, cmap[0])
# for i in range(1, cmap.shape[0]):
#     new_im += np.dot(target == i, cmap[i])
# new_im = Image.fromarray(new_im.astype(np.uint8))
# blend_image = Image.blend(image, new_im, alpha=0.8)
# blend_image.save('tmp.jpg')

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import os
    import glob
    import time
    from tqdm import tqdm

    # src_path = r"D:\ADAXI\Datasets\increment\replay_images_and_labels"
    # split_path = os.listdir(src_path)

    # for file_path in split_path:
    #     print(file_path)
    #     start_time = time.time()
    #     full_path = os.path.join( src_path, file_path + "/label/" )
    #     dst_path = os.path.join( src_path, file_path + "/visualize/" )
    #     if not os.path.exists(dst_path):
    #         os.makedirs(dst_path)
    #     img_names = os.listdir(full_path)
    #     print("img nums: " + str(len(img_names)))
    #     pbar = tqdm(img_names, ncols=60)
    #     for img_name in pbar:
    #         target = np.array(Image.open( os.path.join(full_path, img_name) ))[:, :, np.newaxis]
    #         cmap = color_map()[:, np.newaxis, :]
    #         new_im = np.dot(target == 0, cmap[0])
    #         tss = time.time()
    #         for i in range(1, cmap.shape[0]):
    #             new_im += np.dot(target == i, cmap[i])
    #         # print(time.time()-tss)
    #         new_im = Image.fromarray(new_im.astype(np.uint8))
    #         ts = time.time()
    #         new_im.save( os.path.join(dst_path, img_name) )
    #         # break
    #         # print(time.time()-ts)
    #     print("time: {:.2f}".format(time.time()-start_time))


    src_path = r"C:\ADAXI\Replay_Data_tfrecord\01_out\10-10\label"
    # dst_path = r"D:\ADAXI\SDR-master\tmp\visualize_label"
    dst_path = src_path + "_visualize"
    start_time = time.time()
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    img_names = os.listdir(src_path)
    print("img nums: " + str(len(img_names)))
    pbar = tqdm(img_names, ncols=60)
    for img_name in pbar:
        if os.path.isfile(os.path.join(dst_path, img_name[:-3]+"jpg")):
            continue
        target = np.array(Image.open( os.path.join(src_path, img_name) ))[:, :, np.newaxis]
        cmap = color_map()[:, np.newaxis, :]
        new_im = np.dot(target == 0, cmap[0])
        tss = time.time()
        for i in range(1, cmap.shape[0]):
            new_im += np.dot(target == i, cmap[i])
        # print(time.time()-tss)
        new_im = Image.fromarray(new_im.astype(np.uint8))
        ts = time.time()
        new_im.save( os.path.join(dst_path, img_name[:-3]+"jpg") )
        # break
        # print(time.time()-ts)
    print("time: {:.2f}".format(time.time()-start_time))
