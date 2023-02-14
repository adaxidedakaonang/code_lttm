import os
import glob
import cv2
import numpy as np
import tensorflow as tf

def create_folders(folder_):
    import os
    if not os.path.exists(folder_):
        os.makedirs(folder_)

def decode_jpeg(bytes, channels=3):
    return tf.image.decode_jpeg(bytes, channels=channels, fancy_upscaling=False)


def _process_images(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])

    return input

description_input_label = {
    'input': tf.io.FixedLenFeature([], tf.string)
}

src_path = r"C:\ADAXI\RECALL_pre\RECALL-main\data\replay_images\flickr"


src_files = glob.glob( os.path.join(src_path,'*.tfrecord') )

for file_ in src_files[0:]:
    idx = 1
    file_name = file_[:-9]
    new_path = file_name+"_out"
    img_new_path = os.path.join(new_path, 'image')
    print(file_)
    create_folders(img_new_path)

    parsed_data = tf.data.TFRecordDataset(file_).map(_process_images)
    next_ = parsed_data.make_one_shot_iterator().get_next()

    # train_label = next_["label"]
    
    
    with tf.Session() as sess:
       
        while True:
            try:
                input_= sess.run([next_])
                cv2.imwrite( os.path.join(img_new_path, str(idx).zfill(5)+".jpg"), input_[0][...,::-1] )
                # cv2.imwrite( os.path.join(label_new_path, str(idx).zfill(5)+".png"), label_ )
                # print(idx)
                idx+=1
            except:
                print ("out of data")
                break
