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

def decode_png(bytes, channels=1):
    return tf.image.decode_png(bytes, channels=channels)

def _process_images_and_labels(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])
    label = decode_png(example['label'])

    return {'input': input, 'label': label}

description_input_label = {
    'input': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.string, default_value='')
}

src_path = r"D:\ADAXI\Datasets\increment\replay_images_and_labels\19"


src_files = glob.glob( os.path.join(src_path,'*.tfrecord') )

for file_ in src_files:
    idx = 1
    file_name = file_[:-9]
    new_path = file_name+"_out"
    img_new_path = os.path.join(new_path, 'image')
    label_new_path = os.path.join(new_path, 'label')
    
    create_folders(img_new_path)
    create_folders(label_new_path)

    parsed_data = tf.data.TFRecordDataset(file_).map(_process_images_and_labels)
    next_ = parsed_data.make_one_shot_iterator().get_next()
    train_input = next_["input"]
    train_label = next_["label"]
    
    
    with tf.Session() as sess:
       
        while True:
            try:
                input_, label_ = sess.run([train_input, train_label])
                cv2.imwrite( os.path.join(img_new_path, str(idx).zfill(5)+".png"), input_[...,::-1] )
                cv2.imwrite( os.path.join(label_new_path, str(idx).zfill(5)+".png"), label_ )
                print(idx)
                idx+=1
            except:
                print ("out of data")
                break
