import os
import shutil
from tqdm import tqdm

def rename_file_in_folder(src_file_path, dst_file_path=None, name_length=None):
    dst_file_path = src_file_path + "_rename"
    if not os.path.exists(dst_file_path):
        os.makedirs(dst_file_path)
    src_files = os.listdir(src_file_path)
    total_num = len(src_files)
    if not name_length:
        name_length = len(str(total_num).split())
    idx = 1
    for file in tqdm(src_files):
        src_file = os.path.join( src_file_path, file )
        file_format = file[-4:]
        dst_file = os.path.join( dst_file_path, str(idx).zfill(name_length)+file_format)
        shutil.copy(src_file, dst_file)
        idx+=1



if __name__ == "__main__":
    bath_path = r"C:\ADAXI\Replay_Data_tfrecord"
    cls_list = [str(i).zfill(2)+"_out/image" for i in range(1,10)]
    cls_list += [str(i)+"_out/image" for i in range(10,21)]
    # cls_list = ["16_out/image"]
    cls_list = cls_list[17:18]
    print(cls_list)

    for cls in cls_list:
        file_path = os.path.join( bath_path, cls )
        print(cls)
        rename_file_in_folder(file_path, name_length=5)