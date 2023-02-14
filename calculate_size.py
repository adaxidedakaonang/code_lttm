import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

class FilterManager():
    def __init__(self, file, aim_class, size):
        self.file = self._toNumpy(file)
        self.aim_class = aim_class
        self.aim_size = size
        self.actual_size = None
        self.max_size = None
    
    def classExists(self):
        classes, counts = np.unique(self.file, return_counts=True)
        if len(counts)>1:
            self.max_size = np.max(counts[1:])
        for i in range(len(classes)):
            if self.aim_class==classes[i]:
                self.actual_size = counts[i]
                return True
        return False

    def sizeReach(self):
        r,c = self.file.shape
        total_size = r*c
        if self.actual_size / 1.0 / total_size >=self.aim_size:
            return True
        else:
            return False

    def sizeMax(self):
        if self.max_size is not None:
            if not self.actual_size == self.max_size:
                return False
            else:
                return True

    def _toNumpy(self, file):
        if not file is None:
            return np.array(file)
        else:
            return None


if __name__ == "__main__":

    # root_path = r"D:\ADAXI\Datasets\VOC_SDR\PascalVOC12\SegmentationClassAug"
    # labels = [i for i in range(0,21)]
    # labels = [i for i in range(14,17)]
    # imgs = os.listdir(root_path)
    # for label in labels:
    #     nums = 0
    #     total_sizes = 0
    #     for img_path in tqdm(imgs):
    #         img = Image.open( os.path.join(root_path, img_path) )
    #         filter_manager = FilterManager(file=img, aim_class=label, size=0)
    #         if filter_manager.classExists():
    #             total_sizes += filter_manager.actual_size/(img.size[0]*img.size[1])
    #             nums+=1

    #     print(f"Number {nums}, average size of class {label} is {total_sizes/nums}")
    #     print()


    root_path = r"C:\ADAXI\Replay_Data_for_train"
    cls_path = os.listdir(root_path)
    for cls_p in range(len(cls_path)):
        label_path =  os.path.join( root_path, os.path.join(cls_path[cls_p],"label") )
        labels = glob.glob( os.path.join(label_path,"*.png") )
        total_sizes = 0
        nums = len(labels)
        for label in tqdm(labels):
            lbl = Image.open(label)
            filter_manager = FilterManager(file=lbl, aim_class=cls_p+1, size=0)
            if filter_manager.classExists():
                total_sizes += filter_manager.actual_size/(lbl.size[0]*lbl.size[1])
        print(f"Number {nums}, average size of class {cls_p+1} is {total_sizes/nums}")

    # root_path = r"C:\ADAXI\Replay_Data"
    # cls_path = os.listdir(root_path)
    # for cls_p in range(len(cls_path[10:])):
    #     label_path =  os.path.join( root_path, os.path.join(cls_path[cls_p+10],"RECALL/labels") )
    #     labels = glob.glob( os.path.join(label_path,"*.png") )
    #     total_sizes = 0
    #     nums = len(labels)
    #     for label in tqdm(labels):
    #         lbl = Image.open(label)
    #         filter_manager = FilterManager(file=lbl, aim_class=cls_p+11, size=0)
    #         if filter_manager.classExists():
    #             total_sizes += filter_manager.actual_size/(lbl.size[0]*lbl.size[1])
    #     print(f"average size of class {cls_p+11} is {total_sizes/nums}")
    


