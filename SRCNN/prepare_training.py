import numpy as np
import os
import os.path
import math
from PIL import Image

SUBIMAGE_STRIDE = 14

class DataSet:
    def __init__(self,bicubiced_LR,GT):
        '''
        :param bicubiced_LR:  bicubic预处理后的输入图像，list(Image),Image大小必须相同, 0-255
        :param GT:  高分辨率原始图像，list(Image),Image大小必须相同,0-255
        :return:
        '''
        assert len(bicubiced_LR)==len(GT),(
            "LR 大小: %d, GT 大小: %d" %(len(bicubiced_LR),len(GT))
        )
        self._num_samples = len(bicubiced_LR)
        self._bicubied_LR = np.zeros((len(bicubiced_LR),bicubiced_LR[0].height,bicubiced_LR[0].width,1),dtype=np.float32);
        self._GT = np.zeros((len(bicubiced_LR),GT[0].height,GT[0].width,1),dtype=np.float32);
        for i in range(len(bicubiced_LR)):
            self._bicubied_LR[i] = np.array(bicubiced_LR[i])\
                                       .reshape((bicubiced_LR[0].height,bicubiced_LR[0].width,1))\
                                       .astype(np.float32) * 1.0/255.0
            self._GT[i] = np.array(GT[i]).reshape((GT[0].height,GT[0].width,1)).astype(np.float32) * 1.0/255.0
        self._index_in_epoch = 0
        self._epoch_completed = -1
    @property
    def num_samples(self):
        return self._num_samples
    @property
    def epoch_complete(self):
        return self._epoch_completed
    @property
    def bicubiced_LR(self):
        return self._bicubied_LR
    @property
    def GT(self):
        return self._GT

    def next_batch(self,batch_size):
        assert batch_size>=0
        rst_LR = list()
        rst_GT = list()
        for i in range(batch_size):
            rst_LR.append(self._bicubied_LR[self._index_in_epoch])
            rst_GT.append(self._GT[self._index_in_epoch])
            self._index_in_epoch += 1
            if(self._index_in_epoch == self._num_samples):
                self._index_in_epoch = 0
                self._epoch_completed += 1
        return rst_LR,rst_GT


def getY(image):
    #return image.convert('L')
    #return image.convert('F')
    R,G,B = image.split()
    R_arr = np.array(R,np.float32)
    G_arr = np.array(G,np.float32)
    B_arr = np.array(B,np.float32)
    Y = 0.257*R_arr+0.504*G_arr+0.098*B_arr
    return Image.fromarray(Y)




def read_train_set(train_dir):
    file_list = os.listdir(train_dir)
    bicubiced_LR_patches = list()
    GT_patches = list()
    for file_name in file_list:
        file_path = os.path.join(train_dir, file_name)
        image_src = Image.open(file_path)
        GT = getY(image_src)
        bicubiced_LR = GT.resize((GT.width//2,GT.height//2),Image.BICUBIC).resize((GT.width,GT.height),Image.BICUBIC)
        for h_start in range(0,bicubiced_LR.height-SUBIMAGE_STRIDE,SUBIMAGE_STRIDE):
            for w_start in range(0,bicubiced_LR.width-SUBIMAGE_STRIDE,SUBIMAGE_STRIDE):
                bicubiced_LR_patch = bicubiced_LR.crop((w_start, h_start,
                                                        w_start + 32, h_start + 32))
                GT_patch = GT.crop((w_start + 6, h_start + 6,
                                    w_start + 26, h_start + 26))
                bicubiced_LR_patches.append(bicubiced_LR_patch)
                GT_patches.append(GT_patch)
    train_set = DataSet(bicubiced_LR_patches,GT_patches)
    return train_set


def read_test_set(test_dir):
    file_list = os.listdir(test_dir)
    bicubiced_LR_images = list()
    GT_images = list()
    for file_name in file_list:
        file_path = os.path.join(test_dir, file_name)
        image_src = Image.open(file_path)
        GT = getY(image_src)
        bicubiced_LR = GT.resize((GT.width // 2, GT.height // 2), Image.BICUBIC).resize((GT.width, GT.height),
                                                                                        Image.BICUBIC)
        GT_arr = np.array(GT,np.float32)
        LR_arr = np.array(bicubiced_LR,np.float32)
        LR_arr[LR_arr<=0] = 0
        LR_arr[LR_arr>=255.0] = 255.0
        MSE = np.sum((GT_arr-LR_arr)**2)/GT_arr.shape[0]/GT_arr.shape[1]
        psnr = 10* math.log10(255*255/MSE)
        print('%s bicubic psnr : %f' % (file_path,psnr) )
    bicubiced_LR_images.append(bicubiced_LR)
    GT_images.append(GT.crop((6,6,GT.width-6,GT.height-6)))
    test_set = DataSet(bicubiced_LR_images,GT_images)
    return test_set



if __name__ =='__main__':
    read_test_set('Set5')
    #read_train_set('Set5')