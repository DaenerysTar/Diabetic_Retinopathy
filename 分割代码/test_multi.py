# 这段代码用于测试某训练好的模型，得出Dice和Iou
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

#以下是自己加的内容：
#为了看代码方便，我把忽略的类别的相关代码搬了出来

# ---------------------------------------------------------------
### 加载数据

class Dataset(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """
    # CamVid数据集中用于图像分割的所有标签类别
    '''CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']'''
    
    # 同济数据集中用于图像分割的所有标签类别
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # 从标签中提取特定的类别 (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        #print('布尔mask', mask)
        #print('布尔masks:', masks.shape)
        '''print('布尔mask:', mask.shape)
        # 检查掩码数组
        if is_one_hot_encoded(mask):
            print("掩码数组符合独热编码条件。")
        else:
            print("掩码数组不符合独热编码条件。")'''
        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
    
    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------
### 图像增强

def get_validation_augmentation():
    """调整图像使得图片的分辨率长宽能被32整除"""
    test_transform = [
        
        #自己加的
        #albu.Resize(height=480, width=480, always_apply=True),
        albu.Resize(height=960, width=960, always_apply=True),
        
        #原来的
        #albu.PadIfNeeded(384, 480)
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                #albu.RandomBrightnessContrast(p=1),
                #albu.RandomGamma(p=1),
            ],
            p=1,
        #p=0.9,
        ),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        #预训练模型通常需要输入图像与它们在训练时使用的数据相似。因此，预处理步骤通常包括对图像进行归一化、调整大小、裁剪等操作，以使它们与预训练模型的训练数据相匹配
        #当不使用任何预训练参数时，注释掉下面的代码
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':

    # 数据集所在的目录
    #DATA_DIR = './data/CamVid/'
    DATA_DIR = './data_0530/'

    # 如果目录下不存在CamVid数据集，则克隆下载
    if not os.path.exists(DATA_DIR):
        print('Loading data...')
        os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
        print('Done!')

    # 验证集
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    #针对较小的数据集，建议选择较浅的网络
    ENCODER = 'se_resnet50'#之前表现最好
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
    DEVICE = 'cuda'
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    model = torch.load('./best_model_manet.pth')

    # 加载测试数据集
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 该显卡CPU数为8，故设置num_workers=8，batch_size才和显存有关
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)
    
    #ignore_channels = [1, 2, 3, 4]  # 红色
    #ignore_channels = [0, 2, 3, 4]  # 蓝色
    #ignore_channels = [0, 1, 3, 4]  # 绿色
    ignore_channels = [0, 1, 2, 4]  # 黄色
    
    #ignore_channels = [len(CLASSES) - 1] #在损失函数计算时忽略类别'unlabelled',CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
    loss = smp.utils.losses.DiceLoss(ignore_channels=ignore_channels)

    #此处可以增加别的方法
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=ignore_channels),
        smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=ignore_channels),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    
    with open('test_unet++.txt', 'a') as f:# open和write是自己加的，w表示先删除原有的内容再写
        test_logs = test_epoch.run(test_loader)
        f.write('dice_loss: {}, iou_score: {}, accuracy: {}\n\n'.format(test_logs['dice_loss'], test_logs['iou_score'], test_logs['accuracy']))