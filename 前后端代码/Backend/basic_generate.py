import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset

from PIL import Image
import numpy as np


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
    '''# CamVid数据集中用于图像分割的所有标签类别
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
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
        # 自己加的
        # albu.Resize(height=480, width=480, always_apply=True),
        albu.Resize(height=1440, width=1440, always_apply=True),
        # 原来的
        # albu.PadIfNeeded(384, 480)
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
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# 图像分割结果的可视化展示
'''def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()'''


# 自己新写的
def visualize(**images):
    """Save images in one row."""
    n = len(images)
    # 确保当前目录存在
    if not os.path.exists('./test_result/'):
        os.makedirs('./test_result/')

    for i, (name, image) in enumerate(images.items()):
        # 为每个图像创建一个子图
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        # 保存图像到文件
        filename = f'./test_result/{name}_{i}.png'
        plt.savefig(filename)
        plt.clf()  # 清除图形对象，为下一个图像做准备


# ---------------------------------------------------------------





if __name__ == '__main__':


    x_test_dir = './data/'  # 修改为你的图片路径
    y_test_dir = './anno/'

    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'

    CLASSES = ['red']
    ACTIVATION = 'sigmoid'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    best_model = torch.load('./manet/red.pth', map_location=torch.device('cpu'))
    # 创建测试数据集
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 从测试集中随机挑选1张图片进行测试
    for i in range(1):
        image, gt_mask = test_dataset[0]
        print(image.shape)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        with Image.open('data/test.png') as img:
            # 计算新的大小为原来的三分之一
            new_size = (1440,1440)
            # 缩放图片
            resized_img = img.resize(new_size)

        image_resized=resized_img

        green = np.array([0, 0, 255])

        # 将掩码为1的部分替换为绿色
        # 假设 image 是一个形状为 (H, W, 3) 的 NumPy 数组

        image_with_green = image_resized.copy()  # 先复制一份图像，避免原地修改

        image_with_green=np.array(image_with_green)
        image_with_green[pr_mask == 1] = green
        image_with_green=np.float32(image_with_green)


        # 保存修改后的图像
        cv2.imwrite('pr_image_with_blue.png', cv2.cvtColor(image_with_green, cv2.COLOR_RGB2BGR))


        palette = [0, 0, 0,  # 索引 0：黑色
                   0, 0, 255,  # 索引 1：绿色
                    ]

        # 创建调色板模式的图像
        image = Image.fromarray(pr_mask.astype(np.uint8))
        image.putpalette(palette)

        # 保存图像
        image.save('pr_image.png')






