import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        albu.Resize(height=480, width=480, always_apply=True),
        # 原来的
        # albu.PadIfNeeded(384, 480)
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                # albu.RandomBrightnessContrast(p=1),
                # albu.RandomGamma(p=1),
            ],
            p=1,
            # p=0.9,
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


def visualize_overlay(image, pr_mask, save_path):
    """将预测的颜色叠加到原始照片上，黑色不叠加，保存结果图片。

    Args:
        image (numpy.ndarray): 原始照片数组
        pr_mask (numpy.ndarray): 预测的颜色掩码数组
        save_path (str): 保存结果图片的路径
    """
    # 定义颜色映射表
    color_map = {
        0: (255, 0, 0),  # 红色
        1: (0, 0, 225),  # 蓝色
        2: (0, 255, 0),  # 绿色
        3: (255, 255, 0)  # 黄色
    }

    # 创建空白图像
    result = np.zeros_like(image)

    # 叠加每个类别的颜色到原始照片上，黑色不叠加
    for class_id, color in color_map.items():
        if class_id == 4:  # 跳过黑色
            continue
        # 将掩码中对应类别的像素点提取出来
        class_mask = (pr_mask == class_id).astype(np.uint8)
        # 将原始照片中对应像素点替换为颜色
        result[class_mask == 1] = color

    # 保存叠加后的图片
    cv2.imwrite(save_path, result)


def generate(model_name, img_name):
    x_test_dir = './data/'
    y_test_dir = './anno/'

    # ENCODER = 'se_resnext50_32x4d'
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    # CLASSES = ['car']
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
    # ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # ---------------------------------------------------------------
    # $# 测试训练出来的最佳模型
    file_name = f"{model_name}/best_model.pth"
    # 加载最佳模型
    best_model = torch.load(file_name, map_location=torch.device('cpu'))

    # 创建测试数据集
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    image, gt_mask = test_dataset[0]

    gt_mask = gt_mask.squeeze()
    gt_mask_array = np.array(gt_mask)

    # 获取不同元素及其数量
    unique_elements, counts = np.unique(gt_mask_array, return_counts=True)

    # 打印不同元素及其数量
    for element, count in zip(unique_elements, counts):
        print(f"Element: {element}, Count: {count}")

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    # pr_mask 是一个形状为 (5, 480, 480) 的独热编码掩码
    # 将 pr_mask 转换为 (480, 480) 格式的整数掩码
    int_pr_mask = np.argmax(pr_mask, axis=0).astype(np.uint8)

    int_gt_mask = np.argmax(gt_mask, axis=0).astype(np.uint8)

    # CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
    palette = [255, 0, 0,  # 红
               0, 0, 255,  # 蓝
               0, 255, 0,  # 绿
               255, 255, 0,  # 黄
               0, 0, 0]  # 黑

    # 创建调色板模式的图像
    image = Image.fromarray(int_pr_mask.astype(np.uint8))  # 这句有问题
    image.putpalette(palette)
    image = image.resize((480, 480))
    # 保存图像
    image.save('pr_image.png')

    image = Image.fromarray(int_gt_mask.astype(np.uint8))
    image.putpalette(palette)
    image = image.resize((480, 480))
    # 保存图像
    image.save('gt_image.png')

    # 加载原始彩色图片和P模式存储的图片
    color_image = Image.open('./data/test.png')
    p_image = Image.open('pr_image.png')

    color_image = color_image.resize((480, 480))
    # 获取P模式存储的图片的调色板
    palette = p_image.getpalette()

    # 创建一个新的RGB图像，大小与彩色图片相同
    new_image = Image.new('RGB', color_image.size)

    # 遍历P模式存储的图片的每个像素，根据像素值在调色板中获取对应的颜色，并将这个颜色赋值到新的RGB图像对应位置上
    for y in range(p_image.height):
        for x in range(p_image.width):
            color_index = p_image.getpixel((x, y))
            if color_index in [0, 1, 2, 3]:
                color = palette[color_index * 3:color_index * 3 + 3]  # 获取RGB颜色值
                new_image.putpixel((x, y), tuple(color))
            else:
                # 对于其他像素值，保持原始彩色图片的颜色
                original_color = color_image.getpixel((x, y))
                new_image.putpixel((x, y), original_color)

    # 保存新的RGB图像
    new_image.save('../frontend/src/assets/result.png')
    new_image.save('result_images/result.png')

    new_image = Image.new('RGB', color_image.size)
    # 遍历P模式存储的图片的每个像素，根据像素值在调色板中获取对应的颜色，并将这个颜色赋值到新的RGB图像对应位置上
    for y in range(p_image.height):
        for x in range(p_image.width):
            color_index = p_image.getpixel((x, y))
            if color_index in [0]:
                color = palette[color_index * 3:color_index * 3 + 3]  # 获取RGB颜色值
                new_image.putpixel((x, y), tuple(color))
            else:
                # 对于其他像素值，保持原始彩色图片的颜色
                original_color = color_image.getpixel((x, y))
                new_image.putpixel((x, y), original_color)
    # 保存新的RGB图像
    new_image.save('../frontend/src/assets/result_red.png')
    new_image.save('result_images/result_red.png')

    new_image = Image.new('RGB', color_image.size)
    # 遍历P模式存储的图片的每个像素，根据像素值在调色板中获取对应的颜色，并将这个颜色赋值到新的RGB图像对应位置上
    for y in range(p_image.height):
        for x in range(p_image.width):
            color_index = p_image.getpixel((x, y))
            if color_index in [1]:
                color = palette[color_index * 3:color_index * 3 + 3]  # 获取RGB颜色值
                new_image.putpixel((x, y), tuple(color))
            else:
                # 对于其他像素值，保持原始彩色图片的颜色
                original_color = color_image.getpixel((x, y))
                new_image.putpixel((x, y), original_color)
    # 保存新的RGB图像
    new_image.save('../frontend/src/assets/result_blue.png')
    new_image.save('result_images/result_blue.png')

    new_image = Image.new('RGB', color_image.size)
    # 遍历P模式存储的图片的每个像素，根据像素值在调色板中获取对应的颜色，并将这个颜色赋值到新的RGB图像对应位置上
    for y in range(p_image.height):
        for x in range(p_image.width):
            color_index = p_image.getpixel((x, y))
            if color_index in [2]:
                color = palette[color_index * 3:color_index * 3 + 3]  # 获取RGB颜色值
                new_image.putpixel((x, y), tuple(color))
            else:
                # 对于其他像素值，保持原始彩色图片的颜色
                original_color = color_image.getpixel((x, y))
                new_image.putpixel((x, y), original_color)
    # 保存新的RGB图像
    new_image.save('../frontend/src/assets/result_green.png')
    new_image.save('result_images/result_green.png')

    new_image = Image.new('RGB', color_image.size)
    # 遍历P模式存储的图片的每个像素，根据像素值在调色板中获取对应的颜色，并将这个颜色赋值到新的RGB图像对应位置上
    for y in range(p_image.height):
        for x in range(p_image.width):
            color_index = p_image.getpixel((x, y))
            if color_index in [3]:
                color = palette[color_index * 3:color_index * 3 + 3]  # 获取RGB颜色值
                new_image.putpixel((x, y), tuple(color))
            else:
                # 对于其他像素值，保持原始彩色图片的颜色
                original_color = color_image.getpixel((x, y))
                new_image.putpixel((x, y), original_color)
    # 保存新的RGB图像
    new_image.save('../frontend/src/assets/result_yellow.png')
    new_image.save('result_images/result_yellow.png')
    time.sleep(1)

# # ---------------------------------------------------------------
# if __name__ == '__main__':
#
#     DATA_DIR = './data/'
#
#     # 测试集
#     x_test_dir = os.path.join(DATA_DIR, 'test')
#     y_test_dir = os.path.join(DATA_DIR, 'testannot')
#
#     # ENCODER = 'se_resnext50_32x4d'
#     ENCODER = 'se_resnet50'
#     ENCODER_WEIGHTS = 'imagenet'
#     # CLASSES = ['car']
#     CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
#     # ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
#     # ---------------------------------------------------------------
#     # $# 测试训练出来的最佳模型
#
#     # 加载最佳模型
#     best_model = torch.load('./best_model.pth', map_location=torch.device('cpu'))
#
#     # 创建测试数据集
#     test_dataset = Dataset(
#         x_test_dir,
#         y_test_dir,
#         augmentation=get_validation_augmentation(),
#         preprocessing=get_preprocessing(preprocessing_fn),
#         classes=CLASSES,
#     )
#
#     image, gt_mask = test_dataset[0]
#
#     gt_mask = gt_mask.squeeze()
#     gt_mask_array = np.array(gt_mask)
#
#     # 获取不同元素及其数量
#     unique_elements, counts = np.unique(gt_mask_array, return_counts=True)
#
#     # 打印不同元素及其数量
#     for element, count in zip(unique_elements, counts):
#         print(f"Element: {element}, Count: {count}")
#
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
#     # pr_mask 是一个形状为 (5, 480, 480) 的独热编码掩码
#     # 将 pr_mask 转换为 (480, 480) 格式的整数掩码
#     int_pr_mask = np.argmax(pr_mask, axis=0).astype(np.uint8)
#
#     int_gt_mask = np.argmax(gt_mask, axis=0).astype(np.uint8)
#
#     # CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']
#     palette = [128, 0, 0,  # 红
#                0, 0, 128,  # 蓝
#                0, 128, 0,  # 绿
#                128, 128, 0,  # 黄
#                0, 0, 0]  # 黑
#
#     # 创建调色板模式的图像
#     image = Image.fromarray(int_pr_mask.astype(np.uint8))  # 这句有问题
#     image.putpalette(palette)
#     image = image.resize((480, 480))
#     # 保存图像
#     image.save('pr_image.png')
#
#     image = Image.fromarray(int_gt_mask.astype(np.uint8))
#     image.putpalette(palette)
#     image = image.resize((480, 480))
#     # 保存图像
#     image.save('gt_image.png')
#
#     with Image.open('182.png') as img:
#         # 计算新的大小为原来的三分之一
#         new_size = (1440, 1440)
#         # 缩放图片
#         image = img.resize(new_size)
#     image=image.resize((480,480))
#     image=np.array(image)
#     print(image.shape)
#     print(int_gt_mask.shape)
#     color_dict = {
#         'red': np.array([255, 0, 0]),
#         'yellow': np.array([255, 255, 0]),
#         'green': np.array([0, 255, 0]),
#         'blue': np.array([0, 0, 255])
#     }
#     mask_color = color_dict['red']
#     image[pr_mask == 1] = mask_color
#     cv2.imwrite('result.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#
#
#
#
#
