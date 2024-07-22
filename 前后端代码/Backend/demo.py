import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
import os

import cv2
from PIL import Image

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


CLASSES = ['green']

x_test_dir = './data/'  # 修改为你的图片路径
y_test_dir = './anno/'
#best_model = torch.load('./green.pth')
best_model = torch.load('./green.pth', map_location=torch.device('cpu'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(DEVICE)

# 创建测试数据集，这里我们只处理单张图片，不使用数据增强和预处理
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,  # 因为我们只处理单张图片，所以不提供掩码目录
    classes=CLASSES,  # 确保这个列表包含你想检测的类别
)

# 接下来，我们将使用模型进行预测
image, _ = test_dataset[0]  # 获取单张图片
x_tensor = torch.from_numpy(image).unsqueeze(0)
pr_mask = best_model.predict(x_tensor)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())

# 可视化预测结果
palette = [0, 0, 0,  # 索引 0：黑色
           0, 255, 0,  # 索引 1：绿色
           255, 0, 255]  # 添加更多颜色以匹配你的类别

# 创建调色板模式的图像
image = Image.fromarray(pr_mask.astype(np.uint8))
image.putpalette(palette)

# 保存图像
image.save('pr_image.png')