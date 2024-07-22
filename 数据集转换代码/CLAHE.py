import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def apply_CLAHE_albumentations(input_image_path, output_image_path):
    # 读取输入图片
    image = cv2.imread(input_image_path)

    # OpenCV读取的是BGR格式，转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 定义CLAHE增强
    transform = A.Compose([
        A.CLAHE(clip_limit=1.0,p=1),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ToTensorV2()
    ])

    # 应用增强
    augmented = transform(image=image)
    clahe_image = augmented['image']

    # 将图像从Tensor转换回numpy数组并调整通道顺序
    clahe_image = clahe_image.permute(1, 2, 0).numpy()

    # 将图像从RGB格式转换回BGR格式
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_RGB2BGR)

    # 保存新生成的图片
    cv2.imwrite(output_image_path, clahe_image)


# 示例使用
input_image_path = 'test_input_b.png'
output_image_path = 'output.png'
apply_CLAHE_albumentations(input_image_path, output_image_path)
