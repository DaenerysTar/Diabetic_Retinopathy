from PIL import Image
import numpy as np
import os


def create_gray_mapping():
    """
    创建一个颜色到灰度值的映射表。

    Returns:
        tuple(dict, int): 颜色到灰度值的映射表及类别数量。
    """
    color_to_gray = {
        (128, 0, 0): 0,  # 红色映射到灰度值0
        (0, 0, 128): 1,  # 蓝色映射到灰度值1
        (0, 128, 0): 2,  # 绿色映射到灰度值2
        (128, 128, 0): 3,  # 黄色映射到灰度值3
        (0, 0, 0): 4,  # 黑色（其他）映射到灰度值4
    }

    return color_to_gray, len(color_to_gray)


def create_reverse_mapping(gray_mapping):
    """
    创建一个从灰度值到颜色的映射表。

    Returns:
        dict: 灰度值到颜色的映射表。
    """
    gray_to_color = {gray: color for color, gray in gray_mapping.items()}
    return gray_to_color


def convert_to_color(image, gray_to_color):
    """
    使用灰度到颜色的映射表将灰度图像转换为彩色图像。

    Args:
        image (PIL.Image): 灰度图像。
        gray_to_color (dict): 灰度值到颜色的映射表。

    Returns:
        PIL.Image: 转换后的彩色图像。
    """
    image_array = np.array(image)
    color_image_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for gray, color in gray_to_color.items():
        mask = image_array == gray
        color_image_array[mask] = color

    return Image.fromarray(color_image_array)


# 定义输入输出文件夹
input_folder = 'D:/retino_datasets/data_0530_nolight/testannot'
output_folder = 'D:/retino_datasets/data_0530_nolight/testannot_visible'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 创建颜色到灰度的映射表和反向映射表
color_to_gray, _ = create_gray_mapping()
gray_to_color = create_reverse_mapping(color_to_gray)

# 遍历输入文件夹中的所有灰度图像文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if os.path.isfile(file_path) and filename.lower().endswith('.png'):
        # 读取灰度图像
        grayscale_image = Image.open(file_path).convert('L')

        # 转换灰度图像为彩色图像
        color_image = convert_to_color(grayscale_image, gray_to_color)

        # 保存彩色图像
        save_path = os.path.join(output_folder, filename)
        color_image.save(save_path)
        print(f'Saved color image to {save_path}')
