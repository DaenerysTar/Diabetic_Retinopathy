import numpy as np
import os
from PIL import Image, ImageOps

def create_gray_mapping():
    """
    创建一个颜色到灰度值的映射表。
    
    Returns:
        dict: 颜色到灰度值的映射表。
    """
    # 定义颜色和对应的灰度值
    color_to_gray = {
        (128, 0, 0): 0,   # 红色映射到灰度值0
        (0, 0, 128): 1,   # 蓝色映射到灰度值1
        (0, 128, 0): 2,   # 绿色映射到灰度值2
        (128, 128, 0): 3, # 黄色映射到灰度值3
        (0, 0, 0): 4,     # 黑色（其他）映射到灰度值4
    }
    
    # 确保映射表中的灰度值数量与类别数量相匹配
    num_classes = len(color_to_gray)
    
    return color_to_gray, num_classes

def convert_to_grayscale(image, gray_mapping):
    image_array = np.array(image)
    converted_image = np.zeros(image_array.shape[:2], dtype=np.uint8)
    for color, gray_value in gray_mapping.items():
        mask = np.all(image_array == np.array(color), axis=-1)
        converted_image[mask] = gray_value
    result_image = Image.fromarray(converted_image)
    return result_image


# 原始图像所在的文件夹
original_images_folder = './why'
# 目标保存文件夹
save_folder = './whyannot'

# 确保保存文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历原始图像文件夹中的所有文件
for filename in os.listdir(original_images_folder):
    # 构建完整的文件路径
    original_image_path = os.path.join(original_images_folder, filename)
    
    # 检查文件是否是图像
    if os.path.isfile(original_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        # 读取图像
        image = Image.open(original_image_path)

        # 如果图像是palette模式，转换为RGB模式
        # 当图像是palette模式时，直接使用Image.open打开图像会导致颜色信息丢失，
        # 因为Image.open默认不会保留palette信息。
        # 需要使用ImageOps.expand来将palette图像转换为RGB模式，这样才能正确处理每个像素的颜色索引。
        if image.mode == 'P':
            image = image.convert('RGB')
            print(image.mode)
        
        # 创建灰度映射表
        color_to_gray, num_classes = create_gray_mapping()
        
        # 转换为灰度图像
        result_image = convert_to_grayscale(image, color_to_gray)
        print(result_image.mode)
        
        # 获取不带扩展名的原始文件名
        filename = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # 构建灰度图像的保存路径
        grayscale_path = os.path.join(save_folder, f'{filename}.png')
        
        # 保存灰度图像
        result_image.save(grayscale_path)
        print(f'Saved grayscale image to {grayscale_path}')