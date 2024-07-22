from PIL import Image
import os

def convert_images_to_png(source_folder, target_folder):
    # 检查目标文件夹是否存在，如果不存在，创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.tif')):
            # 构建完整的文件路径
            file_path = os.path.join(source_folder, filename)
            # 读取图片
            img = Image.open(file_path)
            # 构建目标文件路径
            target_path = os.path.join(target_folder, os.path.splitext(filename)[0] + '.png')
            # 转换格式并保存到目标文件夹
            img.save(target_path, 'PNG')

# 设置源文件夹和目标文件夹的路径
source_folder = 'D:/糖尿病眼底病变数据集/data_old/valannot'
target_folder = 'D:/糖尿病眼底病变数据集/data/valannot'

# 调用函数
convert_images_to_png(source_folder, target_folder)
print('success')
