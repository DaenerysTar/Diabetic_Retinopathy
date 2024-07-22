import os
from PIL import Image


def merge_masks(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 文件夹名称与对应的灰度值
    lesion_types = {
        'EX': 0,
        'SE': 1,
        'HE': 2,
        'MA': 3
    }

    # 获取列表中第一个文件夹的所有文件，假设每个子文件夹中文件数目和名称是一致的
    sample_folder = os.path.join(base_dir, 'EX')
    sample_folder = sample_folder.replace(os.sep, '/')
    print(sample_folder)
    filenames = os.listdir(sample_folder)

    for filename in filenames:
        # 创建一个新图像，初始化所有像素值为4
        print(filename)
        print(sample_folder)
        mask_path = os.path.join(sample_folder, filename)
        mask_path = mask_path.replace(os.sep, '/')
        print(mask_path)
        with Image.open(mask_path) as img:
            print(img.mode)
            new_mask = Image.new('L', img.size, 4)

        # 合并掩码
        for lesion, value in lesion_types.items():
            mask_path = os.path.join(base_dir, lesion,  filename)
            mask_path = mask_path.replace(os.sep, '/')
            with Image.open(mask_path) as img:
                pixels = img.load()
                new_pixels = new_mask.load()

                for x in range(img.width):
                    for y in range(img.height):
                        if pixels[x, y] == 255:  # 白色表示掩码
                            new_pixels[x, y] = value
        # 保存新的掩码图片
        new_mask.save(os.path.join(output_dir, filename))


# 设定源文件夹和目标文件夹路径
base_dir = 'D:/糖尿病眼底病变数据集/DDR_original/valid/segmentation label'
output_dir = 'D:/糖尿病眼底病变数据集/DDR_processed/valannot'

# 执行合并操作
merge_masks(base_dir, output_dir)
