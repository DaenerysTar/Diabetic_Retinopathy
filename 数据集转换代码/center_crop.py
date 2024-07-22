from PIL import Image
import os


def center_crop(image_path):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的尺寸
    width, height = image.size

    # 如果宽度和高度相同，则不进行剪裁
    if width == height:
        return None  # 不需要剪裁，返回None

    # 计算输出尺寸为宽和高中的较小值
    output_size = min(width, height)

    # 计算剪裁后的位置
    left = (width - output_size) / 2
    top = (height - output_size) / 2
    right = (width + output_size) / 2
    bottom = (height + output_size) / 2

    # 剪裁图片
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


def batch_center_crop(directory):
    # 循环处理目录中的所有图片文件
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            # 构造文件路径
            file_path = os.path.join(directory, filename)

            # 居中剪裁图片
            cropped_image = center_crop(file_path)

            # 只有当图片被剪裁时才保存
            if cropped_image:
                # 保存覆盖原图片
                cropped_image.save(file_path)


# 定义处理图片的目录
directory = "D:/retino_datasets/data_0530_nolight/valannot"

# 居中剪裁目录中的所有图片
batch_center_crop(directory)
