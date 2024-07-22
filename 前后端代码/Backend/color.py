import cv2
from PIL import Image


def overlay_images(color_img_path, mask_img_path, output_img_path):
    # 读取彩色图像并调整大小
    color_image = cv2.imread(color_img_path)
    color_image = cv2.resize(color_image, (1440, 1440), interpolation=cv2.INTER_AREA)

    # 读取单色图像
    mask_image = cv2.imread(mask_img_path)
    mask_image = cv2.resize(mask_image, (1440, 1440), interpolation=cv2.INTER_AREA)

    # 将彩色图像从BGR格式转换为RGB格式
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 将单色图像转换为灰度图，然后通过二值化分离绿色部分
    gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)

    # 创建一个绿色掩码，其中绿色部分为白色，其余部分为黑色
    green_mask = cv2.merge([binary_mask, binary_mask, binary_mask]) * 255

    # 将彩色图像的对应区域替换为绿色
    color_image[green_mask == 255] = [0, 255, 0]

    # 将图像从RGB格式转换回BGR格式
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # 保存叠加后的图像
    cv2.imwrite(output_img_path, color_image)




# 使用函数
color_img_path = 'data/test.png'  # 彩色图像路径
mask_img_path = 'pr_image.png'  # 只有黑色和绿色的图像路径
output_img_path = 'path_to_save_output_image.jpg'  # 输出图像保存路径

overlay_images(color_img_path, mask_img_path, output_img_path)