import cv2
import numpy as np
import os

def adjust_brightness(image_path, output_image_path, target_brightness=194, max_ratio=2.0, apply_filter=True):
    print(f"Processing image: {image_path}")
    # 读取输入图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be read.")

    # 将图像从BGR格式转换为HSV格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # 计算图像中心和内接圆的半径
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = min(center[0], center[1])

    # 创建掩膜以标识内接圆的9/10大小的同心圆
    smaller_circle_mask = np.zeros_like(v_channel)
    cv2.circle(smaller_circle_mask, center, int(radius * 9 / 10), 255, -1)

    # 创建另一个掩膜以标识整个内接圆
    full_circle_mask = np.zeros_like(v_channel)
    cv2.circle(full_circle_mask, center, radius, 255, -1)

    # 计算9/10内接圆的平均亮度
    smaller_masked_v_channel = cv2.bitwise_and(v_channel, v_channel, mask=smaller_circle_mask)
    total_brightness = np.sum(smaller_masked_v_channel)
    num_pixels = np.count_nonzero(smaller_circle_mask)
    if num_pixels == 0:
        raise ValueError("No pixels in the smaller circle. Cannot calculate brightness.")
    current_brightness = total_brightness / num_pixels

    # 计算亮度调整比例
    brightness_ratio = min(target_brightness / current_brightness, max_ratio)

    # 调整整个内接圆的像素亮度
    full_masked_v_channel = cv2.bitwise_and(v_channel, v_channel, mask=full_circle_mask)
    full_masked_v_channel_adjusted = np.clip(full_masked_v_channel * brightness_ratio, 0, 255).astype(np.uint8)

    # 将调整后的内接圆区域重新合并到图像中
    v_channel_final = np.where(full_circle_mask == 255, full_masked_v_channel_adjusted, v_channel)

    if apply_filter:
        v_channel_final = cv2.GaussianBlur(v_channel_final, (5, 5), 0)

    # 合并HSV通道
    adjusted_hsv_image = cv2.merge([h_channel, s_channel, v_channel_final])
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

    # 保存新生成的图片
    cv2.imwrite(output_image_path, adjusted_image)
    print(f"Processed image saved as {output_image_path}")

def process_directory(input_dir, output_dir, target_brightness=194, max_ratio=2.0, apply_filter=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image_path = image_path.replace(os.sep, '/')#把\替换成/
            output_image_path = os.path.join(output_dir, filename)
            output_image_path = output_image_path.replace(os.sep, '/')  # 把\替换成/
            adjust_brightness(image_path, output_image_path, target_brightness, max_ratio, apply_filter)

# 示例使用
input_directory = 'D:/retino_datasets/data_all_upload/val_old'  # 输入图片文件夹路径，不能有中文
output_directory = 'D:/retino_datasets/data_all_upload/val'  # 输出图片文件夹路径，不能有中文
process_directory(input_directory, output_directory, target_brightness=194, max_ratio=2.0, apply_filter=True)
