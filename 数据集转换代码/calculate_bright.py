import cv2
import numpy as np

def calculate_average_brightness_in_circle(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)

    # 检查图像是否成功读取
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be read.")

    # 将图像从BGR格式转换为HSV格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 获取图像的高度和宽度
    height, width = hsv_image.shape[:2]

    # 计算圆心和半径
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])

    # 创建圆形掩膜
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # 分离V通道
    v_channel = hsv_image[:,:,2]  # V通道是HSV图像的最后一个通道

    # 应用掩膜
    v_channel_masked = cv2.bitwise_and(v_channel, v_channel, mask=mask)

    # 计算亮度的平均值（V通道）
    average_brightness = np.sum(v_channel_masked) / np.count_nonzero(mask)

    return average_brightness

# 示例使用
image_path = 'input.png'
average_brightness = calculate_average_brightness_in_circle(image_path)
print(f"Average Brightness: {average_brightness}")
