import os
from PIL import Image
import numpy as np
from collections import Counter

def count_gray_values(directory):
    gray_value_counts = Counter()
    n = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # Convert image to grayscale
                        img = img.convert('L')
                        # Convert image to numpy array
                        img_array = np.array(img)
                        # Flatten the array and update the counter
                        gray_value_counts.update(img_array.flatten())
                        print(n)
                        n = n + 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Return the total count of different gray values
    return gray_value_counts

# 使用示例：指定目录路径
directory_path = 'D:/retino_datasets/data_0530_nolight/trainannot'
gray_value_counts = count_gray_values(directory_path)

# 打印每个灰度值的计数
for gray_value, count in gray_value_counts.items():
    print(f"Gray value {gray_value}: {count}")
