from PIL import Image

# 读取图像文件
image_path = "TJDR_test_092.png"
image = Image.open(image_path)

# 获取图像的尺寸
width, height = image.size

# 创建一个字典来存储不同像素值的图片
pixel_images = {}

# 遍历图像的每个像素，并保存到相应的图片中
for y in range(height):
    for x in range(width):
        pixel_value = image.getpixel((x, y))
        if pixel_value not in pixel_images:
            # 如果字典中还没有当前像素值对应的图片，创建一个新的图片对象
            pixel_images[pixel_value] = Image.new("RGB", (width, height), color="white")
        # 将当前像素值写入相应的图片中的对应位置
        pixel_images[pixel_value].putpixel((x, y), pixel_value)

# 保存每种像素值对应的图片
for pixel_value, img in pixel_images.items():
    img_path = f"pixel_value_{pixel_value}.png"
    img.save(img_path)

# 打印不同像素值的数量
num_unique_pixels = len(pixel_images)
print("Number of unique pixel values:", num_unique_pixels)
