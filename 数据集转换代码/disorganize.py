import os
import shutil
import random
# A目录中的图片会随机打乱顺序后保存到C目录，B目录中与A目录中图片同名的图片（掩码）也会相应重命名后保存到D目录中。
def rename_and_shuffle_images(A_dir, B_dir, C_dir, D_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(C_dir):
        os.makedirs(C_dir)
    if not os.path.exists(D_dir):
        os.makedirs(D_dir)

    # 获取A目录中的所有文件名
    A_files = sorted([f for f in os.listdir(A_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # 打乱A目录中的文件名顺序
    random.shuffle(A_files)

    # 遍历打乱后的文件，重新命名并保存到C目录和D目录
    for i, a_file in enumerate(A_files):
        # 生成新的文件名
        new_name = f"{i:03d}.png"

        # 拷贝并重命名A目录中的图片到C目录
        a_source_path = os.path.join(A_dir, a_file)
        a_dest_path = os.path.join(C_dir, new_name)
        shutil.copy(a_source_path, a_dest_path)

        # 在B目录中寻找同名图片，并重命名保存到D目录
        b_source_path = os.path.join(B_dir, a_file)
        if os.path.exists(b_source_path):
            b_dest_path = os.path.join(D_dir, new_name)
            shutil.copy(b_source_path, b_dest_path)
            print(f"Copied {a_file} to {a_dest_path} and corresponding mask to {b_dest_path}")
        else:
            print(f"Warning: Corresponding file for {a_file} not found in {B_dir}")

# 示例使用
A_directory = 'D:/retino_datasets/data_all_no_light/val'  # A目录路径
B_directory = 'D:/retino_datasets/data_all_no_light/valannot'  # B目录路径
C_directory = 'D:/retino_datasets/data_0530_nolight/val'  # C目录路径
D_directory = 'D:/retino_datasets/data_0530_nolight/valannot'  # D目录路径
# A目录中的图片会随机打乱顺序后保存到C目录，B目录中与A目录中图片同名的图片（掩码）也会相应重命名后保存到D目录中。
rename_and_shuffle_images(A_directory, B_directory, C_directory, D_directory)
