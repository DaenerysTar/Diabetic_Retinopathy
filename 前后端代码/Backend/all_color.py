import os
import time
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from albumentations import Compose, Resize, Lambda
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Dataset(BaseDataset):
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.ids)

def get_validation_augmentation():
    test_transform = [Resize(height=1440, width=1440, always_apply=True)]
    return Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [Lambda(image=preprocessing_fn), Lambda(image=to_tensor, mask=to_tensor)]
    return Compose(_transform)

def process_color(model_name, color, image, DEVICE):
    file_name = f"{model_name}/{color}.pth"
    best_model = torch.load(file_name, map_location=torch.device('cpu'))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    return color, pr_mask

def save_image(image, mask, color, output_dir):
    mask_color = color_dict[color]
    image_copy = image.copy()
    image_copy[mask == 1] = mask_color
    output_path = os.path.join(output_dir, f'result_{color}.png')
    cv2.imwrite(output_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))





def generate(model_name, img_name):
    x_test_dir = './data/'
    y_test_dir = './anno/'
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['red']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = Dataset(x_test_dir, y_test_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    image, gt_mask = test_dataset[0]

    with Image.open(img_name) as img:
        new_size = (1440, 1440)
        resized_img = img.resize(new_size)
    image_resized = np.array(resized_img)

    output_dir_front = '../frontend/src/assets/'
    output_dir_back = 'result_images/'
    os.makedirs(output_dir_front, exist_ok=True)
    os.makedirs(output_dir_back, exist_ok=True)


    args = [(model_name, color, image, DEVICE) for color in colors]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: process_color(*p), args))

    for color, pr_mask in results:
        save_image(image_resized, pr_mask, color, output_dir_front)
        save_image(image_resized, pr_mask, color, output_dir_back)


    combined_image = image_resized.copy()
    for color, pr_mask in results:
        mask_color = color_dict[color]
        combined_image[pr_mask == 1] = mask_color

    cv2.imwrite('../frontend/src/assets/result.png', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('result_images/result.png', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    time.sleep(1)



model_folder = 'manet'
colors = ['red', 'yellow', 'green', 'blue']
color_dict = {'red': np.array([255, 0, 0]), 'yellow': np.array([255, 255, 0]), 'green': np.array([0, 255, 0]), 'blue': np.array([0, 0, 255])}

# generate('manet', 'test.png')

