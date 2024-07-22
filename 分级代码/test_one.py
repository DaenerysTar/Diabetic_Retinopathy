import torch
from Classifier import Classifier
from GradingDataset import GradingDataset
from preprocess import preprocess
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import test
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import itertools


if __name__ == "__main__":
    # Prepare data (CHANGE PATH HERE)
    img_path = './IDRiD_296.jpg'

    scale = transforms.Resize((512,512))
    to_tensor = transforms.ToTensor()
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    vertical_flip = transforms.RandomVerticalFlip(p=0.3)
    color_jitter = transforms.ColorJitter(brightness=0.01 * torch.abs(torch.randn(1)).item(),
                                          contrast=0.01 * torch.abs(torch.randn(1)).item(),
                                          saturation=torch.abs(0.01 * torch.randn(1)).item(),
                                          hue=torch.abs(0.01 * torch.randn(1)).item()
                                          )
    random_rotation = transforms.RandomRotation(30)
    center_crop = transforms.CenterCrop(512)
    composed = transforms.Compose([scale,
                                   to_tensor])
    
    image = preprocess(img_path)
    image = Image.fromarray(np.uint8(image))
    image = composed(image)
    # 将其扩展为四维张量 (1 x Channels x Height x Width) 以匹配模型的期望输入
    image = image.unsqueeze(0)

    # model
    use_cuda = torch.cuda.is_available()
    model_path = "./best_model.pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # 如果模型使用了 DataParallel 进行并行化处理，需要通过model.module来获取实际模型
    model = model.module

    # 运行模型进行预测
    with torch.no_grad():  # 禁用梯度计算
         output = model(image)  

    # 处理预测结果
    classes = ['No', 'Mild', 'Moderate', 'Severe', 'PDR']
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]  # 获取预测的类别
    print("Predicted class:", predicted_class)
    