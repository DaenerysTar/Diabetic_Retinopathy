import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

#以下是自己加的内容：
#为了看代码方便，我把忽略的类别的相关代码搬了出来
import torch.nn as nn
import re

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs

class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)
    
def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    
    # 添加权重项
    weights = torch.tensor([5/12, 2/3, 3/5, 19/12]).to(pr.device)

    # Apply weights to the channels
    weighted_pr = pr * weights.view(1, -1, 1, 1)
    weighted_gt = gt * weights.view(1, -1, 1, 1)

    tp = torch.sum(weighted_gt * weighted_pr)
    fp = torch.sum(weighted_pr) - tp
    fn = torch.sum(weighted_gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)

    return score

class DiceLoss(Loss):
    def __init__(
        self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs #在此处设置要忽略的类？？
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

def is_one_hot_encoded(mask):
    # 检查每个像素点在类别维度上是否只有一个类别标签为1
    # 使用 np.any 沿着类别维度检查每个像素点是否只有一个 True
    # 如果所有像素点都只有一个 True，则返回 True，表示是独热编码
    return np.all(np.any(mask, axis=-1) == 1)

# ---------------------------------------------------------------
### 加载数据

class Dataset(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """
    # CamVid数据集中用于图像分割的所有标签类别
    '''CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']'''
    
    # 同济数据集中用于图像分割的所有标签类别
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # 从标签中提取特定的类别 (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        #print('布尔mask', mask)
        #print('布尔masks:', masks.shape)
        '''print('布尔mask:', mask.shape)
        # 检查掩码数组
        if is_one_hot_encoded(mask):
            print("掩码数组符合独热编码条件。")
        else:
            print("掩码数组不符合独热编码条件。")'''
        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
    
    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------
### 图像增强

def get_training_augmentation():
    train_transform = [
        albu.Resize(height=960, width=960, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.05, rotate_limit=45, shift_limit=0.05, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=960, min_width=960, always_apply=True, border_mode=0),
        albu.RandomCrop(height=960, width=960, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
            ],
            p=1,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """调整图像使得图片的分辨率长宽能被32整除"""
    test_transform = [
        
        #自己加的
        #albu.Resize(height=480, width=480, always_apply=True),
        albu.Resize(height=960, width=960, always_apply=True),
        
        #原来的
        #albu.PadIfNeeded(384, 480)
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                #albu.RandomBrightnessContrast(p=1),
                #albu.RandomGamma(p=1),
            ],
            p=1,
        #p=0.9,
        ),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        #预训练模型通常需要输入图像与它们在训练时使用的数据相似。因此，预处理步骤通常包括对图像进行归一化、调整大小、裁剪等操作，以使它们与预训练模型的训练数据相匹配
        #当不使用任何预训练参数时，注释掉下面的代码
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':

    # 数据集所在的目录
    #DATA_DIR = './data/CamVid/'
    DATA_DIR = './data_0530_nolight/'

    # 如果目录下不存在CamVid数据集，则克隆下载
    if not os.path.exists(DATA_DIR):
        print('Loading data...')
        os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
        print('Done!')

    # 训练集
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    # 验证集
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    #ENCODER = 'se_resnext50_32x4d'
    #针对较小的数据集，建议选择较浅的网络
    ENCODER = 'se_resnet50'#之前表现最好
    #ENCODER = 'densenet201'
    ENCODER_WEIGHTS = 'imagenet'
    #CLASSES = ['car']
    #自己选择要注意的类别
    CLASSES = ['red', 'blue', 'green', 'yellow', 'unlabelled'] #和单分类问题不同，对于多分类问题，softmax函数对每一个类别的预测概率加起来为1，需要用到所有5个类，但需要手动修改dice损失函数，以让它们只集中于4个类
    #ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    # 用预训练编码器建立分割模型
    # 使用FPN模型
    # model = smp.FPN(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    # )
    # 使用unet++模型
    
    # model_path和if是自己加的
    #model_path = "./best_model_048.pth"
    #model_path = "./best_model_043_061.pth"
    model_path = "./best_model_u0ii9.pth"
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Loaded the best model from last training session.")
    else:
        # 如果不在best_model的基础上训练，则创建新的模型，如果不使用预训练模型，则设置encoder_weights=None
        model = smp.Unet(
        #model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            #encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
        print("No pre-trained model found. Created a new model.")

    #不使用预训练模型时注释掉这一句
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    #preprocessing_fn = ''

    # 加载训练数据集
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 加载验证数据集
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    #train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    #如NVIDIA RTX A4000 是一款性能较强的GPU，具有16GB的显存和较高的带宽。这使得它能够处理相对较大的batch size和并行工作进程
    # 该显卡CPU数为8，故设置num_workers=8，batch_size才和显存有关
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12)
    
    
    ignore_channels = [len(CLASSES) - 1] #在损失函数计算时忽略类别'unlabelled'
    #loss = smp.utils.losses.DiceLoss(ignore_channels=ignore_channels)
    loss = DiceLoss(ignore_channels=ignore_channels)#自己重写一个Diceloss，忽略无标签类
    
    #此处可以增加别的方法
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=ignore_channels),
    ]

    optimizer = torch.optim.Adam([
        #dict(params=model.parameters(), lr=0.0001),
        #在这里修改学习率，对于ENCODER = 'se_resnext50_32x4d'，在10-4、10-3、10-2之间，10-3较好
        dict(params=model.parameters(), lr=0.0001),# 在best_model基础上训练时，别忘了改
    ])

    # 创建一个简单的循环，用于迭代数据样本
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # 进行40轮次迭代的模型训练
    #max_score = 0
    min_dice_loss = 1
    
    #后加的，为了打印图像
    train_losses = []
    valid_losses = []
    train_ious = []
    valid_ious = []
    
    with open('training_logs_unet++.txt', 'w') as f:# open和write是自己加的，w表示先删除原有的内容再写
        for i in range(0, 100):
            with open('training_logs_unet.txt', 'a') as f:# open和write是自己加的，w表示先删除原有的内容再写
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                f.write('Epoch: {}\n'.format(i))
                f.write('Train logs: dice_loss: {}, iou_score: {}\n'.format(train_logs['dice_loss'], train_logs['iou_score']))
                f.write('Valid logs: dice_loss: {}, iou_score: {}\n\n'.format(valid_logs['dice_loss'], valid_logs['iou_score']))


            #后加的，为了打印图像
            train_losses.append(train_logs['dice_loss'])
            valid_losses.append(valid_logs['dice_loss'])
            train_ious.append(train_logs['iou_score'])
            valid_ious.append(valid_logs['iou_score'])

            # 每次迭代保存下训练最好的模型
            #if max_score < valid_logs['iou_score']:
                #max_score = valid_logs['iou_score']
            if min_dice_loss > valid_logs['dice_loss']:
                min_dice_loss = valid_logs['dice_loss']
                torch.save(model, './best_model_unet.pth')
                print('Model saved!')

            '''if i == 5:
                optimizer.param_groups[0]['lr'] = 1e-6
                print('Decrease decoder learning rate to 1e-6!')'''

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

            #后加的，为了打印图像
            # 绘制损失图表并保存
            plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss vs. Epoch')
            plt.savefig('loss_plot_unet.png')  # 保存为 loss_plot.png 文件
            plt.show()

            # 绘制 IoU 分数图表并保存
            plt.figure()
            plt.plot(train_ious, label='Training IoU Score')
            plt.plot(valid_ious, label='Validation IoU Score')
            plt.xlabel('Epoch')
            plt.ylabel('IoU Score')
            plt.legend()
            plt.title('IoU Score vs. Epoch')
            plt.savefig('iou_plot_unet.png')  # 保存为 iou_plot.png 文件
            plt.show()