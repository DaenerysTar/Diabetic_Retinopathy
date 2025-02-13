import torch
from Classifier import Classifier
from GradingDataset import GradingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import test
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import itertools


if __name__ == "__main__":
    # Prepare data (CHANGE PATH HERE)
    root_dir_train = './DRGrading/Original/Training'
    csv_file_train = './DRGrading/Groundtruths/Grading_Training_Labels.csv'

    root_dir_test = './Grading/Original/Testing'
    csv_file_test = './Grading/Groundtruths/Grading_Testing_Labels.csv'
    bengraham = True

    scale = transforms.Resize((512,512))
    to_tensor = transforms.ToTensor()
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    vertical_flip = transforms.RandomVerticalFlip(p=0.3)
    color_jitter = transforms.ColorJitter(brightness=0.01 * torch.abs(torch.randn(1)).item(),
                                          contrast=0.01 * torch.abs(torch.randn(1)).item(),
                                          saturation=torch.abs(0.01 * torch.randn(1)).item(),
                                          hue=torch.abs(0.01 * torch.randn(1)).item()
                                          )
    # random_perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    random_rotation = transforms.RandomRotation(30)
    center_crop = transforms.CenterCrop(512)
    #composed = transforms.Compose([scale,
                                   #horizontal_flip,
                                   #random_rotation,
                                   #vertical_flip,
                                   #to_tensor]
                                 # )

    composed = transforms.Compose([scale,
                                   to_tensor])
    # train and test set
    train_set = GradingDataset(csv_file_train, root_dir_train, composed, bengraham)
    test_set = GradingDataset(csv_file_test, root_dir_test, composed, bengraham)

    # dataloaders
    batch_size = 4
    n = len(train_set)
    train_indices = list(range(int(0.8 * n)))
    val_indices = list(range(int(0.8 * n) + 1, n))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices), num_workers=0)
    val_loader = DataLoader(train_set, batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_indices), num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    print(
        f"# batches in train set {len(train_loader)} | # batches in val set {len(val_loader)}  # batches in test set {len(test_loader)}")

    # model
    use_cuda = torch.cuda.is_available()
    model_path = "./best_model.pth"
    model = torch.load(model_path)

    if use_cuda:
        model = model.to('cuda')  # 将模型移动到 GPU 上
        model = torch.nn.DataParallel(model)  # 使用 DataParallel 包装模型以便在多GPU上运行
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to('cuda' if use_cuda else 'cpu')
    results_df, results_metrics = test(model, test_loader, criterion, 'cuda' if use_cuda else 'cpu')


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting normalize=True.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('./confusion_matrix.png', format='png')
        
    classes = ['No', 'Mild', 'Moderate', 'Severe', 'PDR']
    plot_confusion_matrix(results_metrics['confusion_matrix'], classes)
    print(results_metrics['AUC'])
    print("done")






