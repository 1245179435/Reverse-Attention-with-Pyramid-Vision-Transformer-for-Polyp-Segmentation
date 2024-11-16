import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import os
import  numpy as np
from scipy import misc
import numpy as np
import random
from adapativediceloss import Adaptive_tvMF_DiceLoss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from 融合max import DuAT
from ronghe3 import DuAT
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def horizontal_flip(image):
    return torch.flip(image, [3])  # 水平翻转图像

def vertical_flip(image):
    return torch.flip(image, [2])  # 垂直翻转图像

# def tta_model(model, image):
#     # 原始图像
#     n_image = image
#     # 水平翻转图像
#     h_image = horizontal_flip(image)
#     # 垂直翻转图像
#     v_image = vertical_flip(image)
#
#     # 对每种图像进行预测
#     with torch.no_grad():
#         n_mask = model(n_image)[-1]
#         h_mask = model(h_image)[-1]
#         v_mask = model(v_image)[-1]
#
#     # 恢复翻转的掩码到原始方向
#     h_mask = horizontal_flip(h_mask)  # 再次水平翻转回去
#     v_mask = vertical_flip(v_mask)  # 再次垂直翻转回去
#
#     # 将所有掩码堆叠到一起
#     masks = torch.stack([n_mask, h_mask, v_mask], dim=1)
#
#     # 对每个像素位置进行投票，选取最多类别的结果
#     mean_mask, _ = torch.mode(masks, dim=1)
#     return mean_mask


crition=Adaptive_tvMF_DiceLoss(2)
# crition2=TverskyLoss_Binary()
# 测试函数

def tta_model(model, image):
    # 原始图像
    n_image = image
    # 水平翻转图像
    h_image = horizontal_flip(image)
    # 垂直翻转图像
    v_image = vertical_flip(image)

    # 对每种图像进行预测
    with torch.no_grad():
        n_mask = model(n_image)[-1]
        h_mask = model(h_image)[-1]
        v_mask = model(v_image)[-1]

    # 恢复翻转的掩码到原始方向
    h_mask = horizontal_flip(h_mask)  # 再次水平翻转回去
    v_mask = vertical_flip(v_mask)  # 再次垂直翻转回去

    # 将所有掩码堆叠到一起
    masks = torch.stack([n_mask, h_mask, v_mask], dim=1)

    # 对每个像素位置进行投票，选取最多类别的结果
    mean_mask, _ = torch.mode(masks, dim=1)
    return mean_mask
class CustomDataset(Dataset):
    def __init__(self, img_dir, seg_dir,transform=None):
        """
        img_dir: Path to the image directory.
        seg_dir: Path to the segmentation directory.
        transform: Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        seg_path = os.path.join(self.seg_dir, img_name)  # Assuming the segmentation files are PNGs

        img = Image.open(img_path)
        seg = Image.open(seg_path)
        sample = {'image': img, 'segmentation': seg, "filename": img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define your transform for both image and segmentation
class Transform(object):
    def __init__(self):
        self.image_transform = A.Compose([
            A.Resize(352, 352,interpolation=cv2.INTER_AREA),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.mask_transform = A.Compose([
            A.Resize(352, 352,interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ])

    def __call__(self, sample):
        image, segmentation, img_name = sample['image'], sample['segmentation'], sample["filename"]
        image = np.array(image)
        segmentation = np.array(segmentation)
        segmentation = segmentation.astype(np.uint8)
        image = self.image_transform(image=image)['image']
        segmentation = self.mask_transform(image=segmentation)['image']
        return {'image': image, 'segmentation': segmentation, "filename": img_name}
class Transform1(object):
    def __init__(self):
        self.image_transform =  A.Compose([
            A.Resize(352, 352,interpolation=cv2.INTER_AREA),
            A.Normalize([0.485,0.456,0.406],
            [0.229,0.224,0.225]),
            ToTensorV2()
        ])

        self.mask_transform = A.Compose([
            A.Resize(352, 352,interpolation=cv2.INTER_NEAREST),  # ÄãÐèÒªÊ¹ÓÃ PIL.Image µÄ antialias ²ÎÊýÀŽÊµÏÖ¿¹Ÿâ³Ý
            ToTensorV2()
        ])

        self.augmentation_color = A.Compose([
            A.ColorJitter(brightness=(0.8,1), contrast=(0.8,1), saturation=(0.8,1), hue=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=7, p=0.4),
            A.MotionBlur(blur_limit=7, p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.35),
            A.FancyPCA(alpha=0.1, p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.35),



        ])
        self.augmentation_image = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=25,p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-15, 15), p=0.5),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.4),
            A.PiecewiseAffine(scale=(0.03, 0.05), p=0.4),
            A.ElasticTransform(alpha=3,sigma=50,p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.35),
            A.RandomResizedCrop(height=352, width=352, scale=(0.7, 1.0), p=0.5),
        ])

    def __call__(self, sample):
        image, segmentation, img_name = sample['image'], sample['segmentation'], sample["filename"]

        image=np.array(image)
        segmentation = np.array(segmentation)
        segmentation=segmentation.astype(np.uint8)

        augmented = self.augmentation_image(image=image)
        image = augmented['image']
        replay = augmented['replay']

        segmentation = A.ReplayCompose.replay(replay, image=segmentation)['image']
        # Resize and convert to tensor
        image = self.augmentation_color(image=image)['image']
        image = self.image_transform(image=image)['image']
        segmentation = self.mask_transform(image=segmentation)['image']

        return {'image': image, 'segmentation': segmentation, "filename": img_name}
if torch.cuda.is_available():
    # 创建CUDA设备对象
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Instantiate the dataset
img_dir = r'/home/y/PycharmProjects/pythonProject3/数据集/TrainDataset/image'  # Update with your path
seg_dir = r'/home/y/PycharmProjects/pythonProject3/数据集/TrainDataset/masks'  # Update with your path
transformed_dataset = CustomDataset(img_dir=img_dir, seg_dir=seg_dir,transform=Transform1())

img_dir1 = r'/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-ColonDB/images'  # Update with your path
seg_dir1 = r'/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-ColonDB/masks'
test_dataset=CustomDataset(img_dir=img_dir1, seg_dir=seg_dir1,transform=Transform())

img_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-300/images"
seg_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-300/masks"
test_dataset1=CustomDataset(img_dir=img_dir2, seg_dir=seg_dir2,transform=Transform())

img_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-ClinicDB/images"
seg_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/CVC-ClinicDB/masks"
test_dataset2=CustomDataset(img_dir=img_dir2, seg_dir=seg_dir2,transform=Transform())


img_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/ETIS-LaribPolypDB/images"
seg_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/ETIS-LaribPolypDB/masks"
test_dataset3=CustomDataset(img_dir=img_dir2, seg_dir=seg_dir2,transform=Transform())

img_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/Kvasir/images"
seg_dir2=r"/home/y/PycharmProjects/pythonProject3/数据集/TestDataset/Kvasir/masks"
test_dataset4=CustomDataset(img_dir=img_dir2, seg_dir=seg_dir2,transform=Transform())

def iou(pred, labels, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class and return the average IoU.

    Args:Bottleneck
        pred (torch.Tensor): Model predictions, shape (N, H, W), where N is batch size,
                             each element in tensor should be class index for each pixel.
        labels (torch.Tensor): Ground truth labels, shape (N, H, W), each element should
                               be class index for each pixel.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        mean_iou (float): The mean IoU over all classes.
    """
    # Initialize list to store IoU for all classes
    ious = []

    # Avoid division by zero in IoU computation
    eps = 1e-6

    # Iterate over each class
    for cls in range(num_classes):
        # Intersection: True Positive (TP)
        intersection = ((pred == cls) & (labels == cls)).float().sum()

        # Union: TP + False Positive (FP) + False Negative (FN)
        union = ((pred == cls) | (labels == cls)).float().sum()

        # IoU: Intersection over Union
        iou = (intersection + eps) / (union + eps)  # Added epsilon to avoid division by zero

        # Append the IoU to the list
        ious.append(iou.item())

    # Compute the mean IoU by averaging over all classes
    mean_iou = sum(ious) / len(ious)
    return mean_iou


# 现在你可以使用train_dataset和test_dataset分别创建DataLoader
train_loader = DataLoader(transformed_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=True)
test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=True)
test_loader3 = DataLoader(test_dataset3, batch_size=1, shuffle=True)
test_loader4 = DataLoader(test_dataset4, batch_size=1, shuffle=True)


cnn_model = DuAT()
cnn_model=cnn_model.to(device)

# 定义损失函数
# criterion = nn.CrossEntropyLoss(reduction='none')  # 对每个像素计算损失但不求平均


optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr=0.005,momentum=0.90)
scheduler=StepLR(optimizer_cnn,step_size=50,gamma=0.9)
#37
num_epochs=130
for epoch in range(num_epochs):
    cnn_model.train()

    # 初始化累积损失
    total_loss_cnn_accumulated = 0.0
    num_batches = 0  # 追踪批次的数量
    rate=[1]
    for i, batch in enumerate(train_loader):
        for ra in rate:
            trans=int(round(352*ra/32)*32)
            inputs = batch['image']
            labels = batch['segmentation']
            if ra !=1:
                inputs=F.upsample(inputs,size=(trans,trans),mode="bilinear",align_corners=True)
                labels=F.upsample(labels,size=(trans,trans),mode="nearest")

            inputs = inputs.to(device)
            labels = labels.to(device)
            out=cnn_model(inputs)
            # 初始化批次的总损失
            total_loss_cnn = 0.0
            kappa = torch.Tensor(np.zeros((2))).to(device)
            # loss_cnn_labeled1= crition(outputs_cnn1, labels.squeeze(1),kappa)
            # loss_cnn_labeled2= crition(outputs_cnn2, labels.squeeze(1), kappa)
            # loss_cnn_labeled3= crition(outputs_cnn3, labels.squeeze(1), kappa)
            # loss_cnn_labeled4= crition(outputs_cnn4, labels.squeeze(1), kappa)
            # loss_cnn_labeled5=crition(outputs_cnn5,labels.squeeze(1), kappa)
            loss_cnn_labeled6 = crition(out, labels.squeeze(1), kappa)


            # total_loss_cnn += loss_cnn_labeled1+loss_cnn_labeled2+loss_cnn_labeled3+loss_cnn_labeled4+loss_cnn_labeled6+loss_cnn_labeled5
            total_loss_cnn+=loss_cnn_labeled6
            optimizer_cnn.zero_grad()

            loss =  total_loss_cnn  # 确保这是标量
            loss.backward()  # 先计算梯度
            optimizer_cnn.step()

            # 累积总损失用于报告
            total_loss_cnn_accumulated += total_loss_cnn.item()
            num_batches += 1

    # 每个epoch结束后输出平均损失
    print(
        f"Epoch {epoch + 1}, Average Total Loss CNN: {total_loss_cnn_accumulated / num_batches:.4f}")
    scheduler.step()
    if total_loss_cnn_accumulated / num_batches<=0.0063:
        break

cnn_model.eval()
p1=[]
for j, batch in enumerate((test_loader)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)
    out=cnn_model(image)
    out = torch.argmax(out, dim=1, keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    # print(a)
    p1.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    output_folder2 = r"CVC-ColonDB"
    os.makedirs(output_folder2, exist_ok=True)
    image.save(os.path.join(output_folder2, par))

print("CVC-ColonDB iou",sum(p1)/len(p1))

p2=[]
for j, batch in enumerate((test_loader1)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)

    out = cnn_model(image)
    out = torch.argmax(out, dim=1, keepdim=True)
    # a1 = torch.argmax(a1, dim=1, keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    # print(a)
    p2.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    output_folder2 = r"CVC-300"
    os.makedirs(output_folder2, exist_ok=True)
    image.save(os.path.join(output_folder2, par))

print("CVC-300 iou",sum(p2)/len(p2))
p3=[]
for j, batch in enumerate((test_loader2)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)

    out = cnn_model(image)
    out = torch.argmax(out, dim=1, keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    p3.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    output_folder2 = r"CVC-ClinicDB"
    os.makedirs(output_folder2, exist_ok=True)
    image.save(os.path.join(output_folder2, par))

print("CVC-ClinicDB iou",sum(p3)/len(p3))
p4=[]
for j, batch in enumerate((test_loader3)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)

    out = cnn_model(image)
    out=torch.argmax(out,dim=1,keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    p4.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    output_folder2 = r"ETIS-LaribPolypDB"
    os.makedirs(output_folder2, exist_ok=True)
    image.save(os.path.join(output_folder2, par))

print("ETIS-LaribPolypDB iou",sum(p4)/len(p4))
p5=[]
for j, batch in enumerate((test_loader4)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)

    out = cnn_model(image)
    out = torch.argmax(out, dim=1, keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    # print(a)
    p5.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    output_folder2 = r"Kvasir"
    os.makedirs(output_folder2, exist_ok=True)
    image.save(os.path.join(output_folder2, par))

print("Kvasir iou ",sum(p5)/len(p5))
print("sum iou ",(sum(p5)/len(p5)+sum(p4)/len(p4)+sum(p3)/len(p3)+sum(p2)/len(p2)+sum(p1)/len(p1))/5)

