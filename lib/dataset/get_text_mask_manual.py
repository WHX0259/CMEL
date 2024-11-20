from torch.utils.data import Dataset
import pandas as pd
import random
import os
import json
import random
import torch
from torchvision import transforms as T
# from transformers import BertTokenizer, BertModel
import pandas as pd
import json
from PIL import Image
from transformers import BertTokenizer
import random
from PIL import ImageFilter
import json
import pandas as pd
from collections import defaultdict
from torchvision import transforms
def apply_color_jitter(image, brightness=0.2, contrast=0.2):
    transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, hue=0.0)
    return transform(image)

def apply_filters(image):
    filters = [ImageFilter.BLUR, ImageFilter.DETAIL, ImageFilter.EDGE_ENHANCE, ImageFilter.SMOOTH]
    for filt in filters:
        if random.random() < 0.1:
            image = image.filter(filt)
    return image

def apply_transforms(image, mode, augmentation_prob):
    Transform = []
    aspect_ratio = image.size[1] / image.size[0]
    ResizeRange = random.randint(100, 128)
    Transform.append(transforms.Resize((int(ResizeRange * aspect_ratio), ResizeRange), antialias=True))

    if mode == 'train' and random.random() <= augmentation_prob:
        # RotationRange = random.randint(-10, 10)
        Transform.append(transforms.RandomRotation((-10, 10)))
        CropRange = random.randint(100, 128)
        Transform.append(transforms.CenterCrop((int(CropRange * aspect_ratio), CropRange)))

        ShiftRange_left = random.randint(0, 20)
        ShiftRange_upper = random.randint(0, 20)
        ShiftRange_right = image.size[0] - random.randint(0, 20)
        ShiftRange_lower = image.size[1] - random.randint(0, 20)
        image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

        image = apply_filters(image)

        image = apply_color_jitter(image)
    Transform.append(transforms.ToTensor())
    Transform.append(transforms.Resize([224, 224], antialias=True))

    Transform = transforms.Compose(Transform)
    return Transform(image)


def weak_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def strong_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.0)  # 移除saturation和hue
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.5),
    ])
class Dataset_Slice_text_manual(Dataset):
    def __init__(self, slice_path, fold_json='/data/huixuan/data/data_chi/TRG_patient_folds.json',
                 csv_path='/data/huixuan/data/data_chi/label.csv',
                 clinic_path='/data/huixuan/data/data_chi/survival.csv',
                 manual_csv_path='/data/wuhuixuan/code/Self_Distill_MoE/data/selected_features_22_with_id_label_fold_norm.csv',
                 sentence_json='/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/sentences.json',
                 fold=0, mode='train', extra_fold=None, num_classes = 2, transform=None):
        self.augmentation_prob = 0.5
        self.mode = mode
        with open(sentence_json, 'r') as load_f:
            self.sentence_data = json.load(load_f)
        self.slice_path = slice_path
        self.image_list = []
        with open(fold_json, 'r') as file:
            data = json.load(file)
        self.manual_data = pd.read_csv(manual_csv_path)
        self.manual_data = self.manual_data.fillna(-1)
        self.data = pd.read_csv(clinic_path)#读取临床数据
        self.data['Age'] = self.data['Age'] / 100#应该归一化才对
        self.data = self.data.fillna(-1)
        self.data = self.data.drop(columns=['TRG13_45'])
        if mode == 'train':
            for i in range(10):
                if extra_fold is not None and i == extra_fold:
                    continue
                if i != fold and (extra_fold is None or i != extra_fold):
                    for case_id in data['Fold ' + str(i + 1)]:
                        self.image_list.append(case_id)
        elif mode == 'val':
            for case_id in data['Fold ' + str(fold + 1)]:
                self.image_list.append(case_id)
        elif mode == 'extra':
            if extra_fold is None:
                raise ValueError("When mode is 'extra', extra_fold must be provided.")
            for case_id in data['Fold ' + str(extra_fold + 1)]:
                self.image_list.append(case_id)

        df = pd.read_csv(csv_path)
        self.TRG_dict = dict(zip(df['ID'], df['label']))
        self.transform = transform
        # 统计每个类别的样本数
        self.cls_num_list = [0 for _ in range(num_classes)]
        for case_id in self.image_list:
            case_id = case_id.split('_')[0]
            class_id = self.TRG_dict.get(case_id + '.nii.gz')
            if class_id is not None:
                self.cls_num_list[int(class_id)] += 1


    def __getitem__(self, index):
        if self.image_list[index].endswith('.png'):
            image_name = self.image_list[index].split('_')[0]
        elif self.image_list[index].endswith('.nii.gz'):
            image_name = self.image_list[index]
        case_id = int(image_name)
        class_id = self.TRG_dict[image_name + '.nii.gz']
        sentence = self.sentence_data[image_name]
        tokenizer = BertTokenizer.from_pretrained('/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/BertTokenizer')
        text_input = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors="pt")
        slice_name = image_name + '.png'
        manual_data = self.manual_data[self.manual_data['ID'] == image_name + '.nii.gz']
        manual_features = manual_data.iloc[:, :-3].values
        clinic_features = self.data[self.data['ID'] == case_id]  # 替换 'image_column' 为实际的图像文件名列名
        clinic_features = clinic_features.iloc[:, 1:].values# 除去ID和结局信息
        slice_image = Image.open(os.path.join(self.slice_path, slice_name)).convert('RGB')
        if self.transform is not None:
            slice_image1 = self.transform[0](slice_image)#进行弱增强得到#[1, 224, 224]
            slice_image2 = self.transform[1](slice_image)#进行强增强得到#[1, 224, 224]
            Norm_ = T.Normalize((0.1591, 0.1591, 0.1591), (0.2593, 0.2593, 0.2593))
            slice_image1 = Norm_(slice_image1)
            Norm_ = T.Normalize((0.1591, 0.1591, 0.1591), (0.2593, 0.2593, 0.2593))
            slice_image2 = Norm_(slice_image2)
            slice_image = [slice_image1, slice_image2]
        else:
            slice_image = apply_transforms(slice_image, self.mode, self.augmentation_prob)
            Norm_ = T.Normalize((0.1591, 0.1591, 0.1591), (0.2593, 0.2593, 0.2593))
            slice_image = Norm_(slice_image)

        return slice_image, clinic_features, manual_features, class_id, case_id

    def __len__(self):
        return len(self.image_list)

# from dataset import Dataset_Slice_text_manual

if __name__ == '__main__':
    slice_path = r'/data/wuhuixuan/data/padding_crop'
    fold_json = r'/data/huixuan/data/data_chi/TRG_patient_folds.json'
    manual_csv_path = r'/data/wuhuixuan/code/Self_Distill_MoE/data/selected_features_22_with_id_label_fold_norm.csv'
    sentence_json = r'/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/sentences.json'
    csv_path = r'/data/huixuan/data/data_chi/label.csv'
    weak_aug = weak_transform()
    strong_aug = strong_transform()
    # Training dataset
    train_dataset = Dataset_Slice_text_manual(
        slice_path=slice_path,
        fold_json=fold_json,
        manual_csv_path=manual_csv_path,
        sentence_json=sentence_json,
        csv_path=csv_path,
        fold=0,  # Specify the fold for training
        mode='train',
        transform = [weak_aug, strong_aug]
    )

    # Validation dataset
    val_dataset = Dataset_Slice_text_manual(
        slice_path=slice_path,
        fold_json=fold_json,
        manual_csv_path=manual_csv_path,
        sentence_json=sentence_json,
        csv_path=csv_path,
        fold=0,  # Specify the fold for validation
        mode='val'
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True
    )
    # Print sizes of the datasets
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(train_dataset.cls_num_list)
    for i, j, p, q, r in test_dataloader:#定义了数据集有几个输出就for几个变量，如果顺利运行就非常可以
        
        image = torch.cat(i,dim=0)
        print(image.shape)
        # print(j.shape)
        clinic_data = torch.cat([j,j], dim=0)
        print(clinic_data.shape)
        ml_feature = torch.cat([p,p], dim=0)
        print(ml_feature.shape)
        label = torch.cat([q,q], dim=0)
        print(q)
        print(r)
        continue

# if __name__ == '__main__':#下面这个是测试文件,你们测试好了再去调,测试好之后的文件名直接复制到main 函数中参数设置部分
#     slice_path = r'/data/wuhuixuan/data/padding_crop'#这个是我后来发给你们的
#     fold_json = r'/data/huixuan/data/data_chi/TRG_patient_folds.json'#这个是划分数据集的文件，只看文件名，找到相应文件的位置
#     manual_csv_path = r'/data/wuhuixuan/data/data_chi/selected_features.csv'
#     sentence_json = r'/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/sentences.json'
#     csv_path = r'/data/huixuan/data/data_chi/label.csv'#这个是标签
#     test_dataset = Dataset_Slice_text_manual(#imageFolder是自己编写的数据集的代码，下面是自己的参数
#         slice_path=slice_path,
#         fold_json = fold_json,
#         manual_csv_path=manual_csv_path,
#         sentence_json=sentence_json,
#         csv_path=csv_path,
#         fold=6, mode='train')

#     test_dataloader = torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         batch_size=1,
#         shuffle=True
#     )
#     num=0


