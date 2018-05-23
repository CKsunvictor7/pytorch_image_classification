"""

---
class imbalanced => balance sampler
1    1448
0     847
total = 2,295 = 1,836(train) + 459(val)
* 1154x866
* .jpg

name,invasive
1,0
2,0
3,1
4,0
5,1
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data   # dataloader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from utils import splitter_df
from PIL import Image


IS_VAL = True
csv_file = 'data/train_labels.csv'
train_img_dir = 'data/train'
test_img_dir = 'data/test'


def load_img(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img_f:
            return img_f.convert('RGB')


class Invasive_DB(data.Dataset):
    def __init__(self, list_file, transforms=None):
        """
        load a list of files, not data

        Args:
            list_file: dataframe of data & label
            transforms: transformation funcs for data augmentation
        """
        self.transforms = transforms
        self.list = list_file

    def __getitem__(self, idx):
        """
        load data & do augmentation

        return:
            img: torch.tensor (PIL -> transform.toTensor)
            label: list of scalar (DataFrame -> list)
        """
        img_name = os.path.join(train_img_dir,
                        str(self.list.loc[idx, 'name']) + '.jpg')
        img = load_img(img_name)
        if self.transforms:
            img = self.transforms(img)

        label = self.list.loc[idx, 'invasive']
        return img, label

    def __len__(self):
        return len(self.list)

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.Resize((320, 320)),
        # transforms.RandomResizedCrop((224,224)),
        transforms.Resize((224, 224)),
        transforms.RandomRotation((-50, 50), resample=False, expand=False, center=None),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def runner(net, nb_epochs, batch_size, model_dir, model_name, learning_rate):
    """
    ### Data exploration ###
    df = pd.read_csv('data/train_labels.csv')
    print(df.head(10))
    print(df.loc[1,'name'])
    # print(df['invasive'].value_counts())
    """
    df = pd.read_csv(csv_file)
    # train_list, val_list = [], []
    if IS_VAL:
        train_df, val_df = splitter_df(df, rate=0.8)
    else:
        train_df = df

    # data loader
    dataset = {'train':Invasive_DB(train_df, data_transforms['train']),
               'val':Invasive_DB(val_df, data_transforms['train'])}

    # test if data loader works or not
    print("nb of the train_DB is ", len(dataset['train']))
    print("nb of the val_DB is ", len(dataset['val']))

    data_loaders = {'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=False, num_workers=4)}

    # fine-tuning
    net_ft = net(pretrained=True)

    # find the name of last layer, @resnet18
    # = self.fc: nn.Linear(512 * block.expansion, num_classes)）
    # and replace it with new layer
    net_ft.fc = nn.Sequential(nn.Linear(net_ft.fc.in_features, out_features=2), nn.Sigmoid())
    # put into gpu
    net_ft = net_ft.cuda()

    """
    # to train only last layer, set other layers' requires_grad as False 
    for para in list(model_ft.parameters())[:-2]:
        para.requires_grad = False
    """
    # TODO:nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_ft.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, threshold=0.000001,
        verbose=True)

    # best_model = net_ft
    best_acc = 0.0
    # early stopping config
    ES_obs_times = 5
    ES_monitor_value = 'training_loss' # 'training_loss' or  'val_acc'
    last_n_epoches = []

    for epoch in range(nb_epochs):
        print('epoch-{}/{}'.format(epoch, nb_epochs - 1))

        # one epoch = one training + one validation
        for phase in ['train', 'val']:
            # Sets model to training mode or evaluation mode
            if phase == 'train':
                net_ft.train(mode=True)
            else:
                net_ft.train(mode=False)

            running_loss = 0.0
            running_hits = 0

            # run one epoch
            for imgs, labels in data_loaders[phase]:
                # convert tensor to variable
                # .cuda(): put data into gpu
                # labels: list -> Variable contains torch.LongTensor
                imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

                # set previous grad to zero
                optimizer.zero_grad()
                # outputs: Variable contains torch.FloatTensor of (batch_size, nb of class)
                outputs = net_ft(imgs)

                # loss is Variable contains torch.FloatTensor of size 1
                loss = loss_func(outputs, labels)
                # Use tensor.item() to convert a 0-dim tensor to a Python number
                running_loss += loss.data.item()

                # compute hits:
                # torch.max returns (Variable(Tensor), Variable(LongTensor))
                # 1-D is the max value along dim
                # 2-D is the argmax value along dim
                preds = torch.max(outputs, 1)[1]

                # input of torch.sum() can be Variable or tensor
                # if Variable, output is Variable
                # running_hits += torch.sum(preds == labels).data[0]
                # if tensor, output is scalar value
                running_hits += torch.sum(preds.data == labels.data)


                if phase == 'train':
                    # back propagation, compute gradient
                    loss.backward()
                    # apply gradient
                    optimizer.step()

            # after one epoch
            epoch_loss = running_loss/len(dataset[phase])
            epoch_acc = running_hits.item()/len(dataset[phase])

            # lr_scheduler update
            scheduler.step(epoch_loss)

            print('{}  epoch_acc:{:.4f}  epoch_acc:{:.4f}'.
                  format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                print('new best val_Acc, save the model')
                # remove the previous model
                if best_acc != 0.0:
                    model_name_w_acc = "{}_{:.4f}.pkl".format(model_name, best_acc)
                    os.remove(os.path.join(model_dir, model_name_w_acc))

                best_acc = epoch_acc
                model_name_w_acc = "{}_{:.4f}.pkl".format(model_name, best_acc)
                torch.save(net_ft, os.path.join(model_dir, model_name_w_acc))

                # early stopping by val_acc
                if ES_monitor_value == 'val_acc':
                    if last_n_epoches and epoch_loss > max(last_n_epoches):
                        last_n_epoches.clear()
                    last_n_epoches.append(epoch_loss)

            # early stopping by training loss
            if phase == 'train' and ES_monitor_value == 'training_loss':
                if last_n_epoches and epoch_loss < min(last_n_epoches):
                    last_n_epoches.clear()
                last_n_epoches.append(epoch_loss)

        if len(last_n_epoches) == (ES_obs_times+1):
            print('early stopping by {} with the best validation accuracy:{}'.
                format(ES_monitor_value, best_acc))
            break

        print('-' * 10)
