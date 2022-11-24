from notifier import Notifier
from alive_progress import alive_bar
import numpy as np
import os
import imageio
import glob
import matplotlib.pyplot as plt
import ants
import time
import torch
from torch import nn
from torchsummary import summary
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
import gc
from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


path_file = './data/mri_list.csv'
run_id = 'baseline_2'
PREFIX = f'./models/ConvTorch/{run_id}'


def build_deep_batch(train_data, test_data, batch_size, mini_batch_size, mini_batch_size_test):
    train_chunks = []
    for i in range(0, len(train_data), batch_size):
        chunks = []
        for idx_batch in range(0, batch_size):
            chunks.append(train_data[i + idx_batch])
        train_chunks.append(chunks)

    test_chunks = []
    for i in range(0, len(test_data), batch_size):
        chunks = []
        for idx_batch in range(0, batch_size):
            chunks.append(test_data[i + idx_batch])
        test_chunks.append(chunks)

    _train_batches = []
    for chunk in train_chunks:
        main_chunk = []
        for i in range(0, len(chunk), mini_batch_size):
            mini_chunk = []
            for idx_batch in range(0, mini_batch_size):
                mini_chunk.append(chunk[i + idx_batch])
            main_chunk.append(mini_chunk)
        _train_batches.append(main_chunk)

    _test_batches = []
    for chunk in test_chunks:
        main_chunk = []
        for i in range(0, len(chunk), mini_batch_size_test):
            mini_chunk = []
            for idx_batch in range(0, mini_batch_size_test):
                mini_chunk.append(chunk[i + idx_batch])
            main_chunk.append(mini_chunk)
        _test_batches.append(main_chunk)

    return _train_batches, _test_batches


def load_image_net(path):
    """open image from path"""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

batch_size = 4

def load_mri(img_path):
    imagen_numpy = ants.image_read(img_path).numpy()
    imagen_numpy = ((imagen_numpy - np.min(imagen_numpy))/(np.max(imagen_numpy) - np.min(imagen_numpy)))
    return np.reshape(imagen_numpy, (1 , 197, 233, 189))


class MRI_Dataset(Dataset):
    def __init__(self, path_csv, transform=None):
        self.path_csv = path_csv
        self.transform = transform
        self.data = []
        self.labels = []
        with open(path_file, 'r') as f:
            for line in f:
                path, label = line.split(',')
                self.data.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = load_mri(self.data[idx])
        image = torch.from_numpy(image).float()
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        return image, label


dataset = MRI_Dataset(path_file)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print('Length of trainset: ', len(trainset))
print('Length of testset: ', len(testset))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.functional.relu(input)

class CNNResnet(nn.Module):
    def __init__(self, in_channels, ResBlock, outputs=2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock(32, 32, downsample=True),
            ResBlock(32, 32, downsample=False)
        )

        self.layer4 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer5 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer6 = nn.Sequential(
            ResBlock(32, 32, downsample=True),
            ResBlock(32, 32, downsample=False)
        )

        self.layer7 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer8 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )

        self.layer9 = nn.Sequential(
            ResBlock(32, 32, downsample=True),
            ResBlock(32, 32, downsample=False)
        )

        self.layer10 = nn.Sequential(
            ResBlock(32, 32, downsample=False),
            ResBlock(32, 32, downsample=False)
        )



        self.gap = torch.nn.AdaptiveAvgPool3d(12)
        self.fc1 = torch.nn.Linear(55296, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.layer10(input)
        input = self.gap(input)
        input = input.view(input.size(0), -1)
        # input = torch.flatten(input)
        input = nn.functional.relu(self.fc1(input))
        input = nn.functional.relu(self.fc2(input))
        input = nn.functional.softmax(self.fc3(input))


        return input

notifier = Notifier(
    title='Training Torch Model',
    chat_id=293701727,
    api_token='1878628343:AAEFVRsqDz63ycmaLOFS7gvsG969wdAsJ0w',
)

Numero_clases = 5
epochs_number = 200
total_epochs = 150

maximo_imagenes_batch = 8
maximo_imagenes_batch_test = 8

loss_epoch = []
acc_epoch = []
loss_epoch_test = []
acc_epoch_test = []
rate_notification = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA AVAILABLE' if torch.cuda.is_available() else 'MISSING CUDA')
print('Device', device)
#Red_Neuronal = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

model_save_path = f'{PREFIX}/best_model.pth'
if os.path.exists(model_save_path):
    print('Loading model from', model_save_path)
    Red_Neuronal = torch.load(model_save_path)
    epochs_number = total_epochs - 50
else:
    print('Creating new model')
    Red_Neuronal = CNNResnet(1, ResBlock, outputs=Numero_clases)
    epochs_number = total_epochs

Red_Neuronal.to(device)
#summary(Red_Neuronal, (3, 197, 233, 189))
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss() 
optimizer = optim.Adam(Red_Neuronal.parameters(), lr=0.001, weight_decay=0.001) # 0.001


import shutil
if os.path.exists(f'{PREFIX}/logs'):
    shutil.rmtree(f'{PREFIX}/logs')
    os.mkdir(f'{PREFIX}/logs')

writer = SummaryWriter(log_dir=f'{PREFIX}/logs')
best_loss_test = float('inf')



print('Starting training, epochs:', epochs_number)
accuracy = Accuracy(num_classes=Numero_clases).to(device)

with alive_bar((len(trainloader) + len(testloader)) * epochs_number, dual_line=True, title='Torch training') as bar:

    for epoch in range(epochs_number):
        acc_batch = []
        loss_batch = []
        
        tiempo_inicial = time.time()
        stage = 'train'

        for batch_idx, batch in enumerate(trainloader, 0):
            inputs, labels = batch

            optimizer.zero_grad()
            outputs = Red_Neuronal(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            acc_batch.append(accuracy(outputs, labels.to(device)).detach().cpu().numpy())
            loss_batch.append(loss.item())

            bar.text = f'-> [TRAIN] Epoch: {epoch} Batch: {batch_idx} Loss: {loss_batch[-1]:.4f} Acc: {acc_batch[-1]:.4f}'
            bar()

            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()


        loss_epoch.append(np.mean(loss_batch))
        acc_epoch.append(np.mean(acc_batch))
        print('Changing to TEST')

        acc_batch_test = []
        loss_batch_test = []

        for batch_idx, batch in enumerate(testloader, 0):
            inputs, labels = batch

            outputs = Red_Neuronal(inputs.to(device))
            loss = criterion(outputs, labels.to(device))

            acc_batch_test.append(accuracy(outputs, labels.to(device)).detach().cpu().numpy())
            loss_batch_test.append(loss.item())

            bar.text = f'-> [TEST] Epoch: {epoch} Batch: {batch_idx} Loss: {loss_batch_test[-1]:.4f} Acc: {acc_batch_test[-1]:.4f}'
            bar()

            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        loss_epoch_test.append(np.mean(loss_batch_test))
        acc_epoch_test.append(np.mean(acc_batch_test))

        writer.add_scalar('Loss/train', loss_epoch[-1], epoch)
        writer.add_scalar('Loss/test', loss_epoch_test[-1], epoch)
        writer.add_scalar('Accuracy/train', acc_epoch[-1], epoch)
        writer.add_scalar('Accuracy/test', acc_epoch_test[-1], epoch)

        if epoch % rate_notification == 0 and False:
            try:
                parsed_metrics = f'Epoch {epoch}\n'
                parsed_metrics += f'Loss: {loss_epoch[-1]}\n'
                parsed_metrics += f'Accuracy: {acc_epoch[-1]}\n'
                parsed_metrics += f'Val loss: {loss_epoch_test[-1]}\n'
                parsed_metrics += f'Val accuracy: {acc_epoch_test[-1]}\n'
                #notifier(msg=parsed_metrics)
            except:
                print('No se pudo enviar el mensaje de notificación')

        if loss_epoch_test[-1] < best_loss_test:
            best_loss_test = loss_epoch_test[-1]
            with open(f'{PREFIX}/best_metrics.text', 'w') as file:
                file.write(f'Epoch: {epoch}\n')
                file.write(f'Best loss: {loss_epoch[-1]}\n')
                file.write(f'Best acc: {acc_epoch[-1]}\n')
                file.write(f'Best val loss: {loss_epoch_test[-1]}\n')
                file.write(f'Best val acc: {acc_epoch_test[-1]}\n')
            torch.save(Red_Neuronal, model_save_path)

        with open(f'{PREFIX}/last_metrics.text', 'w') as file:
            file.write(f'Epoch: {epoch}\n')
            file.write(f'Loss: {loss_epoch[-1]}\n')
            file.write(f'Accuracy: {acc_epoch[-1]}\n')
            file.write(f'Val loss: {loss_epoch_test[-1]}\n')
            file.write(f'Val accuracy: {acc_epoch_test[-1]}\n')

df = pd.DataFrame({'loss': loss_epoch, 'acc': acc_epoch, 'loss_test': loss_epoch_test, 'acc_test': acc_epoch_test})
df.to_csv(f'{PREFIX}/metrics.csv', index=False)

writer.flush()
writer.close()
