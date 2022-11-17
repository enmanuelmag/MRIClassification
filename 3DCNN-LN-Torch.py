import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
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


path_file = 'DB'


# function to load data for one specific class 
def cargar_nombres_imagenes(name, path_file):
  Dato =[]
  path_complete = str(path_file + "/" + name)
  list_patients=os.listdir(path_complete)
  for j in list_patients:
    img_path = str(path_complete +'/'+str(j))
    Dato.append(img_path)

  Dato = np.asarray(Dato)
  return Dato
  
def crear_labels(num, clase):
  cero = np.zeros((num,1))
  uno = np.ones((num,1))
  if clase==0:
    # label = np.hstack((uno, cero, cero, cero, cero))
    label = np.hstack((uno, cero))
  if clase==1:
    # label = np.hstack((cero, uno, cero, cero, cero))
    label = np.hstack((cero, uno))
  # if clase==2:
  #   label = np.hstack((cero, cero, uno, cero, cero))
  # if clase==3:
  #   label = np.hstack((cero, cero, cero, uno, cero))
  # if clase==4:
  #   label = np.hstack((cero, cero, cero, cero, uno))
  return label
  
def split_data(data, label):
  training = data[0 : int(np.floor(len(data)*0.8))]
  train_label = label[0 : int(np.floor(len(data)*0.8)),:]
  testing = data[int(np.ceil(len(data)*0.8)) : len(data)]
  testing_label = label[int(np.ceil(len(data)*0.8)) : len(data)]

  return training, train_label, testing, testing_label
  
def shuffle(data, label):
  dato_shuffle = []
  label_shuffle = []
  index = np.arange(len(data))
  np.random.shuffle(index)
  for i in index:
    dato_shuffle.append(data[i])
    label_shuffle.append(label[i,:])

  dato_shuffle = np.asarray(dato_shuffle)
  label_shuffle = np.asarray(label_shuffle)

  return dato_shuffle, label_shuffle
  

# function to load data for one specific class 
def cargar_imagenes(name, num):
    Dato = []
    for j in name:
        img_path = str(j)
        #print(img_path)
        img = ants.image_read(img_path)
        imagen_numpy = img.numpy()
        #data normalizaion
        imagen_numpy=((imagen_numpy - np.amin(imagen_numpy))/(np.amax(imagen_numpy) - np.amin(imagen_numpy)))
        Dato.append(imagen_numpy)

    Dato = np.asarray(Dato)
    Dato = np.reshape(Dato, (num, 1 , 197, 233, 189))
    Dato = torch.tensor(Dato)
#     print("##################################################")
#     print(Dato.shape)
#     print("##################################################")
    return Dato    
  

# datos_sex = cargar_nombres_imagenes("Sex", path_file)
# datos_gamble = cargar_nombres_imagenes("Gamble", path_file)
datos_eat = cargar_nombres_imagenes("Eat", path_file)
# datos_buy = cargar_nombres_imagenes("Buy", path_file)
datos_pd = cargar_nombres_imagenes("PD", path_file)

# label_sex = crear_labels(len(datos_sex),0)
# label_gamble = crear_labels(len(datos_gamble),1)
# label_eat = crear_labels(len(datos_eat),2)
# label_buy = crear_labels(len(datos_buy),3)
# label_pd = crear_labels(len(datos_pd),4)

label_eat = crear_labels(len(datos_eat),0)
label_pd = crear_labels(len(datos_pd),1)

# T_S, L_S, TS_S, LS_S= split_data(datos_sex, label_sex)
# T_G, L_G, TS_G, LS_G= split_data(datos_gamble, label_gamble)
T_E, L_E, TS_E, LS_E= split_data(datos_eat, label_eat)
# T_B, L_B, TS_B, LS_B= split_data(datos_buy, label_buy)
T_P, L_P, TS_P, LS_P= split_data(datos_pd, label_pd)


# training_data = np.hstack((T_S, T_G, T_E, T_B, T_P))
# training_label = np.vstack((L_S, L_G, L_E, L_B, L_P))
# testing_data = np.hstack((TS_S, TS_G, TS_E, TS_B, TS_P))
# testing_label = np.vstack((LS_S, LS_G, LS_E, LS_B, LS_P))


training_data = np.hstack((T_E, T_P))
training_label = np.vstack((L_E, L_P))
testing_data = np.hstack((TS_E, TS_P))
testing_label = np.vstack((LS_E, LS_P))


# training_data = np.hstack((T_S, T_G))
# training_label = np.vstack((L_S, L_G))
# testing_data = np.hstack((TS_S, TS_G))
# testing_label = np.vstack((LS_S, LS_G))

training_data, training_label = shuffle(training_data, training_label)
testing_data, testing_label = shuffle(testing_data, testing_label)

np.savetxt('Testing_labels.txt', np.asarray(testing_label), delimiter=",")
with open('Testing_dataset.txt', 'w') as f:
    for line in testing_data:
        f.write(line)
        f.write('\n')

# print('******************************************************************')
# print('Data shapes:')
# print(training_data.shape)
# print(training_label.shape)
# print(testing_data.shape)
# print(testing_label.shape)
# print('******************************************************************')


# #SEPARABLE CONVOLUTION = DEPTHWISE + COMBINATION CONV STEP
# main_input = Input(shape=(197, 233, 189, 1),dtype='float32',name='main_input')
# First_input_C1 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(main_input)
# First_input_C1_Batch = BatchNormalization(axis=4)(First_input_C1)
# First_Pooling = MaxPool3D(pool_size=(2,2,2))(First_input_C1_Batch)
# First_input_C2 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(First_Pooling)
# First_input_C2_Batch = BatchNormalization(axis=4)(First_input_C2)
# First_input_C3 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(First_input_C2_Batch)
# First_input_C3_Batch = BatchNormalization(axis=4)(First_input_C3)
# Dropout_layer_1 = Dropout(0.3)(First_input_C3_Batch)

# CNN_C6 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_1)
# CNN_C6_Batch = BatchNormalization(axis=4)(CNN_C6)

# CNN_C7 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C6_Batch)
# CNN_C7_Batch = BatchNormalization(axis=4)(CNN_C7)
# # Dropout_layer_2 = Dropout(0.35)(CNN_C7_Batch)
# CNN_C8 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C7_Batch)
# CNN_C8_Batch = BatchNormalization(axis=4)(CNN_C8)
# layer_Addition_1 = Add()([CNN_C8_Batch,CNN_C6_Batch])
# # CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_1)

# CNN_C74 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_1)
# CNN_C74_Batch = BatchNormalization(axis=4)(CNN_C74)
# # Dropout_layer_2 = Dropout(0.35)(CNN_C7_Batch)
# CNN_C84 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C74_Batch)
# CNN_C84_Batch = BatchNormalization(axis=4)(CNN_C84)
# layer_Addition_14 = Add()([CNN_C84_Batch,layer_Addition_1])
# # CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_1)

# CNN_C73 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_14)
# CNN_C73_Batch = BatchNormalization(axis=4)(CNN_C73)
# Dropout_layer_23 = Dropout(0.35)(CNN_C73_Batch)
# CNN_C83 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_23)
# CNN_C83_Batch = BatchNormalization(axis=4)(CNN_C83)
# layer_Addition_13 = Add()([CNN_C83_Batch,layer_Addition_14])
# CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_13)

# CNN_C9 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_Pooling_1)
# CNN_C9_Batch = BatchNormalization(axis=4)(CNN_C9)
# # Dropout_layer_3 = Dropout(0.40)(CNN_C9_Batch)
# CNN_C10 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C9_Batch)
# CNN_C10_Batch = BatchNormalization(axis=4)(CNN_C10)
# layer_Addition_2 = Add()([CNN_C10_Batch,CNN_Pooling_1])
# # CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_2)

# CNN_C93 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_2)
# CNN_C93_Batch = BatchNormalization(axis=4)(CNN_C93)
# # Dropout_layer_3 = Dropout(0.40)(CNN_C9_Batch)
# CNN_C103 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C93_Batch)
# CNN_C103_Batch = BatchNormalization(axis=4)(CNN_C103)
# layer_Addition_23 = Add()([CNN_C103_Batch,layer_Addition_2])
# # CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_2)

# CNN_C92 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_23)
# CNN_C92_Batch = BatchNormalization(axis=4)(CNN_C92)
# Dropout_layer_32 = Dropout(0.40)(CNN_C92_Batch)
# CNN_C102 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_32)
# CNN_C102_Batch = BatchNormalization(axis=4)(CNN_C102)
# layer_Addition_22 = Add()([CNN_C102_Batch,layer_Addition_23])
# CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_22)

# CNN_C91 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_Pooling_2)
# CNN_C91_Batch = BatchNormalization(axis=4)(CNN_C91)

# CNN_C11 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C91_Batch)
# CNN_C11_Batch = BatchNormalization(axis=4)(CNN_C11)
# # Dropout_layer_4 = Dropout(0.45)(CNN_C11_Batch)
# CNN_C12 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C11_Batch)
# CNN_C12_Batch = BatchNormalization(axis=4)(CNN_C12)
# layer_Addition_3 = Add()([CNN_C12_Batch,CNN_C91_Batch])
# # CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_3)

# CNN_C911 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_3)
# CNN_C911_Batch = BatchNormalization(axis=4)(CNN_C911)

# CNN_C111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C911_Batch)
# CNN_C111_Batch = BatchNormalization(axis=4)(CNN_C111)
# # Dropout_layer_41 = Dropout(0.45)(CNN_C111_Batch)
# CNN_C121 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C111_Batch)
# CNN_C121_Batch = BatchNormalization(axis=4)(CNN_C121)
# layer_Addition_31 = Add()([CNN_C121_Batch,CNN_C911_Batch])
# # CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_31)

# CNN_C9111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_31)
# CNN_C9111_Batch = BatchNormalization(axis=4)(CNN_C9111)

# CNN_C1111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C9111_Batch)
# CNN_C1111_Batch = BatchNormalization(axis=4)(CNN_C1111)
# Dropout_layer_411 = Dropout(0.45)(CNN_C1111_Batch)
# CNN_C1211 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_411)
# CNN_C1211_Batch = BatchNormalization(axis=4)(CNN_C1211)
# layer_Addition_311 = Add()([CNN_C1211_Batch,CNN_C9111_Batch])
# CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_311)

# flatten_layer = Flatten()(CNN_Pooling_3)
# dense_layer_1 = Dense(units=100,activation="relu")(flatten_layer)
# Dropout_layer_5 = Dropout(0.20)(dense_layer_1)
# dense_layer_2 = Dense(units=50,activation="relu")(Dropout_layer_5)
# output_layer = Dense(units=2,activation="softmax")(dense_layer_2)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample):
#         super().__init__()
#         if downsample:
#             self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
#                 nn.BatchNorm3d(out_channels)
#             )
#         else:
#             self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             self.shortcut = nn.Sequential()

#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.bn2 = nn.BatchNorm3d(out_channels)

#     def forward(self, input):
#         shortcut = self.shortcut(input)
#         input = nn.ReLU()(self.bn1(self.conv1(input)))
#         input = nn.ReLU()(self.bn2(self.conv2(input)))
#         input = input + shortcut
#         return nn.ReLU()(input)

# class CNNResnet(nn.Module):
#     def __init__(self, in_channels, resblock, outputs=2):
#         super().__init__()
#         self.layer0 = nn.Sequential(
#             nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.MaxPool3d(kernel_size=2, stride=1, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU()
#         )

#         self.layer1 = nn.Sequential(
#             resblock(32, 32, downsample=False),
#             resblock(32, 32, downsample=False)
#         )

#         self.layer2 = nn.Sequential(
#             resblock(32, 64, downsample=True),
#             resblock(64, 64, downsample=False)
#         )

#         self.layer3 = nn.Sequential(
#             resblock(64, 64, downsample=True),
#             resblock(64, 64, downsample=False)
#         )

#         self.layer4 = nn.Sequential(
#             resblock(64, 64, downsample=True),
#             resblock(64, 64, downsample=False)
#         )

# #         self.gap = torch.nn.AdaptiveAvgPool3d(1)
#         self.fc1 = torch.nn.Linear(299520, 256)
#         self.fc2 = torch.nn.Linear(256, 256)
#         self.fc3 = torch.nn.Linear(256, outputs)

#     def forward(self, input):
#         input = self.layer0(input)
#         input = self.layer1(input)
#         input = self.layer2(input)
#         input = self.layer3(input)
#         input = self.layer4(input)
# #         input = self.gap(input)
#         input = torch.flatten(input)
#         input = self.fc1(input)
#         input = self.fc2(input)
#         input = self.fc3(input)

#         return input

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm3d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool3d(2, stride=1)
        self.fc1 = nn.Linear(107520, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

Numero_clases=2
epochs_number = 100
batch_size = 100 # multiplo de 5
maximo_imagenes_batch = 10
maximo_imagenes_batch_test = 5
loss_epoch=[]
acc_epoch=[]
loss_epoch_test=[]
acc_epoch_test=[]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Red_Neuronal = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
# Red_Neuronal = CNNResnet(1, ResBlock, outputs=Numero_clases)
# Red_Neuronal.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(Red_Neuronal, (1, 197, 233, 189))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Red_Neuronal.parameters(), lr=0.0001)

print('Total Training Data ' + str(len(training_data)))
# for i in range (0, epochs_number):  
for i in range (0, 1):  
    #FOR TRAINING
    loss_batch=[]
    loss_training = []
    acc_batch=[]
    acc_training =[]
    tiempo_inicial = time.time()
    index_values_train = np.arange(0,int(len(training_data)),maximo_imagenes_batch)
    running_loss=0
    j=0
    while ((j+1) < int(len(index_values_train))):
#     while (j<1):
        running_loss=0
        tiempo_inicial_batch = time.time()
        for k in range (0,int(batch_size/maximo_imagenes_batch)):
            if ((j+1) < int(len(index_values_train))):
                data_train_net = cargar_imagenes(training_data[index_values_train[j]:index_values_train[j+1]], maximo_imagenes_batch)
                data_train_net = data_train_net.cuda()
                optimizer.zero_grad()
                #forward + backward + optimize
                outputs = Red_Neuronal(data_train_net)
                if (k==0):
                    prediction_output = outputs.cpu().data.numpy()
                else:
                    prediction_output = np.vstack((prediction_output,outputs.cpu().data.numpy()))
                j=j+1

        label_train_net = training_label[index_values_train[j-int(batch_size/maximo_imagenes_batch)]:index_values_train[j]]
        label_train_net = torch.tensor(label_train_net).cuda()
        prediction_output = torch.tensor(prediction_output).cuda()
        
        loss = criterion(prediction_output, label_train_net)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        # print statistics
        accuracy = Accuracy()
        accuracy = accuracy.cuda()
        acc_tmp = accuracy(prediction_output, torch.argmax(label_train_net, dim=1).cuda())
        running_loss += loss.item()
        acc_batch.append(acc_tmp.cpu().data.numpy())
        loss_batch.append(running_loss)
        print('Batch loss = ' + str(running_loss))
        print('tiempo del batch: ', (time.time() - tiempo_inicial_batch))
        
        
    loss_batch = np.asarray(loss_batch)
    acc_batch = np.asarray(acc_batch)
    print('************************************** TESTING **************************************')
    print('********Epoch loss = ' + str(np.average(loss_batch)))
    print('********Epoch Acc = ' + str(np.average(acc_batch)))
    loss_training.append(np.average(loss_batch))
    acc_training.append(np.average(acc_batch))
    print('tiempo del epoch: ', (time.time() - tiempo_inicial))


    #FOR TESTING
    loss_batch_test=[]
    loss_testing = []
    acc_batch_test=[]
    acc_testing = []
    tiempo_inicial = time.time()
    index_values_test = np.arange(0,int(len(testing_data)),maximo_imagenes_batch_test)
    j=0
    while ((j+1) < int(len(index_values_test))):
#     while (j<1):
        running_loss=0
        for k in range (0,int(batch_size/maximo_imagenes_batch_test)):
            if ((j+1) < int(len(index_values_test))):
                data_train_net = cargar_imagenes(testing_data[index_values_test[j]:index_values_test[j+1]], maximo_imagenes_batch_test)
                data_train_net = data_train_net.cuda()
                outputs = Red_Neuronal(data_train_net)
                if (k==0):
                    prediction_output = outputs.cpu().data.numpy()
                else:
                    prediction_output = np.vstack((prediction_output,outputs.cpu().data.numpy()))
                j=j+1

        label_train_net = testing_label[index_values_test[j-int(batch_size/maximo_imagenes_batch_test)]:index_values_test[j]]
        label_train_net = torch.tensor(label_train_net).cuda()
        prediction_output = torch.tensor(prediction_output).cuda()
        
        loss = criterion(prediction_output, label_train_net)
        # print statistics
        accuracy = Accuracy()
        accuracy = accuracy.cuda()
        acc_tmp = accuracy(prediction_output, torch.argmax(label_train_net, dim=1).cuda())
        running_loss += loss.item()
        acc_batch_test.append(acc_tmp.cpu().data.numpy())
        loss_batch_test.append(running_loss)
        print('************************************** TESTING **************************************')
        print('Batch loss = ' + str(running_loss))
        
        
    loss_batch_test = np.asarray(loss_batch_test)
    acc_batch_test = np.asarray(acc_batch_test)
    print('********Epoch loss = ' + str(np.average(loss_batch_test)))
    print('********Epoch Acc = ' + str(np.average(acc_batch_test)))
    loss_testing.append(np.average(loss_batch_test))
    acc_testing.append(np.average(acc_batch_test))
    print('tiempo de test: ', (time.time() - tiempo_inicial))

    #guardar perdida del batch
    np.savetxt('Perdida_Batch_Entrenamiento_%d.csv'%int(i), loss_batch, delimiter=",")
    np.savetxt('Accuracy_Batch_Entrenamiento_%d.csv'%int(i), acc_batch, delimiter=",")
    np.savetxt('Perdida_Batch_Evaluacion_%d.csv'%int(i), loss_batch_test, delimiter=",")
    np.savetxt('Accuracy_Batch_Evaluacion_%d.csv'%int(i), acc_batch_test, delimiter=",")

    loss_testing = np.asarray(loss_testing)
    acc_testing = np.asarray(acc_testing)
    loss_training = np.asarray(loss_training)
    acc_training = np.asarray(acc_training)
    #guardar perdida por epoch
    np.savetxt('Perdida_Entrenamiento.csv', loss_training, delimiter=",")
    np.savetxt('Accuracy_Entrenamiento.csv', acc_training, delimiter=",")
    np.savetxt('Perdida_Evaluacion.csv', loss_testing, delimiter=",")
    np.savetxt('Accuracy_Evaluacion.csv', acc_testing, delimiter=",")
    torch.save(Red_Neuronal, '3DCNN-basedSC.pt')