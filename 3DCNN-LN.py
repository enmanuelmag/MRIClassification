import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imageio
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import ants
import time

path_file = 'DB'

#Set multiple GPUs
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)


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
  Dato =[]
  for j in name:
    img_path = str(j)
    #print(img_path)
    img = ants.image_read(img_path)
    Dato.append(img.numpy())
    # print(len(Dato))

  Dato = np.asarray(Dato)
#   print(Dato.shape)
  #data normalizaion
  Dato=((Dato - np.amin(Dato))/(np.amax(Dato) - np.amin(Dato)))
  print(np.amax(Dato))
  print(np.amin(Dato))
  np.reshape(Dato, (num, 197, 233, 189, 1))
  # print(Dato.shape)
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

print('******************************************************************')
print('Data shapes:')
print(training_data.shape)
print(training_label.shape)
print(testing_data.shape)
print(testing_label.shape)
print('******************************************************************')


#SEPARABLE CONVOLUTION = DEPTHWISE + COMBINATION CONV STEP
with strategy.scope():
   main_input = Input(shape=(197, 233, 189, 1),dtype='float32',name='main_input')
   First_input_C1 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(main_input)
   First_input_C1_Batch = BatchNormalization(axis=4)(First_input_C1)
   First_Pooling = MaxPool3D(pool_size=(2,2,2))(First_input_C1_Batch)
   First_input_C2 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(First_Pooling)
   First_input_C2_Batch = BatchNormalization(axis=4)(First_input_C2)
   First_input_C3 = Conv3D(padding="same",activation='relu',filters=16, kernel_size=(3, 3, 3))(First_input_C2_Batch)
   First_input_C3_Batch = BatchNormalization(axis=4)(First_input_C3)
   Dropout_layer_1 = Dropout(0.3)(First_input_C3_Batch)

   CNN_C6 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_1)
   CNN_C6_Batch = BatchNormalization(axis=4)(CNN_C6)

   CNN_C7 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C6_Batch)
   CNN_C7_Batch = BatchNormalization(axis=4)(CNN_C7)
   # Dropout_layer_2 = Dropout(0.35)(CNN_C7_Batch)
   CNN_C8 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C7_Batch)
   CNN_C8_Batch = BatchNormalization(axis=4)(CNN_C8)
   layer_Addition_1 = Add()([CNN_C8_Batch,CNN_C6_Batch])
   # CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_1)

   CNN_C74 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_1)
   CNN_C74_Batch = BatchNormalization(axis=4)(CNN_C74)
   # Dropout_layer_2 = Dropout(0.35)(CNN_C7_Batch)
   CNN_C84 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C74_Batch)
   CNN_C84_Batch = BatchNormalization(axis=4)(CNN_C84)
   layer_Addition_14 = Add()([CNN_C84_Batch,layer_Addition_1])
   # CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_1)

   CNN_C73 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_14)
   CNN_C73_Batch = BatchNormalization(axis=4)(CNN_C73)
   Dropout_layer_23 = Dropout(0.35)(CNN_C73_Batch)
   CNN_C83 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_23)
   CNN_C83_Batch = BatchNormalization(axis=4)(CNN_C83)
   layer_Addition_13 = Add()([CNN_C83_Batch,layer_Addition_14])
   CNN_Pooling_1 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_13)

   CNN_C9 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_Pooling_1)
   CNN_C9_Batch = BatchNormalization(axis=4)(CNN_C9)
   # Dropout_layer_3 = Dropout(0.40)(CNN_C9_Batch)
   CNN_C10 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C9_Batch)
   CNN_C10_Batch = BatchNormalization(axis=4)(CNN_C10)
   layer_Addition_2 = Add()([CNN_C10_Batch,CNN_Pooling_1])
   # CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_2)

   CNN_C93 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_2)
   CNN_C93_Batch = BatchNormalization(axis=4)(CNN_C93)
   # Dropout_layer_3 = Dropout(0.40)(CNN_C9_Batch)
   CNN_C103 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C93_Batch)
   CNN_C103_Batch = BatchNormalization(axis=4)(CNN_C103)
   layer_Addition_23 = Add()([CNN_C103_Batch,layer_Addition_2])
   # CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_2)

   CNN_C92 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_23)
   CNN_C92_Batch = BatchNormalization(axis=4)(CNN_C92)
   Dropout_layer_32 = Dropout(0.40)(CNN_C92_Batch)
   CNN_C102 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_32)
   CNN_C102_Batch = BatchNormalization(axis=4)(CNN_C102)
   layer_Addition_22 = Add()([CNN_C102_Batch,layer_Addition_23])
   CNN_Pooling_2 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_22)
    
   CNN_C91 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_Pooling_2)
   CNN_C91_Batch = BatchNormalization(axis=4)(CNN_C91)

   CNN_C11 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C91_Batch)
   CNN_C11_Batch = BatchNormalization(axis=4)(CNN_C11)
   # Dropout_layer_4 = Dropout(0.45)(CNN_C11_Batch)
   CNN_C12 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C11_Batch)
   CNN_C12_Batch = BatchNormalization(axis=4)(CNN_C12)
   layer_Addition_3 = Add()([CNN_C12_Batch,CNN_C91_Batch])
   # CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_3)
    
   CNN_C911 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_3)
   CNN_C911_Batch = BatchNormalization(axis=4)(CNN_C911)

   CNN_C111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C911_Batch)
   CNN_C111_Batch = BatchNormalization(axis=4)(CNN_C111)
   # Dropout_layer_41 = Dropout(0.45)(CNN_C111_Batch)
   CNN_C121 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C111_Batch)
   CNN_C121_Batch = BatchNormalization(axis=4)(CNN_C121)
   layer_Addition_31 = Add()([CNN_C121_Batch,CNN_C911_Batch])
   # CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_31)
    
   CNN_C9111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(layer_Addition_31)
   CNN_C9111_Batch = BatchNormalization(axis=4)(CNN_C9111)

   CNN_C1111 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(CNN_C9111_Batch)
   CNN_C1111_Batch = BatchNormalization(axis=4)(CNN_C1111)
   Dropout_layer_411 = Dropout(0.45)(CNN_C1111_Batch)
   CNN_C1211 = Conv3D(padding="same", activation='relu',filters=16,kernel_size=(3, 3, 3))(Dropout_layer_411)
   CNN_C1211_Batch = BatchNormalization(axis=4)(CNN_C1211)
   layer_Addition_311 = Add()([CNN_C1211_Batch,CNN_C9111_Batch])
   CNN_Pooling_3 = MaxPool3D(pool_size=(2,2,2))(layer_Addition_311)

   flatten_layer = Flatten()(CNN_Pooling_3)
   dense_layer_1 = Dense(units=100,activation="relu")(flatten_layer)
   Dropout_layer_5 = Dropout(0.20)(dense_layer_1)
   dense_layer_2 = Dense(units=50,activation="relu")(Dropout_layer_5)
   output_layer = Dense(units=2,activation="softmax")(dense_layer_2)

   Tri_dim_model = Model(inputs=[main_input],outputs=[output_layer])
   Tri_dim_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

#dropout incremental por cada bloque


# #load model
# with strategy.scope():
#     Tri_dim_model = keras.models.load_model("3DCNN-basedSC.h5")
#     Tri_dim_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    

print(Tri_dim_model.summary())

epochs_number = 100
batch_size = 5
maximo_imagenes_batch = 5
loss_epoch=[]
acc_epoch=[]
loss_epoch_test=[]
acc_epoch_test=[]
print('Total Training Data ' + str(len(training_data)))
for i in range (0, epochs_number):
  loss_batch=[]
  acc_batch=[]
  loss_batch_test=[]
  acc_batch_test=[]
  tiempo_inicial = time.time()

  for j in range(0,int(len(training_data))):
    if ((j+1)*maximo_imagenes_batch)<len(training_data):
      tiempo_inicial_batch = time.time()
      data_train_net = cargar_imagenes(training_data[j*maximo_imagenes_batch:(j+1)*maximo_imagenes_batch], maximo_imagenes_batch)
      label_train_net = training_label[j*maximo_imagenes_batch:(j+1)*maximo_imagenes_batch]
      print('***********************************************************************************************************')
      print('Numero de iterador: ', (j+1))
      print('Datos Cargados')
      Tri_dim_model.fit(data_train_net, label_train_net, batch_size=batch_size)
      #guardar parametros training DB por epoch
      perdida, precision = Tri_dim_model.evaluate(data_train_net, label_train_net, batch_size=2)
      print('Batch Loss = ' + str(perdida))
      print('Batch Accuracy = ' + str(precision))
      loss_batch.append(perdida)
      acc_batch.append(precision)
      print('tiempo de 1 superbatch: ', (time.time() - tiempo_inicial_batch))
        
  for j in range(0,int(len(testing_data))):
    if ((j+1)*maximo_imagenes_batch)<len(testing_data):
      data_test_net = cargar_imagenes(testing_data[j*maximo_imagenes_batch:(j+1)*maximo_imagenes_batch], maximo_imagenes_batch)
      label_test_net = testing_label[j*maximo_imagenes_batch:(j+1)*maximo_imagenes_batch]
      perdida_test, precision_test = Tri_dim_model.evaluate(data_test_net, label_test_net, batch_size=2)
      loss_batch_test.append(perdida_test)
      acc_batch_test.append(precision_test)
          
    
  loss_batch = np.asarray(loss_batch)
  acc_batch = np.asarray(acc_batch)
    
  loss_batch_test = np.asarray(loss_batch_test)
  acc_batch_test = np.asarray(acc_batch_test)
  
  print('********************Epoch ' + str(i) + '********************')
  ########################################## GUARDAR PERDIDA DEL BATCH
  print('tiempo del epoch: ', (time.time() - tiempo_inicial))
  print('Loss = ' + str(np.average(loss_batch)))
  loss_epoch.append(np.average(loss_batch))
  print('Accuracy = ' + str(np.average(acc_batch)))
  acc_epoch.append(np.average(acc_batch))
    
  loss_epoch_test.append(np.average(loss_batch_test))
  acc_epoch_test.append(np.average(acc_batch_test))

#guardar perdida del batch
  np.savetxt('Perdida_Entrenamiento_%d.csv'%int(i), loss_batch, delimiter=",")
  np.savetxt('Accuracy_Entrenamiento_%d.csv'%int(i), acc_batch, delimiter=",")
  np.savetxt('Perdida_Evaluacion_%d.csv'%int(i), loss_batch_test, delimiter=",")
  np.savetxt('Accuracy_Evaluacion_%d.csv'%int(i), acc_batch_test, delimiter=",")

#guardar perdida por batch
  np.savetxt('Perdida_Entrenamiento.csv', loss_epoch, delimiter=",")
  np.savetxt('Accuracy_Entrenamiento.csv', acc_epoch, delimiter=",")
  np.savetxt('Perdida_Evaluacion.csv', loss_epoch_test, delimiter=",")
  np.savetxt('Accuracy_Evaluacion.csv', acc_epoch_test, delimiter=",")
  Tri_dim_model.save("3DCNN-basedSC.h5")
  

loss_epoch = np.asarray(loss_epoch)
plt.plot(loss_epoch)
plt.savefig("Grafica Perdida")

acc_epoch = np.asarray(acc_epoch)
plt.plot(acc_epoch)
plt.savefig("Grafica Accuracy")