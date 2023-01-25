import tensorflow.compat.v1 as tf
#显示图片
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
import os
import tensorflow.compat.v1.gfile as gfile
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
# (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data(path='MNIST/mnist.npz')
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
tf.keras.datasets.fashion_mnist.load_data()
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()


fig = plt.figure()
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i],cmap='Greys')
    plt.title("recognize:{}".format(y_train[i]))
    # plt.xticks([])
    # plt.yticks([])
plt.show()

# 数据规范化-cnn准备

img_rows, img_cols = 28,28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)
print(x_train.shape,type(x_train))
print(x_test.shape,type(x_test))


## 进行归一化
X1_train = x_train.astype('float32')
X1_test = x_test.astype('float32')
X1_train /= 255
X1_test /=255
print('train samples :{}'.format(X1_train.shape[0]))
print('test samples :{}'.format(X1_test.shape[0]))


## 数据统计

label,count = np.unique(y_train,return_counts=True)
print(label,count)
fig=plt.figure()
plt.bar(label,count,width=0.7,align="center")
plt.xlabel("label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0,7500)

for a,b in zip(label,count):
    plt.text(a,b,'%d' % b, ha="center",va="bottom",fontsize=10)
plt.show()

# one-hot编码


n_class = 10
print("before one-hot: ",y_train.shape)
Y_train = np_utils.to_categorical(y_train,n_class)
print("after one-hot encoding: ", Y_train.shape)
Y_test = np_utils.to_categorical(y_test,n_class)




#使用sequential model定义mnist cnn网络

model = Sequential(name='Sequential')
# 从这里开始，都是在做特征提取
#第一层卷积，32个3*3的卷积核，激活函数用relu

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=input_shape,name='FirstConv2D'))
#第二层卷积，64个3*3的卷积核，激活函数用relu

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='SecodeConv2D'))

# 增加池化，池化窗口2*2
model.add(MaxPool2D(pool_size=(2,2),name='FirstMaxPool2D'))
#dropout 25%的输入神经元，每次训练的时候，要去掉25%的神经元
model.add(Dropout(0.25,name='FirstDropout_25'))

# 将pooled feature map 全都摊平以后，全连接网络
model.add(Flatten(name='FirstFlatten'))
# 从这里以上都是在做特征提取

# 这里才是在做分类

# 全连接层
model.add(Dense(128,activation='relu',name='FirstDenseRelu'))

# 再dropout掉50%的输入神经元
model.add(Dropout(0.5,name='SecodeDropout50'))
# 使用softmax激活函数做分类，输出各数字的概率

model.add(Dense(n_class,activation='softmax',name='SecondDense_Softmax'))
model.summary()
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())


# 编译模型
# with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
model.compile(loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'],optimizer='adam')
history=model.fit(X1_train,Y_train,batch_size=128,epochs=1,verbose=2,validation_data=(X1_test,Y_test))
    # writer = tf.summary.FileWriter('./summary/keras-mnist-cnn-1',sess.graph)
# writer.close()

# 可视化模型
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'],loc="upper right")
plt.tight_layout()
plt.show()

# 保存模型

plot_model(model,to_file="./summary/keras-mnist-cnn-2.png",show_shapes=True)
save_dir='./FASHION-MNIST/model-cnn/'
if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)
model_name='keras_mnist_cnn.h5'
model_path=os.path.join(save_dir,model_name)
model.save(model_path)


# 重新加载模型，并进行分类

mnist_model = load_model(model_path)
loss_and_metrics = mnist_model.evaluate(X1_test,Y_test,verbose=2)
print("Test loss is:{}% \n".format(loss_and_metrics[0]))
print("Test accuracy is :{}% \n".format(loss_and_metrics[1]*100))

predicted_class = mnist_model.predict(x_test)
correct_indices = np.nonzero(predicted_class==Y_test)[0]
incorrect_indices = np.nonzero(predicted_class!=Y_test)[0]
print('Classified correctly count: {}'.format(len(correct_indices)))
print("Classified incorrectly count: {}".format(len(incorrect_indices)))