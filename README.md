# VGG19 Hand Gesture Classification Model

## 資料集:

<img src="https://github.com/Rainman5042/Hand-Gesture-Classification/blob/main/hand.JPG?raw=true" width=50%>

此資料集英文字母手勢灰階圖片，其中J以及Z為動態手勢，故去除不用，總共24種不同手勢圖片，共2400張。每種手勢有100張圖片。

## 運行環境:

Python 3.5

Jupyter Notebook

Tensorflow 1.13

Keras 2.2.4

tqdm

Scikit-learn 0.21.2

OpenCV 4.1.0

PIL 5.1.0

Maltplotlib

Numpy

Pandas

## 前處理:

1.使用OpenCV讀取影像並刪除圖片右邊黑條:

<img src="https://github.com/Rainman5042/Hand-Gesture-Classification/blob/main/img1.JPG?raw=true" width=40%>

2.使用Otsu做二值化:

<img src="https://github.com/Rainman5042/Hand-Gesture-Classification/blob/main/img2.JPG?raw=true" width=40%>

3.將二值化後的影像做膨脹處理，與原圖相疊去除背景:

<img src="https://github.com/Rainman5042/Hand-Gesture-Classification/blob/main/img3.JPG?raw=true" width=40%>

4.使用OpenCV定位影像並擷取影像，將影像resize成40x40:

<img src="https://github.com/Rainman5042/Hand-Gesture-Classification/blob/main/img4.JPG?raw=true" width=40%>

5.Label使用 One-Hot Encode 處理。

## 模型架構:
使用預訓練的VGG19作為主要模型:

```
def vgg19_model(input_shape):
    vgg19 = VGG19(include_top=False, weights='imagenet',input_shape=input_shape)
    for layer in vgg19.layers:
        layer.trainable = False
    last = vgg19.output
    x = Flatten()(last)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg19.input, outputs=x)
    return model

```

## 訓練模型:

將2400筆圖片以8:2分為訓練以及測試資料

Optimizer = Adam

Epochs = 15

Learning Rate = 0.001

Batch size = 64

## 訓練過程&結果:

```
train_history=model_vgg19.fit(x = img_train,
                              y = label_train_onehot, validation_split=0.1,
                              epochs=Epochs, batch_size=Batch_size,verbose=2)  
```

```
Train on 1728 samples, validate on 192 samples
Epoch 1/15
 - 3s - loss: 6.7590 - acc: 0.3362 - val_loss: 2.7318 - val_acc: 0.7292
Epoch 2/15
 - 1s - loss: 2.2208 - acc: 0.7118 - val_loss: 0.6351 - val_acc: 0.8958
Epoch 3/15
 - 1s - loss: 0.5327 - acc: 0.8652 - val_loss: 0.1562 - val_acc: 0.9792
Epoch 4/15
 - 1s - loss: 0.2459 - acc: 0.9259 - val_loss: 0.1001 - val_acc: 0.9844
Epoch 5/15
 - 1s - loss: 0.1388 - acc: 0.9537 - val_loss: 0.0775 - val_acc: 0.9896
Epoch 6/15
 - 1s - loss: 0.1092 - acc: 0.9641 - val_loss: 0.0558 - val_acc: 0.9896
Epoch 7/15
 - 1s - loss: 0.0729 - acc: 0.9792 - val_loss: 0.0249 - val_acc: 0.9948
Epoch 8/15
 - 1s - loss: 0.0721 - acc: 0.9780 - val_loss: 0.0370 - val_acc: 0.9896
Epoch 9/15
 - 1s - loss: 0.0553 - acc: 0.9809 - val_loss: 0.0301 - val_acc: 0.9948
Epoch 10/15
 - 2s - loss: 0.0479 - acc: 0.9850 - val_loss: 0.0334 - val_acc: 0.9896
Epoch 11/15
 - 2s - loss: 0.0423 - acc: 0.9850 - val_loss: 0.0398 - val_acc: 0.9896
Epoch 12/15
 - 2s - loss: 0.0411 - acc: 0.9907 - val_loss: 0.0424 - val_acc: 0.9844
Epoch 13/15
 - 2s - loss: 0.0313 - acc: 0.9896 - val_loss: 0.0126 - val_acc: 1.0000
Epoch 14/15
 - 2s - loss: 0.0296 - acc: 0.9907 - val_loss: 0.0125 - val_acc: 0.9948
Epoch 15/15
 - 2s - loss: 0.0230 - acc: 0.9942 - val_loss: 0.0098 - val_acc: 0.9948
```

準確率約可達到99%

```
scores = model_vgg19.evaluate(img_val , label_val_onehot)
scores[1]
0.9958333333333333

```


## 如何運行 :

將以下檔案解壓縮到與VGG19.ipynb同個資料夾

[Dataset Download Link](https://drive.google.com/file/d/1b-WILq5S1Q_F3LSSAz7KdmF_jdmQouzV/view?usp=sharing)

執行VGG19.ipynb
