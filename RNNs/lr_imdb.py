#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   load_imdc.py
@Time    :   2023/12/18 12:15:28
@Author  :   xiaolikai 
@Version :   1.0
@Desc    :   利用imdb数据集 训练 LR
'''
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.optimizers import Adam

import matplotlib.pyplot as plt


max_len = 100
embd_dim = 8
vocab_size = 10000  # 只保留前10000个词
batch_size = 128
epoch = 50
learning_rate = 1e-4

def load_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # 将训练数据填充/截断至统一的长度
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    
    # 随机切分生成验证集
    indices = np.random.permutation(len(x_train))
    train_size = int(len(x_train) * 0.8)
    
    x_val = x_train[indices[train_size:], :]
    y_val = y_train[indices[train_size:]]
    x_train = x_train[indices[:train_size], :]
    y_train = y_train[indices[:train_size]]
    
    
    return x_train, y_train, x_val, y_val, x_test, y_test # (25000, 100) (25000,) (25000, 100) (25000)

def visualize_training(history):
    """将模型训练结果可视化

    Args:
        history (_type_): 模型训练历史日志
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    # 绘制损失函数图表
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # 绘制准确度图表
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()
    
def main():
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test = load_imdb()
    
    # 定义模型
    model = Sequential()
    model.add(Embedding(vocab_size, embd_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    # 训练模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                        validation_data=(x_val, y_val))
    
    # 测试集上评估模型
    loss_and_auc = model.evaluate(x_test, y_test)
    print('#' * 10)
    print('loss = ' + str(loss_and_auc[0]))
    print('auc = ' + str(loss_and_auc[1]))
    # 将训练结果绘制成图
    visualize_training(history)
    
    
if __name__ == '__main__':
    main()