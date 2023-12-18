#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   load_imdc.py
@Time    :   2023/12/18 12:15:28
@Author  :   xiaolikai 
@Version :   1.0
@Desc    :   利用imdb数据集 训练 LR
'''
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


max_len = 100
embd_dim = 8
vocab_size = 10000  # 只保留前10000个词
batch_size = 64
epoch = 50

def load_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # 将训练数据填充/截断至统一的长度
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    
    return x_train, y_train, x_test, y_test # (25000, 100) (25000,) (25000, 100) (25000)

def main():
    # 加载数据
    x_train, y_train, x_test, y_test = load_imdb()
    
    # 定义模型
    model = Sequential()
    model.add(Embedding(vocab_size, embd_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    # 训练模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                        validation_data=(x_test, y_test))
    
if __name__ == '__main__':
    main()