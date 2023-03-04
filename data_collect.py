import pyqtgraph as pg
import array
from pylsl import StreamInlet,resolve_stream
import numpy as np
import threading
import time
import cv2 as cv
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import time
from cycle_buffer import RingBuffer
import threading


buffer = RingBuffer(10000, 250)     # 可以存储40s的数据
def LSL():#线程1用于接受LSL数据
    global buffer
    historylength = 250#横坐标长度
    data = np.zeros((8, historylength), dtype = float)
    global index
    i = 0
    streams = resolve_stream('type', 'EEG')  # 类型一致则可以接受到数据
    # !问题定位在这里，应该是没有接收到数据，所以一直卡在这个函数里面
    # create a new inlet to read from stream
    inlet = StreamInlet(streams[0])
    while(True):
        sample, timestamp = inlet.pull_sample()  # timestamp 时间戳 应该是这个地方的问题
        if(i < historylength):
            data[0][i] = sample[0]
            data[1][i] = sample[1]
            data[2][i] = sample[2]
            data[3][i] = sample[3]
            data[4][i] = sample[4]
            data[5][i] = sample[5]
            data[6][i] = sample[6]
            data[7][i] = sample[7]
            i = i + 1
        else:#采集到1s的数据后进行处理 这里直接保存原始数据
            i = 0
            buffer.lock.acquire()
            buffer.append(data)
            buffer.lock.release()