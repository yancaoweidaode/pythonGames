import skierGames as sk
import threading
import random
from data_collect import *
import time
from model import ViT
from torch import nn
import torch
import numpy as np
from torch import Tensor
from torch.autograd import Variable

sk.direction = 1       # 1是左 2是右


# 模型的加载
PATH = 'model.pth'
Wb = np.load('csp_Wb.npy')
model = ViT()
model.load_state_dict(torch.load(PATH))
model = model.cuda()
model.eval()


# data = np.zeros((1,1,8,1000))
# data = np.einsum('abcd, ce -> abed', data, Wb)
# data = torch.from_numpy(data)
# data = Variable(data.type(torch.cuda.FloatTensor))
# Tok, Cls = model(data)
# y_pred = torch.max(Cls, 1)[1]
# print(y_pred.item())

# 在这个线程里面获取数据并进行处理。
def get_data():
   global model
   global Wb
   while (True):
      time.sleep(1.01)                   # 这个和数据的更新有关，historylength设置成为250，LSL就没1s钟往buffer更新数据，根据模型计算一次所需要的实验来调整这个数字，可以使灵活的。
      start_time = time.time()
      print(buffer.begin)
      buffer.lock.acquire()
      data = buffer.get()
      buffer.lock.release()
      data = data[np.newaxis,:]
      data = data[np.newaxis,:]
      data = np.einsum('abcd, ce -> abed', data, Wb)   # 做CSP空间滤波
      data = torch.from_numpy(data)
      data = Variable(data.type(torch.cuda.FloatTensor))
      Tok, Cls = model(data)
      y_pred = (torch.max(Cls, 1)[1]).item()
      print('预测的输出',y_pred)
      if (y_pred == 0):
         sk.direction = 2
      if (y_pred == 1):
         sk.direction = 1
      if (y_pred == 2):
         sk.direction= sk.direction
      end_time = time.time()
      print('预测一次所需要的时间', end_time-start_time)
      print(buffer.begin)
      print('*'*10)

      
th1 = threading.Thread(target=get_data)
th2 = threading.Thread(target=LSL)
th3 = threading.Thread(target=sk.game_play)
th1.start()
th2.start()
th3.start()
th1.join()
th2.join()
th3.join()
