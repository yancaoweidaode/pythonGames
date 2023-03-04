import skierGames as sk
import threading
import random
from data_collect import *
import time

sk.direction = 1

# 在这个线程里面获取数据并进行处理。
def get_data():
   while (True):
      time.sleep(1.01)                   # 这个和数据的更新有关，historylength设置成为250，LSL就没1s钟往buffer更新数据，根据模型计算一次所需要的实验来调整这个数字，可以使灵活的。
      print(buffer.begin)
      buffer.lock.acquire()
      data = buffer.get()
      buffer.lock.release()
      sk.direction = random.randint(1, 2)     # 这一行就用模型去计算数据，如果是左就左，右就右，啥也不是就不修改这个方向。
      print(data.shape)
      print(buffer.begin)
      print(data)
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
