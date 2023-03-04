import numpy as np
import threading
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max, historylength):
        self.max = size_max
        self.historylength = historylength
        self.data = np.zeros((8, size_max), dtype = float)
        self.begin = 0
        self.end = 0
        self.nums = 1000 / historylength
        self.lock = threading.Lock()
    
    def append(self,x):
        """append an element at the end of the buffer"""
        self.data[:,self.end:self.end+self.historylength] = x
        self.end = (self.end+self.historylength) % self.max

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        if (self.begin + 1000 < self.max):
            data = self.data[:,self.begin:self.begin+1000]
            self.begin = (self.begin+self.historylength) % self.max
        else:
            data = np.concatenate((self.data[:,self.begin:self.max], self.data[:, 0:(1000-(self.max-self.begin))]), axis=1)
            self.begin = (self.begin+self.historylength) % self.max
        return data

# x = RingBuffer(2000, 100)
# for i in range(12):
#     data = np.zeros((8, 3))
#     for j in range(8):
#         for k in range(3):
#             data[j, k] = i
#     x.append(data)

# for j in range(13):
#     data = x.get()
#     print(data.shape)
#     print(data)
#     print("*"*10)
# # import time
# tm1 = time.time()
# for i in range(int(1e6)):
#     x.append(i)

# print(x.get()[:10])
# tm2 = time.time()
# print("{:.2f}seconds".format(tm2 - tm1))
# x = np.zeros((10, 20))
# a = np.concatenate((x[:, 0:10], x[:, 10:20]), axis = 1)
# x[0,0] = 1
# print(a.shape)
# print(x.shape)
# print(a)