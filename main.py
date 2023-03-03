import skierGames as sk
import threading
import random

sk.direction = 1
def get_direction():
   while (True):
     sk.direction = random.randint(1, 2)
    #  sk.direction = int(input('direction is: '))


th1 = threading.Thread(target=get_direction)
th2 = threading.Thread(target=sk.game_play)
th1.start()
th2.start()
th1.join()
th2.join()
