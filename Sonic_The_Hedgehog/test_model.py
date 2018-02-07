import time
import cv2
import numpy as np
import tensorflow as tf
from models import Models
from GetFrame import GetFrameThread
from train_model import WIDTH, HEIGHT, LR, MODEL_NAME
from get_keys import key_check
from directkeys import PressKey, ReleaseKey, A, D, J, S

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(J)

def left():
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(J)

def jump():
    PressKey(J)
    ReleaseKey(D)
    ReleaseKey(A)

def right_jump():
    PressKey(D)
    PressKey(J)
    ReleaseKey(S)
    ReleaseKey(A)

def left_jump():
    PressKey(A)
    PressKey(J)
    ReleaseKey(S)
    ReleaseKey(D)

def down():
    PressKey(S)
    ReleaseKey(J)
    ReleaseKey(A)
    ReleaseKey(D)

def do_nothing():
    ReleaseKey(S)
    ReleaseKey(J)
    ReleaseKey(A)
    ReleaseKey(D)
    
def count_down(n):
    for i in range(n)[::-1]:
        print(i, end=' ')
        time.sleep(1)

tf.reset_default_graph()
model = Models(WIDTH, HEIGHT, 7, LR).alexnet_2()
model.load('trained_models/{}'.format(MODEL_NAME))

paused = False
count_down(3)
start_time = time.time()
game = 'Fusion 3.64 - Genesis - SONIC THE               HEDGEHOG'
thread = GetFrameThread(0, 50, 960, 720, window_title_substring=game).start()


    ## To do: Determine how to choose what monitor isd being detected
while(True):
    if not paused:
        if time.time() - start_time  > 0.035:
            screen = thread.return_frame()
            if screen is not None:
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                screen = cv2.resize(screen, (WIDTH, HEIGHT))

                moves = model.predict([screen.reshape(WIDTH, HEIGHT,1)])[0]
                
                if np.argmax(moves) == 1:
                    left()
                elif np.argmax(moves) == 6:
                    down()
                elif np.argmax(moves) == 4:
                    right_jump()
                elif np.argmax(moves) == 3:
                    right()
                elif np.argmax(moves) == 5:
                    jump() 
                elif np.argmax(moves) == 2:
                    left_jump()
                elif np.argmax(moves) == 7:
                    do_nothing()
            start_time = time.time()
                    
    keys = key_check()
    if 't' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
