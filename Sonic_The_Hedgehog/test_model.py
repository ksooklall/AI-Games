import time
import cv2
import numpy as np
import tensorflow as tf
from models import alexnet
from git_grab_screen import grab_screen
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

def count_down(n):
    for i in range(n)[::-1]:
        print(i, end=' ')
        time.sleep(1)
        
tf.reset_default_graph()

model = alexnet(WIDTH, HEIGHT, 7, LR)
model.load('trained_models/{}'.format(MODEL_NAME))

pause = False
def main():
    count_down(2)
    ## To do: Determine how to choose what monitor isd being detected
    while(True):
        if not pause:
            #last = time.time()
            screen = grab_screen(region=(0, 50, 960, 760))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            moves = model.predict([screen.reshape(WIDTH, HEIGHT,1)])[0]
            print(moves)
            
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

            keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
main()
