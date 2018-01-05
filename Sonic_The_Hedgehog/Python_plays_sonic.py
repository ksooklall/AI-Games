import numpy as np
import cv2
import time
from get_keys import key_check
from grab_screen import grab_screen
import os
from train_model import WIDTH, HEIGHT

def keys_to_output(keys):
    output = [0, 0, 0]
    if 'A' in keys:     # Left
        output[0] = 1
    elif 'J' in keys:   # Jump
        output[2] = 1
    else:                # Right
        output[1] = 1
    return output

def process_img(img):
    # Use for edge detection
    return cv2.Canny(img, threshold1=70, threshold2=140)

def show_screen(screen):
    cv2.imshow('window', screen)
    if cv2.waitKey(25) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        return True
    return False

def count_down(n):
    for i in range(n)[::-1]:
        print(i, end=' ')
        time.sleep(1)

file_name  = 'data/training_data_3.npy'
if os.path.isfile(file_name):
    print('Loading: {}'.format(file_name))
    training_data = list(np.load(file_name))
else:
    print('Creating new file')
    training_data = []
batch = 9

if __name__ == '__main__':
    count_down(10)

    ## To do: Determine how to choose what monitor is being detected
    while(True):
        #last = time.time()
        screen = grab_screen(region=(0, 50, 960, 760))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (WIDTH, HEIGHT))

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])
        #cv2.imshow('window', screen)
        #if cv2.waitKey(25) & 0xFF ==ord('q'):
        #    cv2.destroyAllWindows()
        #    break
        #print('FPS: {}'.format(time.time()-last))

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
            print('Saved file')
            training_data = []
            batch += 1
            file_name = 'training_data_{}'.format(batch)
