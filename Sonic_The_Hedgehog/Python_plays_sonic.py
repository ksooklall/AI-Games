import numpy as np
import cv2
import time
from get_keys import key_check
from grab_screen import grab_screen
import os
#from train_model import WIDTH, HEIGHT
from helper import count_down
from sklearn.preprocessing import LabelBinarizer

# Image size
WIDTH = 96
HEIGHT = 76

keys = ['A', 'S', 'D', 'J', ' ']
one_hot = LabelBinarizer().fit_transform(keys)
key_mapping = dict(zip(keys, one_hot))
def keys_to_output(keys):
##    output = [0, 0, 0, 0]
##    if 'A' in keys:     # Left
##        output[0] = 1
##    elif 'D' in keys:   # Right
##        output[2] = 1
##    elif 'S' in keys:   # Down
##        output[1] = 1
##    else:               # Jump
##        output[3] = 1
##    return output
    return key_mapping.get(keys[-1] if keys else 'J', key_mapping['J'])

if __name__ == '__main__':
    file_name  = 'data/training_data_1.npy'
    if os.path.isfile(file_name):
        print('Loading: {}'.format(file_name))
        training_data = list(np.load(file_name))
    else:
        print('Creating new file')
        training_data = []
    batch = 1

    count_down(9)
    ## To do: Determine how to choose what monitor is being detected
    while(True):
        #last = time.time()
        screen = grab_screen(region=(0, 50, 960, 760))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (WIDTH, HEIGHT))

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])
##        cv2.imshow('window', screen)
##        if cv2.waitKey(25) & 0xFF ==ord('q'):
##            cv2.destroyAllWindows()
##            break
        #print('FPS: {}'.format(time.time()-last))

        if not len(training_data) % 1000:
            np.save(file_name, training_data)
            print('Saved file: {}'.format(batch))
            training_data = []
            batch += 1
            file_name = 'training_data_{}'.format(batch)            
