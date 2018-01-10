import os
import cv2
import time
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelBinarizer

from get_keys import key_check
from grab_screen import grab_screen
from helper import count_down


# Image size
WIDTH = 96
HEIGHT = 76

keys = ['A', 'S', 'D', 'J', ' ', 'DJ', 'AJ']
one_hot = LabelBinarizer().fit_transform(keys)
key_mapping = dict(zip(keys, one_hot))
def keys_to_output(keys):
    if 'D' in keys and 'J' in keys:
        return key_mapping['DJ']
    elif 'A' in keys and 'J' in keys:
        return key_mapping['AJ']
    return key_mapping.get(keys[-1] if keys else 'J', key_mapping['J'])

def create_files():
    data_path  = 'data/*.npy'
    batch = max([int(i.split('_')[-1].split('.')[0]) for i in glob(data_path)])+1
    return batch, []

if __name__ == '__main__':
    count_down(9) 
    batch, training_data = create_files()
    file_name = 'data/training_data_{}'.format(batch)
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
            file_name = 'data/training_data_{}'.format(batch)            
