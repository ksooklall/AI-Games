import os
import cv2
import time
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelBinarizer

from get_keys import key_check
from GetFrame import GetFrameThread
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
    data_path  = 'batch_data/*.npy'
    batch = max([int(i.split('_')[-1].split('.')[0]) for i in glob(data_path)])+1
    return batch, []

if __name__ == '__main__':
    count_down(9)
    start_time = time.time()
    batch, training_data = create_files()
    file_name = 'batch_data/training_data_{}'.format(batch)

    game = 'Fusion 3.64 - Genesis - SONIC THE               HEDGEHOG'
    thread = GetFrameThread(0, 50, 960, 720, window_title_substring=game).start()

    ## To do: Determine how to choose what monitor is being detected
    while True:
        if time.time() - start_time  > 0.035:
            screen = thread.return_frame()
            if screen is not None:
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                screen = cv2.resize(screen, (WIDTH, HEIGHT))

                keys = key_check()
                output = keys_to_output(keys)
                training_data.append([screen, output])

                if len(training_data) % 100 == 0:
                    thread.get_fps()
                    
                if len(training_data) > 1000:
                    print('Saved file: {}'.format(batch))
                    np.save(file_name, training_data)
                    thread.stop_now()
                    print('Done')
                    break
            start_time = time.time()    
