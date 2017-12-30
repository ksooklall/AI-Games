import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import pickle

train_data = np.load('training_data_3.npy')

def show_data():
    for data in train_data:
        img = data[0]
        choice = data[1]
        cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        print(choice)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
show_data()
df = pd.DataFrame(train_data)
bdf = pd.DataFrame(df[1].values.tolist(), columns=['A', 'D', 'J'])
print(bdf.sum())

