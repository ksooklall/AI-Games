import numpy as np
import pandas as pd
import cv2
from glob import glob

#data = np.concatenate([np.load('data/training_data_{}.npy'.format(i)) for i in range(2, 17) if i != 9])
data = np.concatenate([np.load(i) for i in glob('data/*.npy')])    
    
def show_data():
    for images in data:
        img = images[0]
        choice = images[1]
        cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        print(choice)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def down_sample(df):
    n_df = df[df['A'] == 1]
    dd = df[df['D'] == 1][:len(n_df)]
    jj = df[df['J'] == 1]
    return pd.concat([n_df, jj, dd])

#show_data()
df = pd.DataFrame(data)
bdf = pd.DataFrame(df[1].values.tolist(), columns=['A', 'D', 'J'])
print(bdf.sum())

sampled_df = down_sample(bdf)
sampled_df['frames'] = df[0].iloc[sampled_df.index]
sampled_df['button'] = list(zip(sampled_df['A'], sampled_df['D'], sampled_df['J']))

training_data = np.array([sampled_df['frames'], sampled_df['button']]).T
np.save('training_data_v1.npy', training_data)

