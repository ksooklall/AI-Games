import numpy as np
from models import alexnet

# Image size
WIDTH = 96
HEIGHT = 76

# Training parameters
LR = 1e-3
EPOCHS = 12
MODEL_NAME = 'sonic_run_-{}-{}-epochs.tfl'.format('alexnet', EPOCHS)

train_data = np.load('training_data_v1.npy')
outputs = 7

model = alexnet(WIDTH, HEIGHT, outputs, LR)

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
y = np.array([i[1] for i in train])

test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

# Train in command line
def train():
    model.fit(X, y, n_epoch=EPOCHS,
              validation_set=(test_X, test_y),
              snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

    # tensorboard --logdir=foo:E:/GensisAI/Sonic The Hedgehog/log
    model.save('trained_models/{}'.format(MODEL_NAME))
#train()
