from testing.getFrame import GetFrameThread
import numpy as np
import cv2
from helper import count_down
from glob import glob
import time

game = 'Fusion 3.64 - Genesis - SONIC THE               HEDGEHOG'
thread = GetFrameThread(0, 50, 960, 720, window_title_substring=game).start()
testing_data = []
def play_video(img):
    for screen in img:
        im = cv2.resize(screen, (960, 720))
        cv2.imshow('frame', im)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break       
count_down(0)
batch = 0
start_time = time.time()
while True:

    if time.time() - start_time  > 0.0125:
        screen = thread.return_frame()
        
        if screen is not None:
            img = cv2.resize(screen, (96, 72))
            testing_data.append(img)
            
            #thread.get_fps()
            if len(testing_data)>5000:
                thread.get_fps()
                print('Saved: {}'.format(batch))
                np.save('testing/data/testing_fps_{}.npy'.format(batch), testing_data)
                batch += 1
                testing_data=[]
        
            if batch > 10:
                thread.stop_now()
                print('Done')
                break
        start_time = time.time()    
#data = np.concatenate([np.load(i) for i in glob('testing/data/*.npy')])
#play_video(data)
