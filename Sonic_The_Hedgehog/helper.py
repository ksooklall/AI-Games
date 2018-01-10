import time
import cv2

def process_img(img):
    # Use for edge detection
    return cv2.Canny(img, threshold1=70, threshold2=140)

def show_screen(screen):
    print('Press q to exit')
    cv2.imshow('window', screen)
    if cv2.waitKey(25) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()

def count_down(n):
    for i in range(n)[::-1]:
        print(i, end=' ')
        time.sleep(1)
    print()
