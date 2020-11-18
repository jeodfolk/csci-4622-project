import cv2
from pywinauto import Desktop
from windowcapture import WindowCapture
from time import time

windows = Desktop(backend="uia").windows()
for i in windows:
    if i.window_text().startswith("GS"): 
        windowname = i.window_text()
        print(windowname)

wincap = WindowCapture(windowname)
screenshot = wincap.get_screenshot()
save = False
n = 0

while(True):
    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    cv2.imshow('a',screenshot)

    if save: 
        cv2.imwrite('/images/caps/ryu/ryu{}.jpg'.format(n), screenshot)

    if cv2.waitKey(1) == ord('q'): #quit
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) == ord('s'): #start saving images
        save = True
    n += 1