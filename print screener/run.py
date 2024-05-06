import pyautogui
from time import sleep
import os
import winsound
STEPS=400

pyautogui.FAILSAFE = True
sleep(5)
for i in range(10):
    print(pyautogui.position())
    sleep(0.1)
1/0
#Point(x=3406, y=1554) reset pos
#x=3401, y=1392 calculate pos
def make_screenshots(index):
    pyautogui.click(x=3406, y=1554, button='left')
    sleep(0.1)
    pyautogui.moveTo(x=914, y=1010)  
    pyautogui.dragRel(10, -150, duration=0.5)
    sleep(0.3)
    pyautogui.screenshot(f"images\\rotated_step_{index:03}.jpg",region=(0,30,3100,2000))  
    pyautogui.dragRel(-10, 150, duration=0.5)
    sleep(0.3)
    pyautogui.screenshot(f"images\\step_{index:03}.jpg",region=(0,30,3100,2000))  
    

# make_screenshots(0)
for i in range(1,STEPS+1):
    # pyautogui.click(x=3401, y=1392, button='left')
    print(pyautogui.mouseinfo.position)
    sleep(1+i/100)
    # make_screenshots(i)


winsound.Beep(1000, 100)  # Beep at 1000 Hz for 100 ms