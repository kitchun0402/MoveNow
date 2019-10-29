import requests
import cv2
import numpy as np
from argparse import ArgumentParser
import pygame
import pygame.camera
from pygame import mixer
import time
from matplotlib import pyplot as plt
from main import start_game
# pygame.init()
# mixer.init()
# mixer.pre_init(16500, -16, 2, 2048)
# a = mixer.Sound('./sound_effect/Perfect.wav')
# a.set_volume(0.4)
# a.play()
# time.sleep(0.55)
# capture = cv2.VideoCapture(0)

# while True:
#     status, frame = capture.read()
#     cv2.imshow('123', frame)
#     if cv2.waitKey(1) and 0xFF == 'q':
#         break
# url = 'http://192.168.31.134:8080/shot.jpg'
# while True:
#     img_resp = requests.get(url)
#     img_array = np.array(bytearray(img_resp.content), dtype = 'uint8')
#     cv2_img = cv2.imdecode(img_array, -1) #-1 unchanged 0 greyscale 
#     cv2.imshow('123',cv2_img)
#     if cv2.waitKey(1) and 0xFF == 'q':
#         break
# img = np.zeros((200,300,3))

# img = cv2.flip(img,1)
# img = cv2.imread('/Users/kitchun/Desktop/FTDS/Github/Human_Pose_Deployment/MoveNow/1234.png')
# x = int(img.shape[1] * 0.73)
# x_new = 0


# for i in range(10):
#     cv2.rectangle(img, (int(img.shape[1] * 0.71), int(img.shape[0] * 0.42)), \
#         (x, int(img.shape[0] * 0.54)), (255, 229, 204), -1)
#     # cv2.rectangle(img, (x_new, 0), (x, 10), (0, 0, 255), cv2.FILLED)
#     x_new = x
#     x += 5
#     # cv2.putText(img, "Hello", (200,250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    
#     cv2.imshow('123',img)
#     cv2.moveWindow('123', 0, 0)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
normal = cv2.imread('./UI_images/button_normal.png')
width = int(normal.shape[1] * 0.58046875) - int(normal.shape[1] * 0.50546875)
height = int(normal.shape[0] * 0.1111111111111111) - int(normal.shape[0] * 0.05694444444444444)
print(normal.shape[1] * 0.671875)
normal = cv2.resize(normal, (width, height))
print(normal.shape)
cv2.imshow('12',normal)
cv2.waitKey(0)
# plt.imshow(img)
# plt.waitforbuttonpress()