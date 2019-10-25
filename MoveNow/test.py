import requests
import cv2
import numpy as np
from argparse import ArgumentParser
capture = cv2.VideoCapture(0)

while True:
    status, frame = capture.read()
    cv2.imshow('123', frame)
    if cv2.waitKey(1) and 0xFF == 'q':
        break
# url = 'http://192.168.31.134:8080/shot.jpg'
# while True:
#     img_resp = requests.get(url)
#     img_array = np.array(bytearray(img_resp.content), dtype = 'uint8')
#     cv2_img = cv2.imdecode(img_array, -1) #-1 unchanged 0 greyscale 
#     cv2.imshow('123',cv2_img)
#     if cv2.waitKey(1) and 0xFF == 'q':
#         break
# img = np.zeros((300,300,3))
# cv2.rectangle(img, (200, 200), (300, 300), (0, 0, 255), cv2.FILLED)
# cv2.putText(img, "Hello", (200,250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
# cv2.imshow('123',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

