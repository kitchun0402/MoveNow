import cv2
import time
from pose_models import LoadModel, PredictPose
from evaluate import Evaluate
from PoseNet.utils import valid_resolution
from annotation import annotation
from argparse import ArgumentParser
import requests
import numpy as np
from utils import bounding_box_coordinates, normalization, zoomin_point, resize_point, centralized_keypoint, find_palm_xy, overlay_transparent
from game_tools import find_box, gamebox
import os
import random
import json
from pygame import mixer
import math
from battle_mode import Battle_Mode
from normal_mode import Normal_Mode
from start_game import Start_Game

ap =  ArgumentParser()
ap.add_argument('-name', dest = 'pose_id', type = str, default = '0', help = "Your Name to display")
ap.add_argument('--ip-webcam', action = 'store_true', default = False, help = "Use IP Webcam application")
ap.add_argument('-ip','--ip-address', type = str, default = None, help = "Input your IP shown on IP Webcam application")
ap.add_argument('--weight-dir', type = str, default = "./model_", help = "Path to the model weight")
ap.add_argument('--model-id', type = int, choices = [50, 75, 100, 101], default = 101)
ap.add_argument('--output-stride', type = int, default = 16)
ap.add_argument('--useGPU', action = 'store_true', default = False)
ap.add_argument('--verbose', action = 'store_true', default = False)
ap.add_argument('--scale-factor', type = float, default = 1, choices = np.arange(0, 1.1, 0.1).round(2), help = "factor to scale down the image to process")
ap.add_argument('-kpms', '--keypoint-min-score', type = float, default = -1, 
    choices =  np.insert(np.arange(0, 1.1, 0.1).round(2), 0, -1), help = "Threshold to determine whether to keep a keypoint")
ap.add_argument('-kpr', '--keypoints-ratio', type = float, default = 1, 
    choices =  np.arange(0, 1.1, 0.1).round(2), help = "Not to draw the pose, the total percentage of the upper part in the total number of keypoints")
ap.add_argument('--threshold-denoise', type = float, default = 0.03, choices =  np.arange(0.01, 0.11, 0.01).round(2), 
    help = "Reduce background noise")
ap.add_argument('--flip', action = "store_true", default = False, help = 'Flip the screen if it\'s inverse')
ap.add_argument('--sec', type = float, default = 5.0, help = 'How many second to change a new pose')
ap.add_argument('--n-poses', type = int, default = 10, help = 'How many poses you wanna play with')
args = vars(ap.parse_args())


posenet = LoadModel(weight_dir = args['weight_dir'], model_id = args['model_id'], output_stride = args['output_stride'], 
    pose_model_name = "PoseNet", useGPU = args['useGPU'], verbose = args['verbose'])

if __name__ == "__main__":
    game_mode = Start_Game(args = args, posenet = posenet)
    if game_mode == "normal":
        Normal_Mode(args = args, posenet = posenet)
    if game_mode == "battle":
        Battle_Mode(args = args, posenet = posenet)
#find the beat