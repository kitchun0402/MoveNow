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
from game_tools import find_box, gamebox, criteria, sound_effect, pose_generator, poser_selection
import os
import random
import json
from pygame import mixer
import math

def Normal_Mode(args, posenet):
    mixer.init()
    # playlist = [music.path for music in os.scandir('./background_music') if music.path.endswith('.mp3')]
    # background_music = random.choice(playlist)
    mixer.music.load("./background_music/2.mp3")
    mixer.music.set_volume(1)
    mixer.music.play(-1)

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
    if args['ip_webcam']:
        assert args['ip_address'] != None, "Please input your IP address shown on IP Webcam application"
        url = args['ip_address'].rstrip('/') + '/shot.jpg'
        assert url.startswith("http://"), "IP address should start with http://"
    else:
        capture = cv2.VideoCapture(0)
    
    target_poses = [pose.path for pose in os.scandir('./players/target_posedata/json') if pose.path.endswith('.json')]
    # print(random.choice(target_poses))
    
    prev_posedata = None 
    # timer = 0
    initial_pose = True
    scores = {'similarity':[], 'mae': []}
    result_img = None
    mae = 0
    textlist = []
    time_to_change_pose = 0
    result_time = 0
    instruction_time = None
    while True:
        starttime = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')
        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        poser = poser_selection(pose_data)
        # for poser in pose_data['poses']:
        """only select the poser with the larger area"""
        if poser:
            cv2_img = annotation (cv2_img, poser,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
        normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400)    
        
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1) #flip the frame

        """instructions"""
        if not instruction_time:
            instruction_time = time.time()
        if time.time() - instruction_time <= 5: #load 8 sec
            cv2_img = instruction_normal(cv2_img)
        else:
            if not prev_posedata: #initiate a pose
                cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True)
            if time_to_change_pose == 0: #initial time to changing a pose
                time_to_change_pose = time.time()
            time_passed = time.time() - time_to_change_pose #time interval
            if time_passed >= 2: #time to change a pose
            # if timer % math.ceil(time_to_change_pose * args['sec']) == 0 and timer != 0: #control time to change another pose
                initial_pose = False #already initiate a pose
                try:
                    #Perfect match (0.9987696908139251, 0.00658765889408432), (0.9975094474004261, 0.00842546430265792)
                    #bad match (0.9662763866452719, 0.028538520119295172), (0.9703551168423131, 0.029034741415598055)
                    similarity, mae = Evaluate(pose_data['poses'][0], prev_posedata)
                    text, cv2_img = criteria(mae, cv2_img)
                    if text == 'Poor':
                        sound_effect("./sound_effect/Poor.wav")
                    elif text == 'Good':
                        sound_effect("./sound_effect/Good.wav")
                    elif text == 'Perfect':
                        sound_effect("./sound_effect/Perfect.wav")
                    # scores['similarity'].append(similarity)
                    scores['mae'].append(mae)
                    textlist.append(text)
                except:
                    # scores['similarity'].append(0)
                    mae = -1 
                    text, cv2_img = criteria(mae, cv2_img) #show missing
                    sound_effect("./sound_effect/Missing.wav")
                    scores['mae'].append(np.nan)
                    textlist.append(text)
                cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True)
                # timer = 0 #reset timer
                time_to_change_pose = 0
            else:
                cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata =  prev_posedata, gen_pose = False) #repeated the same result
                if time_passed <= 1 and not initial_pose: #in second, time to show the evaluation
                # if timer < math.ceil(time_to_change_pose * args['sec'] * 0.3) and timer != 0: #the time of showing the result (sec.)
                    text, cv2_img = criteria(mae, cv2_img)
            
            if len(scores['mae']) == args['n_poses'] and result_time == 0: #intiate a start time to give a buffer to show the evaluation
                result_time = time.time() 
            # if len(scores['mae']) == args['n_poses'] and timer >= math.ceil(time_to_change_pose * args['sec'] * 0.2): #control number of poses played
            if len(scores['mae']) == args['n_poses'] and time.time() - result_time >= 1: #hold 1 sec
                if args['ip_webcam']:
                    result_img = cv2.imdecode(capture, -1)
                else:
                    status, result_img = capture.read()
                break
            # timer += 1

        fps = (1 / (time.time() - starttime))
        cv2.putText(cv2_img, 'FPS: %.1f'%(fps), (round(cv2_img.shape[1] * 0.01), round(cv2_img.shape[0] * 0.03)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32, 32, 32), 2)
        print('FPS: %.1f'%(fps))

        # cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('MoveNow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MoveNow', cv2_img)
        cv2.moveWindow('MoveNow', 0, 0)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(textlist)
    results = {'Perfect': 0, 'Good': 0, 'Poor': 0, 'Missing': 0}
    for result in textlist:
        results[result] += 1
    print(results)
    avg_sim = round(np.mean(np.array(scores['similarity'])) * 100, 2)
    if args['flip']:
        result_img = cv2.flip(result_img, 1) #flip the result page
    
    mixer.music.fadeout(8000)
    result_display(result_img, results)

def result_display(cv2_img, results):
    alpha = 0.7
    overlay = cv2_img.copy()
    cv2.rectangle(overlay, (int(overlay.shape[1] * 0.05), int(overlay.shape[0] * 0.1)), (int(overlay.shape[1] * 0.3), int(overlay.shape[0] * 0.65)), (224, 224, 224), -1)
    cv2.putText(overlay, "Statistics", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Perfect: {results['Perfect']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.3)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Good: {results['Good']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.4)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Poor: {results['Poor']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.5)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Missing: {results['Missing']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.6)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
    result_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    time.sleep(1)
    # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Result", result_img)
    cv2.moveWindow('Result', 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def instruction_normal(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/instruction_normal.png', 0.2805, 0.2914, 0.7219, 0.5792)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img