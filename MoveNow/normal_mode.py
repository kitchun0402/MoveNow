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
from game_tools import find_box, gamebox, criteria, sound_effect, poser_selection, pose_generator
import os
import random
import json
from pygame import mixer
import math
from datetime import datetime

target_poses = [pose.path for pose in os.scandir('./players/target_posedata/json') if pose.path.endswith('.json')]

def Normal_Mode(args, posenet, output_video = None):
    global target_poses
    mixer.init()
    playlist = [music.path for music in os.scandir('./background_music') if music.path.endswith('.mp3')]
    background_music = random.choice(playlist)
    mixer.music.load(background_music)
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
    
    # target_poses = [pose.path for pose in os.scandir('./players/target_posedata/json') if pose.path.endswith('.json')]
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
                cv2_img, prev_posedata, target_poses = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True, repeated_poses = args['repeated_poses'])
            if time_to_change_pose == 0: #initial time to changing a pose
                time_to_change_pose = time.time()
            time_passed = time.time() - time_to_change_pose #time interval
            if time_passed >= 3: #time to change a pose
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
                cv2_img, prev_posedata, target_poses = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True, repeated_poses = args['repeated_poses'])
                # timer = 0 #reset timer
                time_to_change_pose = 0
            else:
                cv2_img, prev_posedata, target_poses = gamebox(cv2_img, target_poses, prev_posedata =  prev_posedata, gen_pose = False, repeated_poses = args['repeated_poses']) #repeated the same result
                if time_passed <= 1 and not initial_pose: #in second, time to show the evaluation
                # if timer < math.ceil(time_to_change_pose * args['sec'] * 0.3) and timer != 0: #the time of showing the result (sec.)
                    text, cv2_img = criteria(mae, cv2_img)
            
            if len(scores['mae']) == args['n_poses'] and result_time == 0: #intiate a start time to give a buffer to show the evaluation
                result_time = time.time() 
        
            if len(scores['mae']) == args['n_poses'] and time.time() - result_time >= 1: #hold 1 sec
                # if args['ip_webcam']:
                #     result_img = cv2.imdecode(capture, -1)
                # else:
                #     status, result_img = capture.read()
                # result_img = cv2.resize(result_img, (1280, 720))
                break
            # timer += 1

        fps = (1 / (time.time() - starttime))
        cv2.putText(cv2_img, 'FPS: %.1f'%(fps), (round(cv2_img.shape[1] * 0.01), round(cv2_img.shape[0] * 0.03)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32, 32, 32), 2)
        print('FPS: %.1f'%(fps))
        # cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('MoveNow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MoveNow', cv2_img)
        cv2.moveWindow('MoveNow', 0, 0)
        # if output_video != None:
        #     output_video.write(cv2_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    # print(textlist)
    results = {'Perfect': 0, 'Good': 0, 'Poor': 0, 'Missing': 0}
    for result in textlist:
        results[result] += 1
    # print(results)
    total_poses = len(textlist)
    perfect_pct = results['Perfect'] / total_poses
    good_pct = results['Good'] / total_poses
    poor_pct = results['Poor'] / total_poses
    missing_pct = results['Missing'] / total_poses
    # if args['flip']:
    #     result_img = cv2.flip(result_img, 1) #flip the result page
    # mixer.music.fadeout(8000)
    # result_display(result_img, results) #old ver.
    """result"""
    time.sleep(1)
    capture_ = cv2.VideoCapture(0)
    start = 0
    result_count = 1
    result_img_ = None
    while True:
        status, frame = capture_.read()
        if start == 0:
            start = time.time()
        if args['flip']:
            frame = cv2.flip(frame, 1) #flip the result page
            frame = cv2.resize(frame, (1280, 720))
        if time.time() - start >= 0.1 and result_count <= 20: 
            try:
                cv2_img = display_result(frame, results, perfect_pct, good_pct, poor_pct, missing_pct, result_count)
            except:
                pass
            result_count += 1
            start = 0
            
        cv2.imshow('Result', cv2_img)
        cv2.moveWindow('Result', 0, 0)
        if output_video != None:
            output_video.write(cv2_img)

        if result_count > 20:
            result_img_ = cv2_img
            result_img_ = cv2.resize(result_img_, (1280, 720))
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture_.release()
    mixer.music.fadeout(8000)
    cv2.imshow("Result", result_img_)
    cv2.moveWindow('Result', 0, 0)
    # if args['imwrite']:
    #     savetime = str(datetime.now().time()).replace(":","")[0:6]
    #     cv2.imwrite(f"./result_images/{savetime}.png", result_img)

    # if output_video != None:
    #     output_video.write(cv2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "homepage"

# def result_display(cv2_img, results):
#     alpha = 0.7
#     overlay = cv2_img.copy()
#     cv2.rectangle(overlay, (int(overlay.shape[1] * 0.05), int(overlay.shape[0] * 0.1)), (int(overlay.shape[1] * 0.3), int(overlay.shape[0] * 0.65)), (224, 224, 224), -1)
#     cv2.putText(overlay, "Statistics", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(overlay, f"Perfect: {results['Perfect']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.3)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
#     cv2.putText(overlay, f"Good: {results['Good']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.4)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
#     cv2.putText(overlay, f"Poor: {results['Poor']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.5)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
#     cv2.putText(overlay, f"Missing: {results['Missing']}", (int(overlay.shape[1] * 0.1), int(overlay.shape[0] * 0.6)), cv2.FONT_HERSHEY_COMPLEX, 1, (96, 96, 96), 2, cv2.LINE_AA)
#     result_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
#     time.sleep(1)
#     # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
#     # cv2.setWindowProperty('Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("Result", result_img)
#     cv2.moveWindow('Result', 0, 0)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def instruction_normal(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/instruction_normal.png', 0.2805, 0.2914, 0.7219, 0.5792)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img

def display_result(cv2_img, results, perfect_pct, good_pct, poor_pct, missing_pct, times):
    alpha = 0.8
    overlay = cv2_img.copy()

    """labels"""
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/seperator.png', 0.0764, 0.1278, 0.1092, 0.6153)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/statistics.png', 0.1031, 0.0402, 0.2852, 0.1194)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/perfect.png', 0.0055, 0.1694, 0.0861, 0.2333)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/good.png', 0.0055, 0.2763, 0.0961, 0.3402) #0.0639
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/poor.png', 0.0055, 0.3832, 0.0961, 0.4471)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/missing.png', 0.0055, 0.4901, 0.0861, 0.554)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    
    """display results"""
    if perfect_pct != 0:
        img, tlx, tly, brx, bry = find_box(overlay, './UI_images/result_perfect.png', 0.0977, 0.1639, 0.2945, 0.2333)
        diff_perfect = tlx + int((brx - tlx) * perfect_pct / 20 * times)
        overlay = overlay_transparent(overlay, img, tlx, tly, (diff_perfect - tlx, bry - tly))
    if good_pct != 0:
        img, tlx, tly, brx, bry = find_box(overlay, './UI_images/result_good.png', 0.0977, 0.2708, 0.2945, 0.3402)
        diff_good = tlx + int((brx - tlx) * good_pct / 20 * times)
        overlay = overlay_transparent(overlay, img, tlx, tly, (diff_good - tlx, bry - tly))
    if poor_pct != 0:
        img, tlx, tly, brx, bry = find_box(overlay, './UI_images/result_poor.png', 0.0977, 0.3777, 0.2945, 0.4471)
        diff_poor = tlx + int((brx - tlx) * poor_pct / 20 * times)
        overlay = overlay_transparent(overlay, img, tlx, tly, (diff_poor - tlx, bry - tly))
    if missing_pct != 0:
        img, tlx, tly, brx, bry = find_box(overlay, './UI_images/result_missing.png', 0.0977, 0.4846, 0.2945, 0.554)
        diff_missing = tlx + int((brx - tlx) * missing_pct / 20 * times)
        overlay = overlay_transparent(overlay, img, tlx, tly, (diff_missing - tlx, bry - tly))

    result_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return result_img
