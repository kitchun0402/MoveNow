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
import os
import random
import json
from pygame import mixer
import math

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


def MoveNow(args = args, posenet = posenet):
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
    while True:
        starttime = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')
        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        poser = poser_selection(pose_data)
        # for poser in pose_data['poses']:
        if poser:
            cv2_img = annotation (cv2_img, poser,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
        normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400)    
        
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1) #flip the frame

        
        # if not time_to_change_pose: #initial time of changing a pose
        #     time_to_change_pose = fps
            #fps < 3 should be * 3
        # if timer == 0: 
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

        cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
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
    
def start_game():
    mixer.init()
    mixer.music.load('./intro_music/LAKEY INSPIRED - Chill Day.mp3')
    mixer.music.set_volume(0.5)
    mixer.music.play(-1)
    if args['ip_webcam']:
        assert args['ip_address'] != None, "Please input your IP address shown on IP Webcam application"
        url = args['ip_address'].rstrip('/') + '/shot.jpg'
        assert url.startswith("http://"), "IP address should start with http://"
    else:
        capture = cv2.VideoCapture(0)
    
    tlx_new = 0
    brx_new = 0
    # timer = 0
    # fps = 0
    counter = 0
    touch = False
    normal_timer = 0
    normal_clicked = False
    battle_timer = 0
    battle_clicked = False
    while True:
        start_time = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')

        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        poser = poser_selection(pose_data)
        if poser:
            cv2_img = annotation (cv2_img, poser,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
            normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400) 
        
        # if timer == 0:
        #     fps = 1 / (time.time() - start_time)
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1)

        # cv2_img = start_game_button(cv2_img)
        if pose_data['poses']:
        #     # x-axis reversed due to the frame being flipped
        #     if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * 0.495) and \
        #         int(pose_data['poses'][0]['l_wrist']['y'] * 0.8) < int(cv2_img.shape[0] * 0.125)) or \
        #         (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * 0.495) and \
        #         int(pose_data['poses'][0]['r_wrist']['y'] * 0.8) < int(cv2_img.shape[0] * 0.125)):
        #         mixer.music.fadeout(5000)
        #         break
            # l_palm_x, l_palm_y = find_palm_xy(pose_data['poses'][0]['l_elbow']['x'], pose_data['poses'][0]['l_elbow']['y'], \
            #     pose_data['poses'][0]['l_wrist']['x'], pose_data['poses'][0]['l_wrist']['y'], 3, 7)
            # print(l_palm_x, l_palm_y)
            print('x', int(pose_data['poses'][0]['l_wrist']['x']))
            print('y', int(pose_data['poses'][0]['l_wrist']['y'] * 0.8))
            print('x', int(cv2_img.shape[1] * 0.51))
            print('x2', int(cv2_img.shape[1] * 0.49))
            print('y', int(cv2_img.shape[0] * 0.1))

            #touch option bar
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.46953125)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.5296875)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.13333333333333333)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.46953125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.5296875)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.13333333333333333)):
                    touch = True
            
            #touch normal button
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.53828125)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.61796875)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.1125)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.53828125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.61796875)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.1125)) and \
                        counter >= 10: #counter >= no. of frame, then the user can touch the button
                    normal_clicked = True
                    if normal_timer == 0:
                        normal_timer = time.time()
                    if time.time() - normal_timer >= 1:
                    # if normal_timer % math.ceil(fps * 2) == 0 and normal_timer != 0: #hold 2 sec to get into normal mode
                        mixer.music.fadeout(5000)
                        game_mode = "normal"
                        return game_mode
                    # normal_timer += 1
            elif not battle_clicked:
                normal_clicked = False
                normal_timer = 0 #ensure the user hold for 1 sec

            #touch battle button
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.63203125)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.71015625)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.1111111111111111)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.63203125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.71015625)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.1111111111111111)) and \
                        counter >= 18: #counter >= no. of frame, then the user can touch the button
                    battle_clicked = True
                    if battle_timer == 0:
                        battle_timer = time.time()
                    if time.time() - battle_timer >= 1:
                    # if battle_timer % math.ceil(fps * 2) == 0 and battle_timer != 0: #hold 2 sec to get into normal mode
                        mixer.music.fadeout(5000)
                        game_mode = "battle"
                        return game_mode
                    # battle_timer += 1
                    
            elif not normal_clicked:
                battle_clicked = False
                battle_timer = 0 #ensure the user hold for 1 sec
            
        if counter == 0:
            # brx_new = int(cv2_img.shape[1] * 0.51)
            # tlx_new = int(cv2_img.shape[1] * 0.49)
            brx_new = int(cv2_img.shape[1] * 0.54)
            tlx_new = int(cv2_img.shape[1] * 0.52)
        
        cv2_img = instruction(cv2_img) #show instruction
        if not touch:
            cv2_img = initial_bar(cv2_img)
            
            # cv2.rectangle(cv2_img, (int(cv2_img.shape[1] * 0.71), int(cv2_img.shape[0] * 0.42)), (int(cv2_img.shape[1] * 0.73) + 200, int(cv2_img.shape[0] * 0.54)), (255, 229, 204), -1)
        else:
            try:
                if counter < 21: #length of the bar
                # if timer % math.ceil(fps * 0.5) == 0 and counter < 21:
                    cv2_img = expanded_bar(cv2_img, tlx_new, brx_new)
                    tlx_new = brx_new
                    brx_new += round(cv2_img.shape[1] * 0.01)
                    counter += 1
                    # timer = 0
                else:
                    cv2_img = expanded_bar(cv2_img, tlx_new, brx_new)
                if counter >= 10:
                    cv2_img = normal_button(cv2_img, click = normal_clicked) #show normal button
                    normal_clicked = False
                if counter >= 18:
                    cv2_img = battle_button(cv2_img, click = battle_clicked) #show battle button
                    battle_clicked = False
                        
            except:
                pass
        
        # cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('MoveNow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("MoveNow", cv2_img)
        cv2.moveWindow("MoveNow", 0, 0)

        # if counter < 21:
        #     timer += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('fps: %.2f'%(1 / (time.time() - start_time)))

def start_game_button(cv2_img):
    alpha = 0.7
    overlay = cv2_img.copy()
    cv2.rectangle(overlay, (int(overlay.shape[1] * 0.495), int(overlay.shape[0] * 0.05)), (int(overlay.shape[1] * 0.635), int(overlay.shape[0] * 0.125)), (224, 224, 224), -1)       
    cv2.putText(overlay, "Move Now", (int(overlay.shape[1] / 2), int(overlay.shape[0] * 0.1)), cv2.FONT_HERSHEY_COMPLEX, 1, (102, 102, 255), 3, cv2.LINE_AA)
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img

def initial_bar(cv2_img):
    # tlx = int(cv2_img.shape[1] * 0.49)
    # tly = int(cv2_img.shape[0] * 0.05)
    # brx = int(cv2_img.shape[1] * 0.51)
    # bry = int(cv2_img.shape[0] * 0.12)
    # cv2.rectangle(cv2_img, (tlx, tly), (brx, bry), (255, 178, 102), -1)
    logo = cv2.imread('./UI_images/button_movenow.png', cv2.IMREAD_UNCHANGED)
    tlx = int(cv2_img.shape[1] * 0.46953125)
    tly = int(cv2_img.shape[0] * 0.0375)
    brx = int(cv2_img.shape[1] * 0.5296875)
    bry = int(cv2_img.shape[0] * 0.13333333333333333)
    cv2_img = overlay_transparent(cv2_img, logo, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img

def expanded_bar(cv2_img, tlx_new, brx_new):
    alpha = 0.6
    overlay = cv2_img.copy()
    cv2.rectangle(overlay, (int(overlay.shape[1] * 0.52), int(overlay.shape[0] * 0.05)), \
        (brx_new, int(overlay.shape[0] * 0.12)), (224, 224, 224), -1)
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    tlx_new = brx_new #for end_bar, update top left x 
    brx_new += round(cv2_img.shape[1] * 0.01) #for end_bar, update bottom right x
    cv2_img = end_bar(cv2_img, tlx_new, brx_new)
    cv2_img = initial_bar(cv2_img)
    return cv2_img

def end_bar(cv2_img, tlx_new, brx_new):
    cv2.rectangle(cv2_img, (tlx_new, int(cv2_img.shape[0] * 0.05)), \
        (brx_new, int(cv2_img.shape[0] * 0.12)), (204, 229, 255), -1)
    return cv2_img

def normal_button(cv2_img, click):
    if click:
        normal = cv2.imread('./UI_images/button_normal_click.png')
    else:
        normal = cv2.imread('./UI_images/button_normal.png')
  
    normal_brx = int(cv2_img.shape[1] * 0.61796875)
    normal_tlx = int(cv2_img.shape[1] * 0.53828125)
    normal_bry = int(cv2_img.shape[0] * 0.1125)
    normal_tly = int(cv2_img.shape[0] * 0.058333333333333334)
    normal = cv2.resize(normal.copy(), (normal_brx - normal_tlx, normal_bry - normal_tly))
    cv2_img[normal_tly:normal_bry, normal_tlx:normal_brx] = normal
    return cv2_img

def battle_button(cv2_img, click):
    if click:
        battle = cv2.imread('./UI_images/button_battle_click.png')
    else:
        battle = cv2.imread('./UI_images/button_battle.png')
    battle_brx = int(cv2_img.shape[1] * 0.71015625)
    battle_tlx = int(cv2_img.shape[1] * 0.63203125)
    battle_bry = int(cv2_img.shape[0] * 0.1111111111111111)
    battle_tly = int(cv2_img.shape[0] * 0.058333333333333334)
    battle = cv2.resize(battle.copy(), (battle_brx - battle_tlx, battle_bry - battle_tly))
    cv2_img[battle_tly:battle_bry, battle_tlx:battle_brx] = battle
    return cv2_img

def gamebox(img, target_poses, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = False, flip = False):
    height, width = img.shape[0:2]
    overlay = img.copy()
    if not battle_mode:
        tlx = int(width * 0.7)
        tly = int(height * 0.5)
        brx = width
        bry = height
        
        if gen_pose == False and prev_posedata: #not to generate pose
            posedata = prev_posedata
        else:
            posedata = pose_generator(target_poses) 
            while posedata == prev_posedata: #make sure the next pose is not the same as previous one
                posedata = pose_generator(target_poses)
    
    else:
        posedata = target_poses
        if gen_pose_left:
            tlx = int(width * 0.30)
            tly = int(height * 0.6)
            brx = int(width * 0.496875)
            bry = int(height * 0.9972222222222222)
        else:
            tlx = int(width * 0.80)
            tly = int(height * 0.6)
            brx = width
            bry = height

    
    pt1, pt2, new_posedata = zoomin_point(posedata, scale_x = 0.2, scale_y = 0.2) #crop the pose
    
    target_h = bry - tly 
    target_w = brx - tlx
    pose_h = pt2[1] - pt1[1]
    pose_w = pt2[0] - pt1[0]
    while pose_h > target_h or pose_w > target_w: #resize the pose within the target frame
        pose_h *= 0.9
        pose_w *= 0.9
    
    new_posedata = resize_point(new_posedata, pose_w, pose_h, pt1=pt1, pt2=pt2, cv2_img=None) #resize
    new_posedata = centralized_keypoint(target_w, target_h, new_posedata) #centalized
    cv2.rectangle(overlay, (tlx, tly), (brx, bry), (32,32,32), -1) #rectangle layer
    pose_img = overlay[tly: bry, tlx:brx] #capture the right corner of the origin frame
    pose_img = annotation(pose_img, new_posedata, keypoint_min_score = -1, keypoints_ratio = 1, threshold_denoise = 0.03, 
            normalized = False, pose_id = '?', scale_x = 0.3, scale_y = 0.3)
    if battle_mode and flip:
        pose_img = cv2.flip(pose_img, 1)
    # pose_img = cv2.resize(pose_img, (brx-tlx, bry - tly))
    
    overlay[tly: bry, tlx:brx] = pose_img #overwrite the origin
    # cv2.putText(overlay, "1", (tlx + round((brx - tlx)/2), tly + round((bry - tly)/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    new_img = cv2.addWeighted(overlay, 0.7, img, 0.2, 0)
    return new_img, posedata



def pose_generator(target_poses):
    # posedata = {"poses": {"nose": {"x": 535.098172745078, "y": 80.71212525175943, "conf": 1}, 
    # "l_eye": {"x": 546.455151619249, "y": 68.4322736531124, "conf": 1}, 
    # "r_eye": {"x": 521.9774041793553, "y": 68.32097089241017, "conf": 1}, 
    # "l_ear": {"x": 562.568912203914, "y": 80.64704511433467, "conf": 1}, 
    # "r_ear": {"x": 505.70463708484715, "y": 81.78348771411403, "conf": 1}, 
    # "l_shoulder": {"x": 588.779500372923, "y": 154.59615995748365, "conf": 1}, 
    # "r_shoulder": {"x": 485.74738074993263, "y": 157.68543523822842, "conf": 1}, 
    # "l_elbow": {"x": 614.0321815824908, "y": 234.88536014636247, "conf": 1}, 
    # "r_elbow": {"x": 469.8867598228774, "y": 241.42418198711167, "conf": 1}, 
    # "l_wrist": {"x": 618.9187685707852, "y": 320.6335292609819, "conf": 1}, 
    # "r_wrist": {"x": 460.9071562341297, "y": 319.50833440984337, "conf": 1}, 
    # "l_hip": {"x": 574.7652525781807, "y": 315.6835846497513, "conf": 1}, 
    # "r_hip": {"x": 508.9622923112426, "y": 317.7696791766586, "conf": 1}, 
    # "l_knee": {"x": 572.3168389994967, "y": 466.9179815326749, "conf": 1}, 
    # "r_knee": {"x": 509.98556769905196, "y": 469.52238699268867, "conf": 1}, 
    # "l_ankle": {"x": 576.4955891789616, "y": 590.6729397132228, "conf": 1},
    # "r_ankle": {"x": 512.8174241591255, "y": 594.3890109174625, "conf": 1}, 
    # "neck": {"x": 537, "y": 156, "conf": 0.8841903507709503}}, "compute_time": "2.305", 
    # "metadata": {"width": 1080, "height": 720, "pose_model_name": "PoseNet", "compute_time": "2.305"}}
    # target = random.shuffle(target_poses)
    pose_path = random.choice(target_poses)
    with open (pose_path, 'r') as pose:
        posedata = json.load(pose)
    return posedata['poses'][0]

def criteria (mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = False):
    #Perfect match (0.9987696908139251, 0.00658765889408432), (0.9975094474004261, 0.00842546430265792)
    #Poor match (0.9662763866452719, 0.028538520119295172), (0.9703551168423131, 0.029034741415598055)
    tlx_pct = 0.5734375
    tly_pct = 0.09722222222222222
    brx_pct = 0.82265625
    bry_pct = 0.30694444444444446
    if battle_mode_left_player: #box on the left
        tlx_pct = 0.340625
        tly_pct = 0.22361111111111112
        brx_pct = 0.4859375
        bry_pct = 0.34444444444444444
    if battle_mode_right_player: #box on the left
        tlx_pct = 0.8609375
        tly_pct = 0.22361111111111112
        brx_pct = 0.99140625
        bry_pct = 0.34444444444444444
    
    if mae == -1:
        text = "Missing"
        # cv2.putText(cv2_img, "Missing", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (96, 96, 96), 4, cv2.LINE_AA)
        img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/missing.png', tlx_pct, tly_pct, brx_pct, bry_pct)
        cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    elif mae < 0.009:
        text = "Perfect" #1000 score (2 combos, 500, 1000, 1500...)
        # cv2.putText(cv2_img, "Perfect", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (102, 102, 255), 4, cv2.LINE_AA)
        img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/perfect.png', tlx_pct, tly_pct, brx_pct, bry_pct)
        cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    elif mae < 0.020:
        text = "Good" #500 score (2 combos, 250, 500, 750...)
        # cv2.putText(cv2_img, "Good", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (102, 178, 255), 4, cv2.LINE_AA)
        img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/good.png', tlx_pct, tly_pct, brx_pct, bry_pct)
        cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    else:
        text = "Poor" #0 score
        # cv2.putText(cv2_img, "Poor", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (0, 128, 255), 4, cv2.LINE_AA)
        img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/poor.png', tlx_pct, tly_pct, brx_pct, bry_pct)
        cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return text, cv2_img

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
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Result", result_img)
    cv2.moveWindow('Result', 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sound_effect(sound_path):
    mixer.init()
    effect = mixer.Sound(sound_path)
    effect.set_volume(0.8)
    effect.play()
    time.sleep(0.6)
 
def find_box(cv2_img, img_path, tlx_pct, tly_pct, brx_pct, bry_pct):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    tlx = int(cv2_img.shape[1] * tlx_pct)
    tly = int(cv2_img.shape[0] * tly_pct)
    brx = int(cv2_img.shape[1] * brx_pct)
    bry = int(cv2_img.shape[0] * bry_pct)
    return img, tlx, tly, brx, bry

def instruction(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    # img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/instruction.png", 0.00625, 0.018055555555555554, 0.33359375, 0.25277777777777777)
    img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/instruction.png", 0.00859375, 0.018055555555555554, 0.3, 0.22777777777777777)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1 )
    return cv2_img

def poser_selection(pose_data):
    posedata_with_max_area = None
    max_area = 0
    for i, posedata in enumerate(pose_data['poses']):
        keypoints = posedata.keys()
        keypoint_list_x = [round(posedata[keypoint]['x']) for keypoint in keypoints]
        keypoint_list_y = [round(posedata[keypoint]['y']) for keypoint in keypoints]
        max_x_boundary, min_x_boundary, max_y_boundary, min_y_boundary = bounding_box_coordinates(keypoint_list_x, keypoint_list_y)
        area = (max_x_boundary - min_x_boundary) * (max_y_boundary - min_y_boundary)
        if i == 0:
            max_area = area
            posedata_with_max_area = posedata
        elif area > max_area:
            max_area = area
            posedata_with_max_area = posedata
    return posedata_with_max_area

def left_right(pose_data, cv2_img, flip):
    l_pose_data = []
    r_pose_data = []
    center_boundary = int(cv2_img.shape[1] * 0.5)
    for pose in pose_data['poses']:
        if flip:
            if pose['neck']['x'] > center_boundary:
                l_pose_data.append(pose)
            else:
                r_pose_data.append(pose)
        else:
            if pose['neck']['x'] < center_boundary:
                l_pose_data.append(pose)
            else:
                r_pose_data.append(pose)
    return {'poses': l_pose_data}, {'poses': r_pose_data}
def countdown(cv2_img, num):
    img, tlx, tly, brx, bry = find_box(cv2_img, f'./UI_images/countdown_{str(num)}.png', 0.48515625, 0, 0.51328125, 0.09722222222222222)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img, str(num)
def countdown_pose(cv2_img):
    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/countdown_pose.png', 0.46640625, 0, 0.53359375, 0.09722222222222222)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img, "pose"
def countdown_movenow(cv2_img):
    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/countdown_movenow.png', 0.43828125, 0, 0.56171875, 0.09722222222222222)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img, "movenow"
def countdown_whole(cv2_img, time_to_change_pose, count_down, timer, \
    l_display_pose, r_display_pose, l_player_act, l_player_follow, l_posedata_with_max_area, r_posedata_with_max_area, follow_mode):
    if timer % math.ceil(time_to_change_pose * 1) == 0 and count_down >= -1 and timer != 0:
        if count_down == 0:
            cv2_img, status = countdown_movenow(cv2_img)
            # cv2_img, status = countdown_pose(cv2_img)
        else:
            cv2_img, status = countdown(cv2_img, count_down)
        count_down -= 1
   
    if count_down == 3: #initial
        cv2_img, status = countdown_movenow(cv2_img) #movenow
    elif count_down + 1 != 0:
        cv2_img, status = countdown(cv2_img, count_down + 1) #countdown 3, 2, 1
        if follow_mode:
            print('here')
            cv2.putText(cv2_img, "hello", (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32,32,32), 3)
        else:
            if l_display_pose and l_player_act:
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            
            if r_display_pose and not l_player_act:
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])

            # if l_player_follow and count_down == 0:
            #     print('Evaluate')
            #     l_player_act = True
            #     l_display_pose = None
    elif count_down == 0:
        if follow_mode:
            pass
        else:
            if l_display_pose and l_player_act:
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            
            if r_display_pose and not l_player_act:
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
            cv2_img, status = countdown_movenow(cv2_img)
        # cv2_img, status = countdown_pose(cv2_img) #pose
    else:
        if follow_mode:
            if l_player_act:
                l_display_pose = None
                l_player_act = False
                follow_mode = False
            else:
                r_display_pose = None
                l_player_act = True
                follow_mode = False
        if l_display_pose or r_display_pose:
            follow_mode = True
        if not follow_mode:
            if l_player_act:
                    l_display_pose = l_posedata_with_max_area
                    l_player_follow = False
                    # follow_mode = True
            if not l_player_act:
                    r_display_pose = r_posedata_with_max_area
                    l_player_follow = True
                    # follow_mode = True

        cv2_img, status = countdown_movenow(cv2_img)
        count_down = 3
    
    return cv2_img, count_down, timer, status, l_display_pose, r_display_pose, l_player_act, l_player_follow, l_posedata_with_max_area, r_posedata_with_max_area, follow_mode

def battle(args = args, posenet = posenet):
    mixer.init()
    # playlist = [music.path for music in os.scandir('./background_music') if music.path.endswith('.mp3')]
    # background_music = random.choice(playlist)
    mixer.music.load("./background_music/2.mp3")
    mixer.music.set_volume(0.5)
    mixer.music.play(-1)
    if args['ip_webcam']:
        assert args['ip_address'] != None, "Please input your IP address shown on IP Webcam application"
        url = args['ip_address'].rstrip('/') + '/shot.jpg'
        assert url.startswith("http://"), "IP address should start with http://"
    else:
        capture = cv2.VideoCapture(0)
    
    target_poses = [pose.path for pose in os.scandir('./players/target_posedata/json') if pose.path.endswith('.json')]
    # print(random.choice(target_poses))
    
    prev_posedata = None 
    timer = 0
    scores = {'similarity':[], 'mae': []}
    result_img = None
    mae = 0
    textlist = []
    time_to_change_pose = 0
    skip_first_frame = True
    start_point = False
    count_down = 3 #for countdown
    l_player_act = True
    l_player_follow = True
    follow_mode = False
    l_display_pose = None
    r_display_pose = None
    while True:
        starttime = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')
        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        
        l_pose_data, r_pose_data = left_right(pose_data, cv2_img, args['flip'])

        cv2.line(cv2_img, (int(cv2_img.shape[1] * 0.5), int(cv2_img.shape[0] * 0.1)), (int(cv2_img.shape[1] * 0.5), int(cv2_img.shape[0])), (153,204,255), 5)
        
        l_posedata_with_max_area = poser_selection(l_pose_data)
        r_posedata_with_max_area = poser_selection(r_pose_data)

        if l_posedata_with_max_area: 
            cv2_img = annotation (cv2_img, l_posedata_with_max_area,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
            normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400)
        if r_posedata_with_max_area: 
            cv2_img = annotation (cv2_img, r_posedata_with_max_area,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
            normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400) 
        
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1) #flip the frame
    

        # if timer <= 10 and not time_to_change_pose:
        #     print('Loading...\n')
        #     timer += 1
        #     continue
        if skip_first_frame:
            skip_first_frame = False
            cv2.imshow('MoveNow', cv2_img)
            cv2.moveWindow('MoveNow', 0, 0)
            continue
        if not time_to_change_pose: #initial time of changing a pose
            # time_to_change_pose = fps
            time_to_change_pose = time.time()
            start_point = True
            # timer = 0
            #fps < 3 should be * 3

    #     cv2_img, count_down, timer, status, l_display_pose, r_display_pose, l_player_act, \
    #         l_player_follow, l_posedata_with_max_area, r_posedata_with_max_area, follow_mode = countdown_whole(cv2_img, time_to_change_pose, count_down, timer, \
    # l_display_pose, r_display_pose, l_player_act, l_player_follow, l_posedata_with_max_area, r_posedata_with_max_area, follow_mode)

        # if time.time() - time_to_change_pose <= 1:            

        time_to_countdown = time.time() - time_to_change_pose
        print(time_to_countdown)
        if time_to_countdown > 3.2: #buffer
            time_to_change_pose = 0
            
            if l_display_pose: #left player acted and saved the pose
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                l_player_follow = False
            
            if r_display_pose: #the right player's pose shown on the left
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
                l_player_follow = True

            print(l_player_act)
            #saved the left player's pose, l_player_follow -> make sure there is no new pose generated in 'follow' mode
            if l_player_act and l_player_follow: 
                l_display_pose = l_posedata_with_max_area
                
            ##saved the right player's pose, not l_player_follow -> make sure there is no new pose generated in 'follow' mode
            if not l_player_act and not l_player_follow: 
                r_display_pose = r_posedata_with_max_area
                # l_player_act = True #change to the left player to act
                
        
        
        
        if  time_to_countdown > 2.4: #1
            cv2_img, status = countdown(cv2_img, num = 1)
            if l_display_pose:
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            if r_display_pose: #the right player's pose shown on the left
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])

        
        
        elif time_to_countdown > 1.6: #2
            cv2_img, status = countdown(cv2_img, num = 2)
            if l_display_pose:
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            if r_display_pose: #the right player's pose shown on the left
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
        
        
        elif time_to_countdown > 0.8: #3
            cv2_img, status = countdown(cv2_img, num = 3)
            if l_display_pose:
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            if r_display_pose: #the right player's pose shown on the left
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])



        elif time_to_countdown >= 0: #begin
            cv2_img, status = countdown_movenow(cv2_img) #load movenow

            #the right player follows the left player
            if not l_player_follow and time_to_countdown <= 0.4 and l_display_pose:
                try:
                    _, mae = Evaluate(r_posedata_with_max_area, l_display_pose)
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = True)
                    if text == 'Poor':
                        sound_effect("./sound_effect/Poor.wav")
                    elif text == 'Good':
                        sound_effect("./sound_effect/Good.wav")
                    elif text == 'Perfect':
                        sound_effect("./sound_effect/Perfect.wav")
                except:
                    mae = -1 
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = True) #show missing
                l_display_pose = None
                l_player_act = False #change to the right player to act
            
            if l_player_follow and time_to_countdown <= 0.4 and r_display_pose:
                try:
                    _, mae = Evaluate(l_posedata_with_max_area, r_display_pose)
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = True, battle_mode_right_player = False)
                    if text == 'Poor':
                        sound_effect("./sound_effect/Poor.wav")
                    elif text == 'Good':
                        sound_effect("./sound_effect/Good.wav")
                    elif text == 'Perfect':
                        sound_effect("./sound_effect/Perfect.wav")
                except:
                    mae = -1 
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = True, battle_mode_right_player = False) #show missing
                r_display_pose = None
                l_player_act = True #change to the right player to act

            cv2.putText(cv2_img, "hello", (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32,32,32), 3)

            if l_display_pose: #the left player's pose shown on the right
                cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
            if r_display_pose: #the right player's pose shown on the left
                cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])

        

        #--------
        # if time_diff >= 0.5 and count_down >= -1:
        #     if count_down == 0:
        #         cv2_img, status = countdown_movenow(cv2_img)
        #         # cv2_img, status = countdown_pose(cv2_img)
        #     else:
        #         cv2_img, status = countdown(cv2_img, count_down)
                
        #     count_down -= 1
        #     time_to_change_pose = 0

        # if count_down == 3: #initial
        #     cv2_img, status = countdown_movenow(cv2_img) #movenow
        # elif count_down + 1 != 0:
        #     cv2_img, status = countdown(cv2_img, count_down + 1) #countdown 3, 2, 1
        #     if follow_mode:
                
        #         pass
        #     else:
        #         if l_display_pose and l_player_act:
        #             cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                
        #         if r_display_pose and not l_player_act:
        #             cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])

                
        # # elif count_down == 0:
        # #     print('here')
        # #     if follow_mode:
        # #         pass
        # #     else:
        # #         if l_display_pose and l_player_act:
        # #             cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                
        # #         if r_display_pose and not l_player_act:
        # #             cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
        # #         cv2_img, status = countdown_movenow(cv2_img)
        #     # cv2_img, status = countdown_pose(cv2_img) #pose
        # else:
        #     if follow_mode:
        #         left = False
        #         right = False
        #         # print(count_down)
                
        #         try:
        #             if l_display_pose:
        #                 _, mae = Evaluate(r_posedata_with_max_area, l_display_pose)
        #                 right = True
        #             text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = left, battle_mode_right_player = right)
        #             if text == 'Poor':
        #                 sound_effect("./sound_effect/Poor.wav")
        #             elif text == 'Good':
        #                 sound_effect("./sound_effect/Good.wav")
        #             elif text == 'Perfect':
        #                 sound_effect("./sound_effect/Perfect.wav")
        #         except:
        #             mae = -1 
        #             text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = left, battle_mode_right_player = right) #show missing
        #         if l_player_act:
        #             l_display_pose = None
        #             l_player_act = False
        #             follow_mode = False
        #         else:
        #             r_display_pose = None
        #             l_player_act = True
        #             follow_mode = False

        #     if l_display_pose or r_display_pose: #already store a pose
        #         follow_mode = True

        #     if not follow_mode:
                
        #         if l_player_act:
        #                 l_display_pose = l_posedata_with_max_area
        #                 l_player_follow = False
                        
        #         if not l_player_act:
        #                 r_display_pose = r_posedata_with_max_area
        #                 l_player_follow = True
                        
        #     cv2.putText(cv2_img, "hello", (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32,32,32), 3)
        #     cv2_img, status = countdown_movenow(cv2_img)
        #     count_down = 3
        #----------------
       
        #display pose

        # if l_display_pose and l_player_act and count_down >= 0:
        #     cv2_img, prev_posedata = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
        #     if count_down == 0:
        #         l_player_act = False
        #         l_display_pose = None
            # if not l_player_follow and status == 'change':
            #     cv2.putText(cv2_img, "hello", (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32,32,32), 3)
            #     l_player_act = False
            #     l_display_pose = None
        # elif timer < math.ceil(time_to_change_pose * args['sec'] * 0.3) and timer != 0:
        #     cv2.putText(cv2_img, "hello", (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32,32,32), 3)
        # if r_display_pose and not l_player_act and count_down >= 0:
           
        #     cv2_img, prev_posedata = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
        #     if count_down == 0:
        #         l_player_act = True
        #         r_display_pose = None
            # if l_player_follow and count_down == 0:
            #     print('Evaluate')
            #     l_player_act = True
            #     l_display_pose = None
        # timer += 1
    #     if timer == 0: #initial a pose
    #         cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True)
        # time_passed = time.time() - time_to_change_pose #time interval
        # if time_passed >= 2: #time to change a pose
        # # if timer % math.ceil(time_to_change_pose * args['sec']) == 0 and timer != 0: #control time to change another pose
        #     initial_pose = False #already initiate a pose
        #     try:
        #         #Perfect match (0.9987696908139251, 0.00658765889408432), (0.9975094474004261, 0.00842546430265792)
        #         #bad match (0.9662763866452719, 0.028538520119295172), (0.9703551168423131, 0.029034741415598055)
        #         similarity, mae = Evaluate(pose_data['poses'][0], prev_posedata)
        #         text, cv2_img = criteria(mae, cv2_img)
        #         if text == 'Poor':
        #             sound_effect("./sound_effect/Poor.wav")
        #         elif text == 'Good':
        #             sound_effect("./sound_effect/Good.wav")
        #         elif text == 'Perfect':
        #             sound_effect("./sound_effect/Perfect.wav")
        #         # scores['similarity'].append(similarity)
        #         scores['mae'].append(mae)
        #         textlist.append(text)
        #     except:
        #         # scores['similarity'].append(0)
        #         mae = -1 
        #         text, cv2_img = criteria(mae, cv2_img) #show missing
        #         scores['mae'].append(np.nan)
        #         textlist.append(text)
        #     cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True)
        #     # timer = 0 #reset timer
        #     time_to_change_pose = 0
        # else:
        #     cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata =  prev_posedata, gen_pose = False) #repeated the same result
        #     if time_passed <= 1 and not initial_pose: #in second, time to show the evaluation
        #     # if timer < math.ceil(time_to_change_pose * args['sec'] * 0.3) and timer != 0: #the time of showing the result (sec.)
        #         text, cv2_img = criteria(mae, cv2_img)
    #     # print(scores['mae'])
    #     if len(scores['mae']) == args['n_poses'] and timer >= math.ceil(time_to_change_pose * args['sec'] * 0.2): #control number of poses played
    #         if args['ip_webcam']:
    #             result_img = cv2.imdecode(capture, -1)
    #         else:
    #             status, result_img = capture.read()
    #         break
    #     timer += 1
        # cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('MoveNow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MoveNow', cv2_img)
        cv2.moveWindow('MoveNow', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = (1 / (time.time() - starttime))
        cv2.putText(cv2_img, 'FPS: %.1f'%(fps), (round(cv2_img.shape[1] * 0.01), round(cv2_img.shape[0] * 0.03)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32, 32, 32), 2)
        print('FPS: %.1f'%(fps))
    # print(textlist)
    # results = {'Perfect': 0, 'Good': 0, 'Poor': 0, 'Missing': 0}
    # for result in textlist:
    #     results[result] += 1
    # print(results)
    # avg_sim = round(np.mean(np.array(scores['similarity'])) * 100, 2)
    # if args['flip']:
    #     result_img = cv2.flip(result_img, 1)
    
    # mixer.music.fadeout(8000)
    # result_display(result_img, results)
if __name__ == "__main__":
    # game_mode = start_game()
    # if game_mode == "normal":
    #     MoveNow()
    # if game_mode == "battle":
    battle()
#find the beat