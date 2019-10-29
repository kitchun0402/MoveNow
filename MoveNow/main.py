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
    mixer.music.set_volume(0.5)
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
    timer = 0
    scores = {'similarity':[], 'mae': []}
    result_img = None
    mae = 0
    textlist = []
    time_to_change_pose = 0
    while True:
        starttime = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')
        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        for poser in pose_data['poses']:
            cv2_img = annotation (cv2_img, poser,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
            normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400)    
        
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1) #flip the frame

        fps = (1 / (time.time() - starttime))
        cv2.putText(cv2_img, 'FPS: %.1f'%(fps), (round(cv2_img.shape[1] * 0.01), round(cv2_img.shape[0] * 0.03)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32, 32, 32), 2)
        print('FPS: %.1f'%(fps))
        if not time_to_change_pose: #initial time of changing a pose
            time_to_change_pose = fps
            #fps < 3 should be * 3
        if timer == 0: #initial a pose
            cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata = None, gen_pose = True)
        if timer % int(time_to_change_pose * args['sec']) == 0 and timer != 0: #control time to change another pose
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
            timer = 0 #reset timer
        else:
            cv2_img, prev_posedata = gamebox(cv2_img, target_poses, prev_posedata =  prev_posedata, gen_pose = False) #repeated the same result
            if timer < int(time_to_change_pose * args['sec'] * 0.3) and timer != 0: #the time of showing the result (sec.)
                text, cv2_img = criteria(mae, cv2_img)
        # print(scores['mae'])
        if len(scores['mae']) == args['n_poses'] and timer >= int(time_to_change_pose * args['sec'] * 0.3): #control number of poses played
            if args['ip_webcam']:
                result_img = cv2.imdecode(capture, -1)
            else:
                status, result_img = capture.read()
            break
        timer += 1
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
        result_img = cv2.flip(result_img, 1)
    
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
    timer = 0
    fps = 0
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
        for poser in pose_data['poses']:
            cv2_img = annotation (cv2_img, poser,keypoint_min_score = args['keypoint_min_score'], keypoints_ratio = args['keypoints_ratio'], threshold_denoise = args['threshold_denoise'], 
            normalized = False, pose_id = args['pose_id'], resize = False, resize_W = 200, resize_H = 400) 
        if timer == 0:
            fps = 1 / (time.time() - start_time)
        
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
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.13333333333333333)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.46953125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.5296875)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.13333333333333333)):
                    touch = True
            
            #touch normal button
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.53828125)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.61796875)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.1125)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.53828125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.61796875)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.1125)) and \
                        counter >= 10: #counter >= no. of frame, then the user can touch the button
                    normal_clicked = True
                    if normal_timer % math.ceil(fps * 2) == 0 and normal_timer != 0: #hold 2 sec to get into normal mode
                        mixer.music.fadeout(5000)
                        game_mode = "normal"
                        return game_mode
                    normal_timer += 1
            elif not battle_clicked:
                normal_clicked = False
                normal_timer = 0 #ensure the user hold for 2 sec

            #touch battle button
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.63203125)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.71015625)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.1111111111111111)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.63203125)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.71015625)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.3) < int(cv2_img.shape[0] * 0.1111111111111111)) and \
                        counter >= 18: #counter >= no. of frame, then the user can touch the button
                    battle_clicked = True
                    
                    if battle_timer % math.ceil(fps * 2) == 0 and battle_timer != 0: #hold 2 sec to get into normal mode
                        mixer.music.fadeout(5000)
                        game_mode = "battle"
                        return game_mode
                    battle_timer += 1
                    
            elif not normal_clicked:
                battle_clicked = False
                battle_timer = 0 #ensure the user hold for 2 sec
            
        if counter == 0:
            # brx_new = int(cv2_img.shape[1] * 0.51)
            # tlx_new = int(cv2_img.shape[1] * 0.49)
            brx_new = int(cv2_img.shape[1] * 0.54)
            tlx_new = int(cv2_img.shape[1] * 0.52)
        if not touch:
            cv2_img = initial_bar(cv2_img)
            # cv2.rectangle(cv2_img, (int(cv2_img.shape[1] * 0.71), int(cv2_img.shape[0] * 0.42)), (int(cv2_img.shape[1] * 0.73) + 200, int(cv2_img.shape[0] * 0.54)), (255, 229, 204), -1)
        else:
            try:
                if timer % math.ceil(fps * 0.5) == 0 and counter < 21:
                    cv2_img = expanded_bar(cv2_img, tlx_new, brx_new)
                    tlx_new = brx_new
                    brx_new += round(cv2_img.shape[1] * 0.01)
                    counter += 1
                    timer = 0
                    
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
        
        cv2.namedWindow('MoveNow', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('MoveNow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("MoveNow", cv2_img)
        cv2.moveWindow("MoveNow", 0, 0)

        if counter < 21:
            timer += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

def gamebox(img, target_poses, prev_posedata = None,  gen_pose = False):
    height, width = img.shape[0:2]
    tlx = int(width * 0.7)
    tly = int(height * 0.5)
    brx = width
    bry = height
    overlay = img.copy()
    
    if gen_pose == False and prev_posedata: #not to generate pose
        posedata = prev_posedata
    else:
        posedata = pose_generator(target_poses) 
        while posedata == prev_posedata: #make sure the next pose is not the same as previous one
            posedata = pose_generator(target_poses)

    
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

def criteria (mae, cv2_img):
    #Perfect match (0.9987696908139251, 0.00658765889408432), (0.9975094474004261, 0.00842546430265792)
    #Poor match (0.9662763866452719, 0.028538520119295172), (0.9703551168423131, 0.029034741415598055)
    if mae == -1:
        text = "Missing"
        cv2.putText(cv2_img, "Missing", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (96, 96, 96), 4, cv2.LINE_AA)
    elif mae < 0.009:
        text = "Perfect" #1000 score (2 combos, 500, 1000, 1500...)
        cv2.putText(cv2_img, "Perfect", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (102, 102, 255), 4, cv2.LINE_AA)
    elif mae < 0.020:
        text = "Good" #500 score (2 combos, 250, 500, 750...)
        cv2.putText(cv2_img, "Good", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (102, 178, 255), 4, cv2.LINE_AA)
    else:
        text = "Poor" #0 score
        cv2.putText(cv2_img, "Poor", (int(cv2_img.shape[1] * 0.6), int(cv2_img.shape[0] * 0.2)), cv2.FONT_HERSHEY_PLAIN, 4, (0, 128, 255), 4, cv2.LINE_AA)
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
    effect.set_volume(0.4)
    effect.play()
    time.sleep(0.6)
 

if __name__ == "__main__":
    game_mode = start_game()
    # time.sleep(5)
    if game_mode == "normal":
        MoveNow()

#find the beat