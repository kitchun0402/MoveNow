import cv2
import time
from pose_models import LoadModel, PredictPose
from annotation import annotation
import requests
import numpy as np
from utils import overlay_transparent
from game_tools import find_box
import os
import random
from pygame import mixer
import math
from game_tools import poser_selection, sound_effect

def Start_Game(args, posenet):
    mixer.init()
    mixer.music.load('./intro_music/LAKEY INSPIRED - Chill Day.mp3')
    mixer.music.set_volume(1)
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
        
        if args['flip']:
            cv2_img = cv2.flip(cv2_img, 1)

        # cv2_img = start_game_button(cv2_img)
        if pose_data['poses']:
            print('x', int(pose_data['poses'][0]['l_wrist']['x']))
            print('y', int(pose_data['poses'][0]['l_wrist']['y'] * 0.8))
            print('x', int(cv2_img.shape[1] * 0.51))
            print('x2', int(cv2_img.shape[1] * 0.49))
            print('y', int(cv2_img.shape[0] * 0.1))

            #touch option bar
            if (int(pose_data['poses'][0]['l_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.43)) and \
                    int(pose_data['poses'][0]['l_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.5296875)) and \
                    int(pose_data['poses'][0]['l_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.15083)) or \
                (int(pose_data['poses'][0]['r_wrist']['x']) < int(cv2_img.shape[1] * (1 - 0.43)) and \
                    int(pose_data['poses'][0]['r_wrist']['x']) > int(cv2_img.shape[1] * (1 - 0.5296875)) and \
                    int(pose_data['poses'][0]['r_wrist']['y'] * 0.5) < int(cv2_img.shape[0] * 0.15083)):
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
                        sound_effect('./sound_effect/click_button.wav')
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
                        sound_effect('./sound_effect/click_button.wav')
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
    tlx = int(cv2_img.shape[1] * 0.43)
    tly = int(cv2_img.shape[0] * 0.02)
    brx = int(cv2_img.shape[1] * 0.5296875)
    bry = int(cv2_img.shape[0] * 0.15083)
    # tlx = int(cv2_img.shape[1] * 0.46953125)
    # tly = int(cv2_img.shape[0] * 0.0375)
    # brx = int(cv2_img.shape[1] * 0.5296875)
    # bry = int(cv2_img.shape[0] * 0.13333333333333333)
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

def instruction(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    # img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/instruction.png", 0.00625, 0.018055555555555554, 0.33359375, 0.25277777777777777)
    img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/instruction.png", 0.00859375, 0.018055555555555554, 0.4, 0.22777777777777777)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img