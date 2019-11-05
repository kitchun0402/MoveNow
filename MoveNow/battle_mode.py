import cv2
import time
from pose_models import LoadModel, PredictPose
from evaluate import Evaluate
from annotation import annotation
import requests
import numpy as np
from utils import bounding_box_coordinates, overlay_transparent
from game_tools import find_box, gamebox, criteria, sound_effect, poser_selection, combo
import os
import random
import json
from pygame import mixer
import math
from datetime import datetime

def Battle_Mode(args, posenet, output_video = None):
    mixer.init()
    playlist = [music.path for music in os.scandir('./background_music') if music.path.endswith('.mp3')]
    background_music = random.choice(playlist)
    mixer.music.load(background_music)
    mixer.music.set_volume(1)
    mixer.music.play(-1)

    if args['ip_webcam']:
        assert args['ip_address'] != None, "Please input your IP address shown on IP Webcam application"
        url = args['ip_address'].rstrip('/') + '/shot.jpg'
        assert url.startswith("http://"), "IP address should start with http://"
    else:
        capture = cv2.VideoCapture(0)
    
    overall_time = 0
    evaluation = False
    result_img = None
    mae = 0
    time_to_change_pose = 0
    skip_first_frame = True
    instruction_time = None
    l_player_act = random.choice([True, False])
    l_player_follow = l_player_act
    l_display_pose = None
    r_display_pose = None
    intense_music = True
    """Left heath bar"""
    l_tlx = None
    l_tly = None
    l_brx = None
    l_bry = None
    l_brx_o = None
    l_combo = []
    """Right heath bar"""
    r_tlx = None
    r_tly = None
    r_brx = None
    r_bry = None
    r_brx_o = None
    r_combo = []
    """Control the movement of the heath bar"""
    division = 10
    r_health_counter = 0
    r_average_hit = None
    l_health_counter = 0
    l_average_hit = None

    while True:
        starttime = time.time()
        if not overall_time:
            overall_time = time.time()
        if args['ip_webcam']:
            img_resp = requests.get(url)
            capture = np.array(bytearray(img_resp.content), dtype = 'uint8')
        pose_data, image_name, cv2_img = PredictPose(model = posenet, capture = capture, ip_webcam = args['ip_webcam'], scale_factor= args['scale_factor'], 
            output_stride= args['output_stride'], useGPU= args['useGPU'])
        
        """seperate posedata into two sets"""
        l_pose_data, r_pose_data = left_right(pose_data, cv2_img, args['flip'])
        
        """middle line"""
        cv2_img = seperator(cv2_img)
        # cv2.line(cv2_img, (int(cv2_img.shape[1] * 0.5), int(cv2_img.shape[0] * 0.1)), (int(cv2_img.shape[1] * 0.5), int(cv2_img.shape[0])), (153,204,255), 5)
        
        
        """save the front poser only"""
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
        

        """skip 1st frame to get better loading"""
        
        if skip_first_frame:
            skip_first_frame = False
            cv2.imshow('MoveNow', cv2_img)
            cv2.moveWindow('MoveNow', 0, 0)
            if output_video != None:
                output_video.write(cv2_img)
            continue
        """instructions"""
        if not instruction_time:
            instruction_time = time.time()
        if time.time() - instruction_time <= 6: #load 8 sec
            cv2_img = instruction_battle(cv2_img)
        else:
            if not time_to_change_pose: #initial time of changing a pose
                time_to_change_pose = time.time()

            """who to pose"""
            cv2_img = who_to_pose(cv2_img, left_to_pose = l_player_act)

            time_to_countdown = time.time() - time_to_change_pose
            print('Time: ', time_to_countdown)
            """Buffer"""
            if time_to_countdown > 3.2: #buffer
                time_to_change_pose = 0
                
                if l_display_pose: #left player acted and saved the pose
                    cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = False)
                    l_player_follow = False
                    evaluation = True
                
                if r_display_pose: #the right player's pose shown on the left
                    cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = True)
                    l_player_follow = True
                    evaluation = True

                #saved the left player's pose, l_player_follow -> make sure there is no new pose generated in 'follow' mode
                if l_player_act and l_player_follow: 
                    l_display_pose = l_posedata_with_max_area
                    
                #saved the right player's pose, not l_player_follow -> make sure there is no new pose generated in 'follow' mode
                if not l_player_act and not l_player_follow: 
                    r_display_pose = r_posedata_with_max_area
                
                    
            """Countdown: 1"""
            if  time_to_countdown > 2.4: #1
                cv2_img, l_tlx, l_tly, l_brx, l_bry, _ = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, intense_music = intense_music)
                cv2_img, r_tlx, r_tly, r_brx, r_bry, _ = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o = r_brx_o, intense_music = intense_music)
                cv2_img = countdown(cv2_img, num = 1)

                if l_display_pose:
                    cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = False)
                if r_display_pose: #the right player's pose shown on the left
                    cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = True)
                """Countdown: 2"""
            elif time_to_countdown > 1.6: #2
                cv2_img, l_tlx, l_tly, l_brx, l_bry, _ = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, intense_music = intense_music)
                cv2_img, r_tlx, r_tly, r_brx, r_bry, _ = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o= r_brx_o, intense_music = intense_music)
                cv2_img = countdown(cv2_img, num = 2)

                if l_display_pose:
                    cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = False)
                if r_display_pose: #the right player's pose shown on the left
                    cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])        
                    cv2_img = followme(cv2_img, shown_on_left = True)
                """Countdown: 3"""
            elif time_to_countdown > 0.8: #3
                cv2_img, l_tlx, l_tly, l_brx, l_bry, _ = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, intense_music = intense_music)
                cv2_img, r_tlx, r_tly, r_brx, r_bry, _ = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o= r_brx_o, intense_music = intense_music)
                cv2_img = countdown(cv2_img, num = 3)

                if l_display_pose:
                    cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = False)
                if r_display_pose: #the right player's pose shown on the left
                    cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = True)
                """sustain the effect"""
                if not l_player_follow and l_player_act:
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = True)
                    l_player_act = False #change to the right player to act
            
                if l_player_follow and not l_player_act:
                    text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = True, battle_mode_right_player = False)
                    l_player_act = True #change to the right player to act
                    
                """Countdown: MoveNow"""
            elif time_to_countdown >= 0: #begin
                cv2_img, l_tlx, l_tly, l_brx, l_bry, _ = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, intense_music = intense_music)
                cv2_img, r_tlx, r_tly, r_brx, r_bry, _ = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o= r_brx_o, intense_music = intense_music)
                cv2_img = countdown_movenow(cv2_img) #load movenow
                if not l_brx_o:
                    l_brx_o = l_brx
                if not r_brx_o:
                    r_brx_o = r_brx
                """The right player follows the left player"""
                if not l_player_follow and l_display_pose:
                    if evaluation:
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
                            sound_effect("./sound_effect/Missing.wav")
                        evaluation = False
                        
                        """combo"""
                        if r_combo == []: #empty list
                            r_combo.append(text)
                        elif text in r_combo:
                            r_combo.append(text)
                        else:
                            r_combo = []
                            r_combo.append(text)
                        
                    else:
                        text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = True) #show missing
                    
                    """hit"""
                    hit = hit_pct (result = text, result_list = r_combo)
                    r_average_hit = hit / division
                    print('r_average_hit', l_average_hit)

                    """reset the pose"""
                    l_display_pose = None
                    
                """The left player follows the right player"""
                if l_player_follow and r_display_pose:
                    if evaluation:
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
                            sound_effect("./sound_effect/Missing.wav")
                        evaluation = False
                        
                        """combo"""
                        if l_combo == []: #empty list
                            l_combo.append(text)
                        elif text in l_combo:
                            l_combo.append(text)
                        else:
                            l_combo = []
                            l_combo.append(text)

                    else:
                        text, cv2_img = criteria(mae, cv2_img, battle_mode_left_player = True, battle_mode_right_player = False) 
                
                    """hit"""
                    hit = hit_pct (result = text, result_list = l_combo)
                    l_average_hit = hit / division
                    # print('l_average_hit', l_average_hit)

                    """reset the pose"""
                    r_display_pose = None
                    
                """followme"""
                if l_display_pose: #the left player's pose shown on the right
                    cv2_img, _ = gamebox(cv2_img, l_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = False)
                if r_display_pose: #the right player's pose shown on the left
                    cv2_img, _ = gamebox(cv2_img, r_display_pose, prev_posedata = None,  gen_pose = False, gen_pose_left = True, battle_mode = True, flip = args['flip'])
                    cv2_img = followme(cv2_img, shown_on_left = True)
            """show combo"""
            if time_to_countdown <= 2.5 and len(r_combo) > 1 and r_average_hit:
                cv2_img = combo(cv2_img, left_player = False)

            elif time_to_countdown <= 2.5 and len(l_combo) > 1 and l_average_hit:
                cv2_img = combo(cv2_img, left_player = True)

            """hit"""
            #the right player hits the left player
            if time_to_countdown >= 0 and r_average_hit and r_health_counter < division:
                if text == "Poor" or text == "Missing":
                    cv2_img, r_tlx, r_tly, r_brx, r_bry, intense_music = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o= r_brx_o, \
                        hit = r_average_hit, intense_music = intense_music)
                else:
                    cv2_img, l_tlx, l_tly, l_brx, l_bry, intense_music = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, \
                        hit = r_average_hit, intense_music = intense_music) 
                
                if r_health_counter == division - 1:
                    r_health_counter = 0
                    r_average_hit = None
                r_health_counter += 1
                
            #the left player hits the right player
            if time_to_countdown >= 0 and l_average_hit and l_health_counter < division:
                if text == "Poor" or text == "Missing":
                    cv2_img, l_tlx, l_tly, l_brx, l_bry, intense_music = l_health_bar(cv2_img, l_tlx = l_tlx, l_tly = l_tly, l_brx = l_brx, l_bry = l_bry, l_brx_o = l_brx_o, \
                        hit = l_average_hit, intense_music = intense_music)
                else:
                    cv2_img, r_tlx, r_tly, r_brx, r_bry, intense_music = r_health_bar(cv2_img, r_tlx = r_tlx, r_tly = r_tly, r_brx = r_brx, r_bry = r_bry, r_brx_o= r_brx_o, \
                        hit = l_average_hit, intense_music = intense_music)
                
                if l_health_counter == division - 1:
                    l_health_counter = 0
                    l_average_hit = None
                l_health_counter += 1

            
            """Finish Game"""
            if l_brx <= l_tlx:
                cv2_img = winner_symbol(cv2_img, l_win = False)
                result_img = cv2_img
                break
            if r_brx <= r_tlx: 
                cv2_img = winner_symbol(cv2_img, l_win = True)
                result_img = cv2_img
                break

            print(time.time() - overall_time)
            if time.time() - overall_time >= 180: #3mins
                if (l_brx - l_tlx) > (r_brx - r_tlx):
                    l_win = True
                elif (l_brx - l_tlx) < (r_brx - r_tlx):
                    l_win = False
                else:
                    l_win = None
                if l_win != None:
                    cv2_img = winner_symbol(cv2_img, l_win = l_win)
                result_img = cv2_img
                break


        cv2.imshow('MoveNow', cv2_img)
        cv2.moveWindow('MoveNow', 0, 0)
        if output_video != None:
            output_video.write(cv2_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = (1 / (time.time() - starttime))
        cv2.putText(cv2_img, 'FPS: %.1f'%(fps), (round(cv2_img.shape[1] * 0.01), round(cv2_img.shape[0] * 0.03)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (32, 32, 32), 2)
        print('FPS: %.1f'%(fps))
 
    """Result"""
    mixer.music.fadeout(8000)
    cv2.imshow("Result", result_img)
    if args['imwrite']:
        savetime = str(datetime.now().time()).replace(":","")[0:6]
        cv2.imwrite(f"./result_images/{savetime}.png", result_img)
    cv2.moveWindow('Result', 0, 0)
    if output_video != None:
        output_video.write(result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "homepage"

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
    return cv2_img
def countdown_pose(cv2_img):
    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/countdown_pose.png', 0.46640625, 0, 0.53359375, 0.09722222222222222)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img
def countdown_movenow(cv2_img):
    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/countdown_movenow.png', 0.43828125, 0, 0.56171875, 0.09722222222222222)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img

def l_health_bar(cv2_img, l_tlx = None, l_tly = None, l_brx = None, l_bry = None, l_brx_o = None, hit = 0, intense_music = True):
    alpha = 0.8
    overlay = cv2_img.copy()

    """Background"""
    img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/health_bar_bg.png", 0.07265625, 0.015277777777777777, 0.375, 0.09444444444444444)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))

    """Actual Health"""
    if l_brx_o:
        if l_brx - l_tlx >= int((l_brx_o - l_tlx) * 0.3): #change the color of the health bar
            health_img = "./UI_images/health_bar_green.png"
        else:
            health_img = "./UI_images/health_bar.png"
            if intense_music:
                playlist = [music.path for music in os.scandir('./intense_music') if music.path.endswith('.mp3')]
                background_music = random.choice(playlist)
                mixer.music.stop()
                mixer.music.load(background_music)
                mixer.music.set_volume(1)
                mixer.music.play(-1)
                intense_music = False

    if not l_tlx: #inital
        img, l_tlx, l_tly, l_brx, l_bry = find_box(cv2_img, "./UI_images/health_bar_green.png", 0.07265625, 0.015277777777777777, 0.375, 0.09444444444444444)
        
    else:
        img = cv2.imread(health_img, cv2.IMREAD_UNCHANGED)
    if l_brx_o: #initiate an original point
        l_brx = int(l_brx - (l_brx_o - l_tlx) * hit)
    
    """only remain the background if l_brx <= l_tlx"""
    try:
        overlay = overlay_transparent(overlay, img, l_tlx, l_tly, (l_brx - l_tlx, l_bry - l_tly))
    except:
        pass
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img, l_tlx, l_tly, l_brx, l_bry, intense_music

def r_health_bar(cv2_img, r_tlx = None, r_tly = None, r_brx = None, r_bry = None, r_brx_o = None, hit = 0, intense_music = True):
    
    alpha = 0.8
    overlay = cv2_img.copy()
    """Background"""
    img, tlx, tly, brx, bry = find_box(cv2_img, "./UI_images/health_bar_bg.png", 0.60625, 0.015277777777777777, 0.90859375, 0.09444444444444444)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))

    """Actual Health"""
    if r_brx_o:
        if r_brx - r_tlx >= int((r_brx_o - r_tlx) * 0.3): #change the color of the health bar
            health_img = "./UI_images/health_bar_green.png"
        else:
            health_img = "./UI_images/health_bar.png" #change to red color
            if intense_music:
                playlist = [music.path for music in os.scandir('./intense_music') if music.path.endswith('.mp3')]
                background_music = random.choice(playlist)
                mixer.music.stop()
                mixer.music.load(background_music)
                mixer.music.set_volume(1)
                mixer.music.play(-1)
                intense_music = False

    if not r_tlx: #inital
        img, r_tlx, r_tly, r_brx, r_bry = find_box(cv2_img, "./UI_images/health_bar_green.png", 0.60625, 0.015277777777777777, 0.90859375, 0.09444444444444444)
    else:
        img = cv2.imread(health_img, cv2.IMREAD_UNCHANGED)
    if r_brx_o: #initiate an original point
        r_brx = int(r_brx - (r_brx_o - r_tlx) * hit)

    """only remain the background if rbrx <= r_tlx"""
    try:
        overlay = overlay_transparent(overlay, img, r_tlx, r_tly, (r_brx  - r_tlx, r_bry - r_tly))
    except:
        pass
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img, r_tlx, r_tly, r_brx, r_bry, intense_music

def hit_pct(result, result_list):
    multiplier = 1
    print(result_list)
    if result_list != []:
        multiplier = len(result_list)
    if result == "Poor" or result == "Good":
        hit = 0.10 * multiplier
        return hit
    elif result == "Perfect":
        hit = 0.15 * multiplier
        return hit
    elif result == "Missing":
        hit = 0.4 * multiplier
        return hit
def winner_symbol(cv2_img, l_win):
    if l_win:
        tlx_pct = 0.35
        tly_pct = 0.14
        brx_pct = 0.49
        bry_pct = 0.33
    else:
        tlx_pct = 0.85
        tly_pct = 0.14
        brx_pct = 0.99
        bry_pct = 0.33

    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/winner.png', tlx_pct, tly_pct, brx_pct, bry_pct)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img

def seperator(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/seperator.png', 0.4875, 0.09583333333333334, 0.51015625, 0.9958333333333333)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img

def followme(cv2_img, shown_on_left):
    if shown_on_left:
        tlx_pct = 0.30
        tly_pct = 0.50
        brx_pct = 0.50
        bry_pct = 0.59
    else:
        tlx_pct = 0.80
        tly_pct = 0.50
        brx_pct = 1.00
        bry_pct = 0.59

    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/followme.png', tlx_pct, tly_pct, brx_pct, bry_pct)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img

def who_to_pose(cv2_img, left_to_pose):
    tlx_pct = 0.45
    tly_pct = 0.37
    brx_pct = 0.55
    bry_pct = 0.48
    if left_to_pose:
        int_img = './UI_images/instruction_l_pose.png'
    else:
        int_img = './UI_images/instruction_r_pose.png'
    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, int_img, tlx_pct, tly_pct, brx_pct, bry_pct)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img

def instruction_battle(cv2_img):
    alpha = 0.8
    overlay = cv2_img.copy()
    img, tlx, tly, brx, bry = find_box(overlay, './UI_images/instruction_battle.png', 0.2805, 0.2914, 0.7219, 0.5792)
    overlay = overlay_transparent(overlay, img, tlx, tly, (brx - tlx, bry - tly))
    cv2_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, -1)
    return cv2_img