import cv2
from utils import bounding_box_coordinates, normalization, zoomin_point, resize_point, centralized_keypoint, find_palm_xy, overlay_transparent
from annotation import annotation
from pygame import mixer
import time
import random
import json
def find_box(cv2_img, img_path, tlx_pct, tly_pct, brx_pct, bry_pct):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    tlx = int(cv2_img.shape[1] * tlx_pct)
    tly = int(cv2_img.shape[0] * tly_pct)
    brx = int(cv2_img.shape[1] * brx_pct)
    bry = int(cv2_img.shape[0] * bry_pct)
    return img, tlx, tly, brx, bry

def gamebox(img, target_poses, prev_posedata = None,  gen_pose = False, gen_pose_left = False, battle_mode = False, flip = False, repeated_poses = True):
    height, width = img.shape[0:2]
    overlay = img.copy()
    flag = True
    if not battle_mode:
        tlx = int(width * 0.7)
        tly = int(height * 0.5)
        brx = width
        bry = height
        
        if gen_pose == False and prev_posedata: #not to generate pose
            posedata = prev_posedata
        else:
            posedata, target_poses = pose_generator(target_poses, repeated_poses = repeated_poses) 
            while posedata == prev_posedata: #make sure the next pose is not the same as previous one
                posedata, target_poses = pose_generator(target_poses, repeated_poses = repeated_poses)
    
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

    if flag and (pose_h < int(target_h * 0.4) or pose_w < int(target_w * 0.4)):
        flag = False
        while pose_h < int(target_h * 0.4) or pose_w < int(target_w * 0.4):
            pose_h *= 1.1
            pose_w *= 1.1
    else:
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
    return new_img, posedata, target_poses
def pose_generator(target_poses, repeated_poses = True):
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
    # random.shuffle(target_poses)
    pose_path = random.choice(target_poses)
    if not repeated_poses:
        target_poses.remove(pose_path)
    with open (pose_path, 'r') as pose:
        posedata = json.load(pose)
    return posedata['poses'][0], target_poses

def criteria (mae, cv2_img, battle_mode_left_player = False, battle_mode_right_player = False):
    #Perfect match (0.9987696908139251, 0.00658765889408432), (0.9975094474004261, 0.00842546430265792)
    #Poor match (0.9662763866452719, 0.028538520119295172), (0.9703551168423131, 0.029034741415598055)
    tlx_pct = 0.5734375
    tly_pct = 0.09722222222222222
    brx_pct = 0.85
    bry_pct = 0.23
    if battle_mode_left_player: #box on the left
        tlx_pct = 0.29
        tly_pct = 0.15
        brx_pct = 0.45 #49
        bry_pct = 0.25 #30
    if battle_mode_right_player: #box on the right
        tlx_pct = 0.79
        tly_pct = 0.15
        brx_pct = 0.95 #99
        bry_pct = 0.25 #30
    
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

def sound_effect(sound_path):
    mixer.init()
    effect = mixer.Sound(sound_path)
    effect.set_volume(1)
    effect.play()
    time.sleep(0.6)

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

def combo(cv2_img, left_player):
    if left_player: #box on the left
        tlx_pct = 0.35
        tly_pct = 0.25
        brx_pct = 0.45
        bry_pct = 0.30
    if not left_player: #box on the right
        tlx_pct = 0.85
        tly_pct = 0.25
        brx_pct = 0.95
        bry_pct = 0.30
    img, tlx, tly, brx, bry = find_box(cv2_img, './UI_images/combo.png', tlx_pct, tly_pct, brx_pct, bry_pct)
    cv2_img = overlay_transparent(cv2_img, img, tlx, tly, (brx - tlx, bry - tly))
    return cv2_img

def screen_record(capture, output_file_path, fps = 20):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video = cv2.VideoWriter(output_file_path, fourcc, fps, size)
    return output_video
