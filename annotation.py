import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
import os
from utils import normalization, bounding_box_coordinates, drawing
from tqdm import tqdm
import re

def annotation (cv2_img, posedata,keypoint_min_score = 0.5, keypoints_ratio = 0.5, threshold_denoise = 0.03, normalized = True, 
                pose_id = 0, resize = True, resize_W = 200, resize_H = 400, scale_x = 0.1, scale_y = 0.1):
    """
    posedata: dictionary output from json file
    keypoint_min_score: threshold to determine whether to keep a keypoint, default 0.5
    keypoints_ratio: if the total number of "eye", "ear", "nose" and "neck" 
                    is equal to or greater than certain percentage of the total number of keypoints, default 0.5 (i.e 50%)
    threshold_denoise: within 0.01 - 0.1, default 0.03 (trial and error)
    pose_id: a mark on the annotated image
    resize: True if resize the normalized pose
    resize_W: width of the new pose
    resize_H: height of the new pose
    """
    keys = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
            'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
            'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'neck']
    color_BGR = [(255,178,102), (255,178,102), (255,178,102), (255,178,102), (255,178,102), (255,255,102),
                 (255,255,102), (255,255,102), (255,255,102), (255,255,102), (255,255,102),
                 (178,255,102), (178,255,102),(178,255,102),(178,255,102),
                 (178,255,102), (178,255,102), (255,178,102)]
    color_dict = dict(zip(keys,color_BGR))
    line_pair = [('l_wrist','l_elbow'),('l_elbow','l_shoulder'),
                ('r_wrist','r_elbow'),('r_elbow','r_shoulder'),
                ('l_ankle','l_knee'),('l_knee','l_hip'),
                ('r_ankle','r_knee'),('r_knee','r_hip'),
                # ('l_eye','l_ear'),('r_eye','r_ear'),
                ('l_shoulder','l_hip'),('r_shoulder','r_hip'),
                ('l_hip','r_hip'),('l_shoulder','r_shoulder')]
                # ('l_eye','r_eye'),
                # ('l_eye','nose'),('r_eye','nose'),('nose','l_shoulder'),('nose','r_shoulder'),
                # ('neck','nose'),('neck','l_shoulder'),('neck','r_shoulder')]

    upper_part = ("eye", "ear", "nose", "neck")
    # middle_part = ("wrist", "elbow", "shoulder")
    # lower_part = ("ankle", "knee", "hip")

    # cv2_img = np.zeros((cv2_img.shape[0], cv2_img.shape[1], 3))
    # cv2_img.fill(0)
    overlay = cv2_img.copy()
    alpha = 0.8 #transparency
    new_img = None

    area =  cv2_img.shape[0]*cv2_img.shape[1]  #height * width
    lineThickness, circle_radius, fontScale, space, thickness = drawing(area) #for drawing lines, circles, texts, rectangle
    
    keypoints = posedata.keys()
    keypoints_keep = [keypoint for keypoint in keypoints \
        if posedata[keypoint]['conf'] >= keypoint_min_score] #only select keypoins with conf >= keypoint_min_score
    keypoint_list_x = [round(posedata[keypoint]['x']) for keypoint in keypoints]
    keypoint_list_y = [round(posedata[keypoint]['y']) for keypoint in keypoints]
    mean_xdiff = np.mean(abs(np.diff(keypoint_list_x)/cv2_img.shape[1])) #based on the width
    mean_ydiff = np.mean(abs(np.diff(keypoint_list_x)/cv2_img.shape[0])) #based on the height
    
    upper_part_score = 0
    # middle_part_score = 0
    # lower_part_score = 0
    for upper in upper_part:
        for keypoint in keypoints_keep:
            if upper in keypoint:
                upper_part_score += 1

    # for middle in middle_part:
    #     for keypoint in keypoints_keep:
    #         if middle in keypoint:
    #             middle_part_score += 1

    # for lower in lower_part:
    #     for keypoint in keypoints_keep:
    #         if lower in keypoint:
    #             lower_part_score += 1

    if upper_part_score / len(keypoints_keep) >= keypoints_ratio or \
        (mean_xdiff <= threshold_denoise and mean_ydiff<=threshold_denoise): 
        if normalized:
            # cv2_img = np.zeros((200,200,3))
            # cv2_img.fill(255)
            return None
        else:
            return cv2_img
  
    max_x_boundary, min_x_boundary, max_y_boundary, min_y_boundary = bounding_box_coordinates(keypoint_list_x, keypoint_list_y, scale_x = scale_x, scale_y = scale_y)
    
    #make sure the boundary didn't exceed the original image
    if  max_x_boundary > overlay.shape[1]:
        max_x_boundary = overlay.shape[1]
    if min_x_boundary < 0:
        min_x_boundary = 0
    if max_y_boundary > overlay.shape[0]:
        max_y_boundary = overlay.shape[0]
    if min_y_boundary < 0:
        min_y_boundary = 0    

    if normalized: 
        org_W = max_x_boundary - min_x_boundary #bounding box's width
        org_H = max_y_boundary - min_y_boundary #bounding box's height
        overlay = overlay[min_y_boundary: min_y_boundary + org_H, min_x_boundary :min_x_boundary + org_W] #crop the pose
        new_keypoint_list_x = np.array(keypoint_list_x) - min_x_boundary #find new keypoints' coordinate after cropping
        new_keypoint_list_y = np.array(keypoint_list_y) - min_y_boundary #find new keypoints' coordinate after cropping
       

        posedata, new_keypoint_list_x, new_keypoint_list_y = normalization (new_keypoint_list_x, new_keypoint_list_y, posedata, 600, 600)
        max_x_boundary, min_x_boundary, max_y_boundary, min_y_boundary = bounding_box_coordinates(new_keypoint_list_x, new_keypoint_list_y, scale_x = scale_x, scale_y = scale_y)
        
        overlay = np.zeros(((max_y_boundary - min_y_boundary),(max_x_boundary - min_x_boundary),3)) 
        overlay.fill(255) #create a white image
        
        new_area = (max_y_boundary - min_y_boundary) * (max_x_boundary - min_x_boundary)
        if resize:
            new_area = resize_W * resize_H #width * height
        lineThickness, circle_radius, fontScale, space, thickness = drawing(new_area) #updated 

        cv2_img = overlay.copy()

    for keypoint1, keypoint2 in line_pair:
        if keypoint1 in keypoints_keep and keypoint2 in keypoints_keep:
            keypoint1_xy = (int(round(posedata[keypoint1]['x'])), 
            int(round(posedata[keypoint1]['y'])))
            
            keypoint2_xy = (int(round(posedata[keypoint2]['x'])), 
            int(round(posedata[keypoint2]['y'])))

            cv2.line(overlay, keypoint1_xy, keypoint2_xy, color_dict[keypoint1], int(lineThickness))

    for keypoint in keypoints_keep:
        keypoint_x = round((posedata[keypoint]['x']))
        keypoint_y = round((posedata[keypoint]['y']))

        cv2.circle(overlay,(int(keypoint_x), int(keypoint_y)), int(circle_radius), (102, 102, 255), -1)
    
    pt1 = (int(min_x_boundary), int(max_y_boundary)) #bottom right
    rec_color = np.random.rand(3) * 255

    if not normalized:
        pt1 = (int(min_x_boundary), int(min_y_boundary))
        # cv2.rectangle(overlay, pt1, (int(max_x_boundary), int(max_y_boundary)), rec_color, int(lineThickness))
        # cv2.rectangle(overlay, pt1, (int(max_x_boundary), int(max_y_boundary)), (224,224,224), cv2.FILLED)
    
    # cv2.putText(overlay,f"id {pose_id}",(pt1[0],pt1[1]-space),cv2.FONT_HERSHEY_COMPLEX,fontScale,rec_color,int(thickness))
    new_img = cv2.addWeighted(overlay, alpha, cv2_img, 1 - alpha, 0) #transparent
    
    if normalized and resize:
        new_img = cv2.resize(new_img, (resize_W, resize_H))
        
    if type(new_img) == type(None):
        return cv2_img
    else:
        return new_img

def save_annotated_img (cv2_img_list, posedata_list, save_dir = "./annotated_img", indi_pose = True, indi_pose_dir = "./indi_annotated_img", 
                        img_format = 'jpg', keypoint_min_score = 0.5, keypoints_ratio = 0.5, threshold_denoise = 0.03,
                        resize = True, resize_W = 200, resize_H = 400):
    '''
    cv2_img_list: should be a list of images, if there is only one image, please put it into a list as well
    posedata_list: should be a list of path, if there is only one posedata, please put it into a list as well
    save_dir: path to saving images
    indi_pose: True if save individual pose image
    indi_pose_dir: path to saving individual pose image
    img_format: input and output image format
    keypoint_min_score: threshold to determine whether to keep a keypoint, default 0.5
    min_keypoints: if the total number of "eye", "ear", "nose" and "neck" 
                    is equal to or greater than certain percentage of the total number of keypoints, default 0.5 (i.e 50%)
    threshold_denoise: within 0.01 - 0.1, default 0.03 (trial and error)
    resize: True if resize the normalized pose
    resize_W: width of the new pose
    resize_H: height of the new pose
    '''
    assert type(img_list) == list, "Should be a list of cv2 images!"
    assert type(posedata_list) == list, "Should be a list of paths!"
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(indi_pose_dir) and indi_pose == True:
        os.mkdir(indi_pose_dir)
    
    file_names = [re.search(r'(?:\/.+\/)(.+\.json)', path).group(1)[:-5] for path in posedata_list]
    
    for cv2_img, posedata, file_name in tqdm(zip(cv2_img_list,posedata_list, file_names)):
#         cv2_img = cv2.imread(img)
        cv2_img_individual = cv2_img.copy()
        with open(posedata, 'r') as doc:
            posedata_ = json.load(doc)['poses']
        for id_, poser in enumerate(posedata_):
            cv2_img = annotation (cv2_img, poser, keypoint_min_score = keypoint_min_score, keypoints_ratio = keypoints_ratio, threshold_denoise = threshold_denoise, normalized= False, pose_id = id_, 
            resize = resize, resize_W = 200, resize_H = 400)
            if indi_pose:
                cv2_img_ = annotation (cv2_img_individual, poser, keypoint_min_score = keypoint_min_score, keypoints_ratio = keypoints_ratio, threshold_denoise = threshold_denoise, normalized= True, pose_id = id_, 
                resize = resize, resize_W = 200, resize_H = 400)

                if type(cv2_img_) != type(None):
                    cv2.imwrite(os.path.join(indi_pose_dir, file_name + '_' + str(id_) +'_.jpg'), cv2_img_)
        cv2.imwrite(os.path.join(save_dir, file_name + '.' + img_format), cv2_img)



def display_in_ipynb(cv2_img_list, posedata_list, keypoint_min_score = 0.5, keypoints_ratio = 0.5, threshold_denoise = 0.03, resize = True, resize_W = 200, resize_H = 400):
    '''
    cv2_img_list: should be a list of cv2 images, if there is only one image, please put it into a list as well
    posedata_list: should be a list of paths, if there is only one posedata, please put it into a list as well
    keypoint_min_score: threshold to determine whether to keep a keypoint, default 0.5
    min_keypoints: if the total number of "eye", "ear", "nose" and "neck" 
                    is equal to or greater than certain percentage of the total number of keypoints, default 0.5 (i.e 50%)
    threshold_denoise: within 0.01 - 0.1, default 0.03 (trial and error)
    resize: True if resize the normalized pose
    resize_W: width of the new pose
    resize_H: height of the new pose
    '''
    assert type(img_list) == list, "Should be a list of cv2 images!"
    assert type(posedata_list) == list, "Should be a list of paths!"
    file_names = [re.search(r'(?:\/.+\/)(.+\.json)', path).group(1)[:-5] for path in posedata_list]

    for cv2_img, posedata, file_name in tqdm(zip(cv2_img_list,posedata_list, file_names)):
        print(f'[Posedata: {file_name}]')
        ax1 = plt.figure(figsize = (10,10))
        ax1 = plt.subplots_adjust(wspace=0.5, hspace= 0.5)
        cv2_img_individual = cv2_img.copy()
        with open(posedata, 'r') as doc:
            posedata_ = json.load(doc)['poses']
        
        i = 1 
        num_row = round(len(posedata_) / 4) #total number of row in the subplot
        if num_row <= 0:
            num_row = 1
        for id_, poser in enumerate(posedata_):
            cv2_img = annotation (cv2_img, poser, keypoint_min_score = keypoint_min_score, keypoints_ratio = keypoints_ratio, threshold_denoise = threshold_denoise, normalized= False, pose_id = id_, 
            resize = resize, resize_W = 200, resize_H = 400)
            cv2_img_ = annotation (cv2_img_individual, poser, keypoint_min_score = keypoint_min_score, keypoints_ratio = keypoints_ratio, threshold_denoise = threshold_denoise, normalized= True, pose_id = id_, 
            resize = resize, resize_W = 200, resize_H = 400)

            if type(cv2_img_) == type(None):
                continue
            else:
                ax1 = plt.subplot(num_row, 4, i)
                ax1 = plt.imshow(cv2_img_.astype('int64'))
                i += 1
    #     ax = plt.subplot(num_row, 1, 1)
        ax2 = plt.figure(figsize = (10,10))
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        ax2 = plt.imshow(cv2_img.astype('int64'))
        plt.show()
if __name__ == '__main__':
    img_list = sorted([img.path for img in os.scandir('./players/target') if img.path.endswith('.' + 'jpg')])
    posedata_list = sorted([pose.path \
        for pose in os.scandir('./players/target/json') if pose.path.endswith('.json')])

    cv2_img_list = [cv2.imread(img) for img in img_list]
    
    save_annotated_img (cv2_img_list, posedata_list, save_dir = "./players/target/image", indi_pose = False, indi_pose_dir = "./players/image", img_format = 'jpg', 
                    keypoint_min_score = -1, keypoints_ratio = 1, threshold_denoise = 0.03, resize = True, resize_W = 200, resize_H = 400)