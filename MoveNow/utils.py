import numpy as np
import cv2
import time
def bounding_box_coordinates (keypoint_list_x, keypoint_list_y, scale_x = 0.1, scale_y = 0.1):
    max_y = max(keypoint_list_y)
    min_y = min(keypoint_list_y)
    max_x = max(keypoint_list_x)
    min_x = min(keypoint_list_x)
    margin_y = round((max_y - min_y) * scale_y)
    margin_x = round((max_x - min_x) * scale_x)
    max_x_boundary = max_x + margin_x
    min_x_boundary = min_x - margin_x
    max_y_boundary = max_y + margin_y
    min_y_boundary = min_y - margin_y
    return max_x_boundary, min_x_boundary, max_y_boundary, min_y_boundary

def drawing (img_area):
    if img_area < 500000:
        #   # line
        #     lineThickness = 2
        #     # circle
        #     circle_radius = 3
        #     # text
        #     fontScale = 0.6
        #     space = 6
        #     thickness = 2
            # line
            lineThickness = 8
            # circle
            circle_radius = 4
            # text
            fontScale = 0.6
            space = 9
            thickness = 2
    elif img_area < 1000000:
            # line
            lineThickness = 10
            # circle
            circle_radius = 7
            # text
            fontScale = 0.6
            space = 9
            thickness = 2
    elif img_area < 5000000:
            # line
            lineThickness = 12
            # circle
            circle_radius = 6
            # text
            fontScale = 2
            space = 10
            thickness = 2
    else:
            # line
            lineThickness = 8
            # circle
            circle_radius = 10
            # text
            fontScale = 2
            space = 10
            thickness = 2
    return lineThickness, circle_radius, fontScale, space, thickness

# def normalization (cv2_img, target_W, target_H):
#     w_ratio = target_W / cv2_img.shape[1]
#     h_ratio = target_H / cv2_img.shape[0] 
#     cv2_img = cv2.resize(cv2_img, (target_W,target_H))
#     return cv2_img, w_ratio, h_ratio

def normalization (keypoint_list_x, keypoint_list_y, posedata, target_W, target_H):
    euclidean_distance =  np.sqrt(sum(keypoint_list_x ** 2 + keypoint_list_y ** 2))
    # mean_x = np.mean(keypoint_list_x)
    # mean_y = np.mean(keypoint_list_y)
    new_keypoint_list_x = []
    new_keypoint_list_y = []
    for keypoint, x, y in zip(posedata.keys(), keypoint_list_x, keypoint_list_y):
        temp_x = int(round((x) / euclidean_distance * target_W ))
        temp_y = int(round((y)/ euclidean_distance * target_H ))
        posedata[keypoint]['x']= temp_x
        posedata[keypoint]['y']= temp_y
        new_keypoint_list_x.append(temp_x)
        new_keypoint_list_y.append(temp_y)
        # print(keypoint, 'Oldx: ', x, 'new: ', posedata[keypoint]['x'])
        # print(keypoint, 'Oldy: ', y, 'new: ', posedata[keypoint]['y'])

    return posedata, new_keypoint_list_x, new_keypoint_list_y

def build_neck(l_shoulder_data, r_shoulder_data):
    neck_x = int(round((l_shoulder_data['x'] + r_shoulder_data['x']))/ 2)
    neck_y = int(round(l_shoulder_data['y'] + r_shoulder_data['y']) / 2)
    neck_conf = (l_shoulder_data['conf'] + r_shoulder_data['conf']) / 2
    return neck_x, neck_y, neck_conf

def resize_point(posedata,new_width,new_height,pt1=None,pt2=None,cv2_img=None):
    """
    posedata: cv2_img,1 pose dictionary output from json file
    output: resize_image,new_posedata
    """
    try:
        height = pt2[1] - pt1[1]
        width = pt2[0] - pt1[0]
    except:
        height,width,channel = cv2_img.shape
    keypoints = posedata.keys()
    keypoint_list_x = [round(posedata[keypoint]['x']) for keypoint in keypoints]
    keypoint_list_y = [round(posedata[keypoint]['y']) for keypoint in keypoints]
    keypoint_list_conf = [posedata[keypoint]['conf'] for keypoint in keypoints]

    x_ratio = new_width/width
    y_ratio = new_height/height
    # print("resize_point: ")
    # print("X_ratio: ",x_ratio)
    # print("Y_ratio: ",y_ratio)
    new_keypoint_list_x = np.array(keypoint_list_x)*x_ratio
    new_keypoint_list_y = np.array(keypoint_list_y)*y_ratio
    new_posedata = { list(keypoints)[i]:{'x':new_keypoint_list_x[i],
                                            'y':new_keypoint_list_y[i],
                                            'conf':keypoint_list_conf[i]} for i in range(len(new_keypoint_list_x))}

    if pt1==None:
        resize_image = cv2.resize(cv2_img,(new_width,new_height))
        return resize_image,new_posedata
    else:
        return new_posedata

def zoomin_point(posedata, scale_x = 0.1, scale_y = 0.1):
    """
    input: posedata: 1 pose dictionary output from json file
    output: pt1,pt2,new_posedata
    """
    keypoints = posedata.keys()
    keypoint_list_x = [round(posedata[keypoint]['x']) for keypoint in keypoints]
    keypoint_list_y = [round(posedata[keypoint]['y']) for keypoint in keypoints]
    keypoint_list_conf = [posedata[keypoint]['conf'] for keypoint in keypoints]
    max_x_boundary, min_x_boundary, max_y_boundary, min_y_boundary = bounding_box_coordinates (keypoint_list_x, keypoint_list_y, scale_x, scale_y)
    pt1 = (min_x_boundary, min_y_boundary)
    pt2 = (max_x_boundary, max_y_boundary)
    (x1,y1) = pt1
    
    new_keypoint_list_x = [x-x1   for x in keypoint_list_x]
    new_keypoint_list_y = [y-y1 for y in keypoint_list_y]
    new_posedata = { list(keypoints)[i]:{'x':new_keypoint_list_x[i],
                                        'y':new_keypoint_list_y[i],
                                        'conf':keypoint_list_conf[i]} for i in range(len(new_keypoint_list_x))}
    return pt1,pt2,new_posedata

def centralized_keypoint(target_w, target_h, posedata):
    center_x = target_w / 2
    center_y = target_h / 2 * 0.6
    stand_xpt = posedata['neck']['x']
    stand_ypt = posedata['neck']['y']
    for keypoint in posedata.keys():
        if keypoint == 'neck':
            posedata[keypoint]['x'] = center_x
            posedata[keypoint]['y'] = center_y
        else:
            posedata[keypoint]['x'] = center_x + (posedata[keypoint]['x'] - stand_xpt)
            posedata[keypoint]['y'] = center_y + (posedata[keypoint]['y'] - stand_ypt)
    return posedata

def find_palm_xy (elbow_x, elbow_y, wrist_x, wrist_y, m, n):
    palm_x = (wrist_x * (m + n) - (n * elbow_x)) // m
    palm_y = (wrist_y * (m + n) - (n * elbow_y)) // m
    return palm_x, palm_y

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2
	
	@param      background_img    The background image
	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
	@param      x                 x location to place the top-left corner of our overlay
	@param      y                 y location to place the top-left corner of our overlay
	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
	@return     Background image with overlay on top
	"""
	
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img