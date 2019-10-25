# miro_intern_week1: 2D-Multi-Person Human Pose Estimation
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

> If you wish to make an apple pie from scratch, you must first invent the universe.
> ~ Carl Sagan

## Goals
1. Implement [PoseNet on PyTorch](https://github.com/rwightman/posenet-pytorch) and get it to output pose data in [this json format](_testset/json/1.json)
    *   the model weights you need can be accessed [here](http://files.johnho.ca/owncloud/index.php/s/3eZUxj0om7n3bdX)
2. Come up with a pose evaluation metrics to compare your results to [this test set](_testset/)
    *   Some level of annotation might help here. See example [here](_testset/annotated_img/).

Note that some sample code are already included [here](pose_models.py) and [here](evaluate.py) to help you get started but definitely feel free to change them as needed.

## Requirements
To avoid [dependency hell](https://medium.com/knerd/the-nine-circles-of-python-dependency-hell-481d53e3e025) let's all use [virtualenv](https://virtualenv.pypa.io/en/latest/) for this project. I highly recommend [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

In [requirements.txt](requirements.txt) I have included a few useful packages for this project. Please keep adding to this file as you go.

So using **Python 3.7** and **virtualenv** we should all be able to create the same environment by:
```
mkvirtualenv --python=`which python3.7` NameOfYourEnv
$ workon NameOfYourEnv
(NameOfYourEnv) $ pip install -r requirements.txt
```

## Guideline
Use [pose_models.py](pose_models.py)
#### 1. Load Model
```
model = LoadModel(verbose= True)
```
#### 2. Predict Poses and Save results into json file
```
ls_img = sorted([img.path for img in os.scandir('./_testset')if img.path.endswith('.jpg')]) #all images
for img_path in ls_img:
    print('\n', img_path,'\n')
    pose_data, image_name = PredictPose(model, img_path)
    save_to_json(pose_data, image_name, output_json_path= './_testset/json_with_neck')
```
Use [evaluate.py](evaluate.py)
#### 3. Compare two posedata
```
for i in range(1,8):
    with open(os.getcwd() + f'/_testset/json/{i}.json','r') as doc:
        TargetPoseData = json.load(doc)
    with open(os.getcwd() + f'/_testset/json_with_neck/00000{i}.json','r') as doc:
        TestPoseData = json.load(doc)
    print(f'\n\nIn image {i}:\n\n')
    print(PoseCompare( TargetPoseData, TestPoseData, normalize = True))
```
Use [annotation.py](annotation.py)
#### 4. Annotate Poses
```
#Save Annotated Image(s)
img_list = sorted([img.path for img in os.scandir('./_testset') if img.path.endswith('.' + 'jpg')])
posedata_list = sorted([pose.path for pose in os.scandir('./_testset/json_with_neck') if pose.path.endswith('.json')])

cv2_img_list = [cv2.imread(img) for img in img_list]
        
save_annotated_img (cv2_img_list, posedata_list, save_dir = "./annotated_img", indi_pose = True, indi_pose_dir = "./indi_annotated_img", img_format = 'jpg', keypoint_min_score = 0.5, keypoints_ratio = 0.5, threshold_denoise = 0.03, resize = False, resize_W = 200, resize_H = 400)
```
```
#Display Annotated Image(s) in jupyter notebook
img_list = sorted([img.path for img in os.scandir('./_testset') if img.path.endswith('.' + 'jpg')])
posedata_list = sorted([pose.path \
    for pose in os.scandir('./_testset/json_with_neck') if pose.path.endswith('.json')])

cv2_img_list = [cv2.imread(img) for img in img_list]

display_in_ipynb(cv2_img_list, posedata_list, keypoint_min_score = 0.5, keypoints_ratio = 0.5, threshold_denoise = 0.03, resize = True, resize_W = 200, resize_H = 400)

```

