# MoveNow
## About
***MoveNow*** is a pose matching game which has two game modes for now. ***The Normal mode*** is to follow some standard poses and get scores. By following some professional poses, it can help improve the users' posture or guide them to the right direction. Besides, ***the Battle mode***, literally, let two users play with each other which is more interactive and fun. Apart from just playing with randomly poses, the user is also allowed to input a collection of poses on his/her own.
## Gameplay (No Sound)
#### Click the preview picture below 
[![](http://img.youtube.com/vi/cmCYNqqbHUU/0.jpg)](http://www.youtube.com/watch?v=cmCYNqqbHUU "MoveNow")
## Steps (Terminal)
```
git clone
pip install -r requirements.txt

#with GPU
python main.py --flip --useGPU --n-poses 10 

#without GPU
python main.py --flip --n-poses 10 --scale-factor 0.5
"""
if it doesn't work, try python3.
Besides, use GPU with scale factor, which is set to 1, would have better experience.
"""
```
## Reminders
- Download ***pytorch with gpu version*** if your computer has GPU.

- If you are using CPU and the fps is low, please adjust the scale factor to ***below 0.5*** but it will affect the accuracy of Human Pose Estimation.

- Press ***Q*** to quit the program

## Upload Your Own Poses / Background Music
### Your Own Poses (can only handle some easy poses)
- Save images of poses to ***MoveNow/players***, default: ***JPG***
- Run pose_models.py on the terminal
- Check if the output (json file) is saved to ***MoveNow/players/target_posedata/json***

### Background Music
- Add your favourite background music to ***MoveNow/background_music***, default: ***MP3***

## Arguments
***1. --weight-dir***, type = str, default = ***"./model_"***
- Path to the model weight
   
***2. --model-id***, type = int, choices = [50, 75, 100, 101], default = ***101***

***3. --useGPU***, default = ***False***

***4. --scale-factor***, type = float, choices = 0 to 1, default = ***1***
- Factor to scale down the image to process
	
***5. --flip***, default = ***False***
- Flip the screen if it's inverse
	
***6. --sec***, type = float, default = ***5.0***
- How many second to change a new pose

***7. --n-poses***, type = int, default = ***10***
- How many poses you wanna play with

***8. -o / --output-video***, default = ***False***
- Record a gameplay for your first play (***works in macOS***)
	
***9. --output-name***, type = str, default = ***"gameplay.mp4"***
- The name of the output video

***10. --output-fps***, type = float, default = ***20***
- The output video's fps

***11. --imwrite***, default = ***False***
- Save the result pic to the result_images directory (***works in macOS***)
	
***12. --repeated-poses***, default = ***False***
- Repeat the poses in ***Normal Mode***


