# ros-perception

This repository contains the perception module developed for the final project of Seoul National University’s course: **How to Make a Robot with Artificial Intelligence (M2177.002600), Fall 2025**
- Original course repository: https://github.com/roy9852/make_ai_robot
- This repository focuses exclusively on our team’s **perception** system, which I was fully responsible for designing, implementing, and tuning.

---

## Project Overview
This repository includes the ROS-based perception code that enables the robot to detect objects in its field of view and trigger actions based on visual conditions.
The perception module:
- Processes camera input from ROS bag files
- Performs real-time object detection
- Publishes commands based on mission-specific rules
The final system achieved high detection accuracy, as demonstrated in the video attached below.

## 📹 Result video
<p align="center">
  <img src="https://github.com/user-attachments/assets/2305a7b9-3319-476c-bbb6-4be45394015f" width="400"/>
  <img src="https://github.com/user-attachments/assets/12fbcf45-6bdb-47a2-a62a-b1737d879751" width="400"/>
</p>


## Repository Structure
```
datasets/            # Training data manually collected from given map file then labeled
experiments/         # Saved result after training
scripts/             # Scripts used for fine tuning
src/                 
 ├─ perception/      # Main perception module (object detection & decision logic)
 └─ module_test/     # Interface viewer and test utilities
```
- The interface viewer is used to visualize the robot’s camera view.
- A recorded ROS bag file containing the robot’s visual input is played back([link](https://drive.google.com/file/d/1aQZ1qr5Q9m-JcWN4qgewoQ4dtE0N_2QQ/view)).
- During playback, the implemented perception module detects objects and publishes outputs accordingly.

## Perception Mission Description
- The perception mission is defined as follows (more detail on original repo):
  - If an edible object is detected whose bounding-box center lies within the middle 3/5 of the image → publish "bark"
- Additional constraints:
  - The system must accurately detect objects within a 3-meter radius
  - Detection must be robust under realistic robot operating conditions

## Model & Training
- The perception system is based on **YOLOv8s**
- Training data was manually labeled using **Roboflow**
The model was fine-tuned multiple times
Additional images were incrementally added based on experimental results
Each iteration improved detection accuracy and stability


## Environment Setup
- This project was developed and tested on Ubuntu 22.04 with ROS Jazzy, using a Python virtual environment (venv) for dependency management. However, since this repository only contains the implementation of the perception module, it does not include full system build or deployment configurations. For detailed instructions on environment setup, workspace configuration, and overall system build, please refer to the original course repository.

  ```bash
  $ python3.10 -m venv venv
  $ source venv/bin/activate
  $ (venv) $ pip install -r requirements.txt
  ```
  
## How to Run
- YOLO training / fine-tuning can be executed using the following script:

  ```bash
  $ source ~/ros-perception/venv/bin/activate
  $ (venv) python ./scripts/train.py
  $ (venv) python ./scripts/finetune_pizza.py 
  ```

- The perception module can be executed using the following script:

  ```
  $ source ~/ros-perception/venv/bin/activate
  $ (venv) export PYTHONNOUSERSITE=1
  $ (venv) ~/ros-perception/venv/bin/python3 -m perception.perception_node
  $ (venv) ~/ros-perception/venv/bin/python3 -m module_test.interface_viewer
  ```

---
Last Updated: Dec 26, 2025