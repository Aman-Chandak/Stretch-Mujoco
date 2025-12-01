# Stretch MuJoCo Object Detection and Grasping

This repository contains Python scripts for simulating a Hello Robot Stretch RE3 robot in MuJoCo. It demonstrates how to perform object detection using a simulated RGB-D camera and execute a heuristic-based pick-and-place task.

## Overview

The project consists of two main scripts:
1.  **`stretch/detect.py`**: Performs object detection and 3D position estimation.
2.  **`stretch/detect_grasp.py`**: Integrates detection with a control sequence to grasp and lift the detected object.

The simulation uses a custom MuJoCo model of the Stretch robot (`models/stretch.xml`) and a scene with a table and objects (`models/scene.xml`).

## Prerequisites

Ensure you have the following Python packages installed:

*   `mujoco`
*   `numpy`
*   `opencv-python`

You can install them using pip:

```bash
pip install mujoco numpy opencv-python
```

## Usage

### 1. Object Detection (`detect.py`)

This script loads the simulation, moves the robot's head to look at the table, and captures RGB and Depth images. It then:
*   Detects a red object using HSV color thresholding.
*   Calculates the object's centroid in pixel coordinates.
*   Reads the depth value at the centroid.
*   Deprojects the 2D pixel + depth to a 3D point in the camera frame.
*   Transforms the 3D point to the world frame using the camera's pose.
*   Displays the RGB image with the detected contour and the Depth image.

**Run command:**
```bash
python3 stretch/detect.py
```

### 2. Detection and Grasping (`detect_grasp.py`)

This script builds upon the detection logic to perform a manipulation task. It launches a MuJoCo passive viewer to visualize the robot's actions in real-time.

**Sequence of actions:**
1.  **Head Movement**: Tilts the head down to view the workspace.
2.  **Perception**: Captures images, detects the target object, and calculates its 3D world coordinates.
3.  **Grasping Sequence**:
    *   **Approach**: Calculates inverse kinematics (heuristics) for the lift, arm extension, and wrist yaw to align with the object.
    *   **Pre-Grasp**: Moves the gripper to a position directly above the object.
    *   **Descend**: Lowers the lift to bring the gripper around the object.
    *   **Grasp**: Closes the gripper fingers.
    *   **Lift**: Raises the lift mechanism to pick up the object.

**Run command:**
```bash
python3 stretch/detect_grasp.py
```

## File Structure

*   `stretch/`
    *   `detect.py`: Standalone object detection script using OpenCV visualization.
    *   `detect_grasp.py`: Integrated detection and control script using MuJoCo viewer.
*   `models/`
    *   `scene.xml`: The main MuJoCo scene file including the robot, table, and objects.
    *   `stretch.xml`: The Stretch robot model definition.
    *   `assets/`: Directory containing meshes and textures for the robot and environment.

## Notes

*   The detection logic assumes a specific red object (`object2` in `scene.xml`). You may need to adjust HSV thresholds in the scripts if you change the object's color.
*   The grasping logic uses a heuristic approach specific to the Stretch robot's kinematics (Lift-Arm-Wrist) and may need tuning for different object positions or sizes.
