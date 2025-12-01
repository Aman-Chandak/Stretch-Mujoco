import mujoco
import numpy as np
import cv2
import time
import mujoco.viewer 

IMG_WIDTH = 480
IMG_HEIGHT = 360
CAMERA_RGB = "d435i_camera_rgb"
CAMERA_DEPTH = "d435i_camera_depth"
OBJECT_BODY_NAME = "object2"


model_path = "/home/aman/stretch_mujoco/models/scene.xml" 

GRIPPER_OPEN_CTRL = 0.03
GRIPPER_CLOSE_CTRL = -0.002

TARGET_WRIST_PITCH = 0.0 
TARGET_WRIST_ROLL = 0.0 

PRE_GRASP_Z_OFFSET = 0.4 
GRASP_Z_OFFSET = -0.07
LIFT_Z_OFFSET = 0.15

STEPS_PER_PHASE_SHORT = 100
STEPS_PER_PHASE_MOVE = 1000 # Combined steps for Lift+Extend for one phase


model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


actuator_ids = {
    name: model.actuator(name).id
    for name in ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt"]
}
joint_ids = {
    name: model.joint(name).id
    for name in ["joint_lift", "joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"]
}
arm_joint_names = [name for name in model.names.decode('utf-8').split('\x00') if name.startswith('joint_arm_l')]
arm_joint_ids = [model.joint(name).id for name in arm_joint_names]
joint_ids["gripper_slide"] = model.joint("joint_gripper_slide").id


cam_id_rgb = model.camera(CAMERA_RGB).id
cam_id_depth = model.camera(CAMERA_DEPTH).id


viewer = mujoco.viewer.launch_passive(model, data)
renderer = mujoco.Renderer(model, IMG_HEIGHT, IMG_WIDTH)


print("Moving head to look at table...")
data.ctrl[actuator_ids["head_pan"]] = -1.5
data.ctrl[actuator_ids["head_tilt"]] = -0.8
for _ in range(500):
    mujoco.mj_step(model, data)
    if viewer and viewer.is_running(): viewer.sync()

print("Head movement complete.")
time.sleep(0.5)


print("Starting perception phase...")
fovy_rad = np.deg2rad(model.cam_fovy[cam_id_rgb])
fy = IMG_HEIGHT / (2 * np.tan(fovy_rad / 2))
fx = IMG_WIDTH / IMG_HEIGHT * fy
cx = IMG_WIDTH / 2
cy = IMG_HEIGHT / 2

renderer.update_scene(data, camera=CAMERA_RGB)
rgb_image = renderer.render()

renderer.enable_depth_rendering()
renderer.update_scene(data, camera=CAMERA_DEPTH)
depth_image = renderer.render()
renderer.disable_depth_rendering()

rgb_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

p_world_object = None

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx_pixel = int(M["m10"] / M["m00"])
        cy_pixel = int(M["m01"] / M["m00"])
        cx_pixel = np.clip(cx_pixel, 0, IMG_WIDTH - 1)
        cy_pixel = np.clip(cy_pixel, 0, IMG_HEIGHT - 1)

        metric_depth = depth_image[cy_pixel, cx_pixel]

        if metric_depth > 0:
            x_cam_opt = (cx_pixel - cx) * metric_depth / fx
            y_cam_opt = (cy_pixel - cy) * metric_depth / fy
            p_cam_optical = np.array([x_cam_opt, y_cam_opt, metric_depth])

            R_opt_to_gl = np.array([[1,  0,  0], [0, -1,  0], [0,  0, -1]])
            p_cam_gl = R_opt_to_gl @ p_cam_optical

            cam_pos = data.cam_xpos[cam_id_rgb]
            cam_mat = data.cam_xmat[cam_id_rgb].reshape(3, 3)
            p_world_object = cam_pos + cam_mat @ p_cam_gl

            print(f"Detected object centroid at pixel: ({cx_pixel}, {cy_pixel})")
            print(f"Metric depth value read directly: {metric_depth:.4f} meters")
            print(f"Calculated World Coordinates: {p_world_object}")

            obj_id = model.body(OBJECT_BODY_NAME).id
            gt_pos_world = data.body(OBJECT_BODY_NAME).xpos
            print(f"Ground Truth World Coordinates ({OBJECT_BODY_NAME}): {gt_pos_world}")

        else:
            print("Invalid depth value read at object pixel.")
    else:
        print("Detected contour has zero area.")
else:
    print("No red object detected.")


if p_world_object is not None:
    print("\nStarting Grasping Sequence ")

    # Calculate Target Controls 
    def calculate_heuristic_controls(target_pos_world):
        base_pos = data.body('base_link').xpos
        delta_pos_world = target_pos_world - base_pos

        target_lift_ctrl = target_pos_world[2] * 0.8 
        lift_range = model.joint('joint_lift').range
        target_lift_ctrl = np.clip(target_lift_ctrl, lift_range[0], lift_range[1])

        dist_xy = np.linalg.norm(delta_pos_world[:2])
        effective_dist = max(0, dist_xy - 0.37) 
        arm_ctrl_range = model.actuator('arm').ctrlrange
        target_arm_ctrl = (effective_dist / 0.6) * (arm_ctrl_range[1] - arm_ctrl_range[0]) + arm_ctrl_range[0] 
        target_arm_ctrl = np.clip(target_arm_ctrl, arm_ctrl_range[0], arm_ctrl_range[1])
        target_yaw_angle = -np.arctan2(delta_pos_world[1], delta_pos_world[0])
        print(f"Yaw angle:{target_yaw_angle}")
        target_wrist_yaw_ctrl = target_yaw_angle - 1 
        yaw_range = model.actuator('wrist_yaw').ctrlrange
        target_wrist_yaw_ctrl = np.clip(target_wrist_yaw_ctrl, yaw_range[0], yaw_range[1])

        return target_lift_ctrl, target_arm_ctrl, target_wrist_yaw_ctrl

    #Simulate steps
    def simulate_steps(duration_steps):
         for _ in range(duration_steps):
            mujoco.mj_step(model, data)
            if viewer and viewer.is_running(): viewer.sync()
         time.sleep(0.5) 


    #  Open Gripper
    print("Opening gripper...")
    data.ctrl[actuator_ids['gripper']] = GRIPPER_OPEN_CTRL
    simulate_steps(STEPS_PER_PHASE_SHORT)

    # Pre-Grasp Movement 
    print("Moving to pre-grasp position...")
    pos_pre_grasp = p_world_object + np.array([0, 0, PRE_GRASP_Z_OFFSET])
    target_lift, target_arm, target_yaw = calculate_heuristic_controls(pos_pre_grasp)

    # Stage 1a: Lift arm vertically and orient wrist
    print(f"  Stage 1a: Lifting to target lift ctrl: {target_lift:.2f}")
    data.ctrl[actuator_ids['lift']] = target_lift
    data.ctrl[actuator_ids['wrist_pitch']] = TARGET_WRIST_PITCH
    data.ctrl[actuator_ids['wrist_roll']] = TARGET_WRIST_ROLL
    data.ctrl[actuator_ids['arm']] = 0.0 # Retracted arm
    data.ctrl[actuator_ids['wrist_yaw']] = 0.0 # Neutral yaw
    simulate_steps(STEPS_PER_PHASE_MOVE // 2) # Half duration for lift

    # Stage 1b: Extend arm and yaw wrist
    print(f"  Stage 1b: Extending to target arm ctrl: {target_arm:.2f}, yaw ctrl: {target_yaw:.2f}")
    data.ctrl[actuator_ids['arm']] = target_arm
    data.ctrl[actuator_ids['wrist_yaw']] = target_yaw
    # Keep lift, pitch, roll from previous stage
    data.ctrl[actuator_ids['lift']] = data.ctrl[actuator_ids['lift']]
    data.ctrl[actuator_ids['wrist_pitch']] = data.ctrl[actuator_ids['wrist_pitch']]
    data.ctrl[actuator_ids['wrist_roll']] = data.ctrl[actuator_ids['wrist_roll']]
    simulate_steps(STEPS_PER_PHASE_MOVE // 2) # Half duration for extend/yaw


    # Grasp Movement
    print("Moving to grasp position...")
    pos_grasp = p_world_object + np.array([0, 0, GRASP_Z_OFFSET])
    target_lift, target_arm, target_yaw = calculate_heuristic_controls(pos_grasp)

    
    print(f"  Stage 2a: Lowering to target lift ctrl: {target_lift:.2f}")
    data.ctrl[actuator_ids['lift']] = target_lift
    data.ctrl[actuator_ids['wrist_pitch']] = TARGET_WRIST_PITCH
    data.ctrl[actuator_ids['wrist_roll']] = TARGET_WRIST_ROLL
    
    data.ctrl[actuator_ids['arm']] = data.ctrl[actuator_ids['arm']]
    data.ctrl[actuator_ids['wrist_yaw']] = data.ctrl[actuator_ids['wrist_yaw']]
    simulate_steps(STEPS_PER_PHASE_MOVE // 2)

    # Stage 2b: Fine-tune Extension and Yaw
    print(f"  Stage 2b: Adjusting Extension/Yaw to arm ctrl: {target_arm:.2f}, yaw ctrl: {target_yaw:.2f}")
    data.ctrl[actuator_ids['arm']] = target_arm
    data.ctrl[actuator_ids['wrist_yaw']] = target_yaw
    # Keep lift, pitch, roll
    data.ctrl[actuator_ids['lift']] = data.ctrl[actuator_ids['lift']]
    data.ctrl[actuator_ids['wrist_pitch']] = data.ctrl[actuator_ids['wrist_pitch']]
    data.ctrl[actuator_ids['wrist_roll']] = data.ctrl[actuator_ids['wrist_roll']]
    simulate_steps(STEPS_PER_PHASE_MOVE // 2)


    # 4. Close Gripper
    print("Closing gripper...")
    data.ctrl[actuator_ids['gripper']] = GRIPPER_CLOSE_CTRL
    simulate_steps(STEPS_PER_PHASE_SHORT * 2) 


    #  Lift Object 
    print("Lifting object...")
    pos_lift_target_world = pos_grasp + np.array([0, 0, LIFT_Z_OFFSET])
    target_lift, _, _ = calculate_heuristic_controls(pos_lift_target_world)

    # Stage 3: Lift vertically
    print(f"  Stage 3: Lifting Vertically to target lift ctrl: {target_lift:.2f}")
    data.ctrl[actuator_ids['lift']] = target_lift
    # Keep other arm joints as they were during grasp
    data.ctrl[actuator_ids['arm']] = data.ctrl[actuator_ids['arm']]
    data.ctrl[actuator_ids['wrist_yaw']] = data.ctrl[actuator_ids['wrist_yaw']]
    data.ctrl[actuator_ids['wrist_pitch']] = data.ctrl[actuator_ids['wrist_pitch']]
    data.ctrl[actuator_ids['wrist_roll']] = data.ctrl[actuator_ids['wrist_roll']]
    simulate_steps(STEPS_PER_PHASE_MOVE)


    print(" Grasping Sequence Complete ")

    print("Simulation running. Close the viewer window to exit.")
    while viewer.is_running():
         viewer.sync()


else:
    print("Object not detected, skipping grasp.")
    print("Displaying simulation. Close viewer window to exit.")
    if viewer:
        while viewer.is_running():
           viewer.sync()


if viewer:
    viewer.close()

print("Script finished.")