import mujoco
import numpy as np
import cv2

IMG_WIDTH = 480
IMG_HEIGHT = 360
CAMERA_RGB = "d435i_camera_rgb"
CAMERA_DEPTH = "d435i_camera_depth"
OBJECT_BODY_NAME = "object2" 

model_path = "/home/aman/stretch_mujoco/models/scene.xml" 


model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

data.ctrl[model.actuator("head_pan").id] = -1.5 #to look at the table
data.ctrl[model.actuator("head_tilt").id] = -0.8 #to look at the table

for _ in range(500):
    mujoco.mj_step(model, data)

cam_id_rgb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_RGB)

fovy_rad = np.deg2rad(model.cam_fovy[cam_id_rgb])
fy = IMG_HEIGHT / (2 * np.tan(fovy_rad / 2))
fx = IMG_WIDTH / IMG_HEIGHT * fy 
cx = IMG_WIDTH / 2
cy = IMG_HEIGHT / 2
print(f"RGB Camera Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")

cam_id_depth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_DEPTH)

renderer = mujoco.Renderer(model, IMG_HEIGHT, IMG_WIDTH) 


# Get RGB image
renderer.update_scene(data, camera=CAMERA_RGB)
rgb_image = renderer.render()
rgb_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
# Get Depth image
renderer.enable_depth_rendering() 
renderer.update_scene(data, camera=CAMERA_DEPTH) # Render from the depth camera perspective
depth_image = renderer.render() 
renderer.disable_depth_rendering()


print(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
min_depth_val = np.min(depth_image)
max_depth_val = np.max(depth_image)
print(f"Raw depth image value range: min={min_depth_val:.4f}, max={max_depth_val:.4f}")


# Detect red object in RGB image
hsv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx_pixel = int(M["m10"] / M["m00"])
        cy_pixel = int(M["m01"] / M["m00"])

        cx_pixel = np.clip(cx_pixel, 0, IMG_WIDTH - 1)
        cy_pixel = np.clip(cy_pixel, 0, IMG_HEIGHT - 1)
        print(f"Detected object centroid at pixel: ({cx_pixel}, {cy_pixel})")

        metric_depth = depth_image[cy_pixel, cx_pixel]
        print(f"Metric depth value read directly from pixel: {metric_depth:.4f} meters")


        # Deproject pixel coordinates to camera frame coordinates (using RGB camera intrinsics)
        x_cam = (cx_pixel - cx) * metric_depth / fx
        y_cam = (cy_pixel - cy) * metric_depth / fy
        p_cam = np.array([x_cam, y_cam, metric_depth])
        print(f"Coordinates in Camera Frame ({CAMERA_RGB}): {p_cam}")
        p_cam_optical = np.array([x_cam, y_cam, metric_depth])
        print(f"Coordinates in Camera Optical Frame (X right, Y down, Z fwd): {p_cam_optical}")

        # Convert from Optical (X right, Y down, Z fwd) to OpenGL (X right, Y up, Z backward)

        R_opt_to_gl = np.array([[1,  0,  0],
                                [0, -1,  0],
                                [0,  0, -1]])
        p_cam_gl = R_opt_to_gl @ p_cam_optical
        print(f"Coordinates converted to OpenGL Frame (X right, Y up, Z back): {p_cam_gl}")

        # Transform point from camera frame (OpenGL convention) to world frame
        cam_pos = data.cam_xpos[cam_id_rgb]
        cam_mat = data.cam_xmat[cam_id_rgb].reshape(3, 3)
        p_world = cam_pos + cam_mat @ p_cam_gl
        print(f"Calculated World Coordinates: {p_world}")

        obj_id = model.body(OBJECT_BODY_NAME).id
        gt_pos_world = data.body(OBJECT_BODY_NAME).xpos
        print(f"Ground Truth World Coordinates ({OBJECT_BODY_NAME}): {gt_pos_world}")
        gt_dist_from_cam = np.linalg.norm(gt_pos_world - cam_pos)
        print(f"Ground Truth Distance from Camera ({CAMERA_RGB}): {gt_dist_from_cam:.4f} meters")
        

        # Visualization
        cv2.drawContours(rgb_cv, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(rgb_cv, (cx_pixel, cy_pixel), 5, (255, 255, 255), -1)
        coord_text = f"World: ({p_world[0]:.2f}, {p_world[1]:.2f}, {p_world[2]:.2f})"
        cv2.putText(rgb_cv, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb_cv, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Detected Object (RGB)", rgb_cv)

        depth_display = (np.clip(depth_image, 0.0, 1.0) * 255).astype(np.uint8)
        depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        cv2.circle(depth_display_color, (cx_pixel, cy_pixel), 5, (0, 0, 255), -1) 
        depth_text = f"Depth: {metric_depth:.3f}m (Norm: {metric_depth:.3f})"
        cv2.putText(depth_display_color, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(depth_display_color, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Depth Image (Normalized)", depth_display_color)

        print("Displaying images. Press any key in an image window to exit.")
        while True:
            key = cv2.waitKey(100)
            if key != -1:
                break
            if (cv2.getWindowProperty("Detected Object (RGB)", cv2.WND_PROP_VISIBLE) < 1 or
                cv2.getWindowProperty("Depth Image (Normalized)", cv2.WND_PROP_VISIBLE) < 1):
                break

        cv2.destroyAllWindows()
    else:
        print("Detected contour has zero area.")
else:
    print("No red object detected.")
