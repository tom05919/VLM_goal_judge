import rclpy
from isaacsim_controller import IsaacSimPublisher, clip_angle
import numpy as np
import time
from PIL import Image

DT = 1 / 3
HEADING_GAIN = 1.0
RAW_ANGULAR_LIMIT = 1.0
MAXW = 0.45
FX = 272.5
DEADBAND_PX = 5
PUBLISH_INTERVAL = 0.1

# intake SAM2 image
# figure out midpoint of the target
# calulate off set from the midpoint of the image and calculate the angle to center the target
# publish the angle to the robot

def calculate_offset(segmentation_result, image: Image) -> float:
    midpoint_x = image.width / 2

    if not segmentation_result.scores:
        return 0.0
    else :
        best_i = int(np.argmax(segmentation_result.scores))
        mask = segmentation_result.masks[best_i]
        x_coor = np.where(mask)[1]
        target_x = np.mean(x_coor)
        offset_x = midpoint_x - target_x

    return offset_x

def turn_angle(offset: float) -> float:
    if abs(offset) < DEADBAND_PX:
        return 0.0

    theta = np.arctan2(offset, FX)
    angular_vel_value = HEADING_GAIN * clip_angle(theta) / DT
    angular_vel_value = np.clip(
        angular_vel_value,
        -RAW_ANGULAR_LIMIT,
        RAW_ANGULAR_LIMIT,
    )

    if abs(angular_vel_value) <= MAXW:
        angular_vel_value_limit = angular_vel_value
    else:
        angular_vel_value_limit = MAXW * np.sign(angular_vel_value)

    return float(angular_vel_value_limit)


def center_target(
    offset: float,
    sim: bool = False,
    cmd_vel_topic: str | None = None,
) -> None:
    angular_vel = turn_angle(offset)
    if angular_vel == 0.0:
        return

    theta = np.arctan2(offset, FX)
    turn_duration = abs(theta / angular_vel)

    rclpy.init()
    node = IsaacSimPublisher(sim=sim, cmd_vel_topic=cmd_vel_topic)
    try:
        time.sleep(1.0)  # Allow the new ROS publisher to discover its subscriber.
        turn_end = time.monotonic() + turn_duration
        while time.monotonic() < turn_end:
            node.publish_velocity(0.0, angular_vel)
            time.sleep(PUBLISH_INTERVAL)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
