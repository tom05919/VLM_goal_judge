import rclpy
from isaacsim_controller import IsaacSimPublisher
import numpy as np
from PIL import Image
from run_grounded_sam2 import SegmentationResult

# intake SAM2 image
# figure out midpoint of the target
# calulate off set from the midpoint of the image and calculate the angle to center the target
# publish the angle to the robot

def calculate_offset(segmentation_result: SegmentationResult, image: Image) -> float:
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

def pd_control(offset: float) -> float:
    EPS = 1e-8
    DT = 1 / 3
    if np.abs(offset) < EPS:
        angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
    else:
        angular_vel_value = np.arctan(dy / dx) / DT

    linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
    angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

    # Velocity limitation
    maxv, maxw = 0.3, 0.3
    if np.abs(linear_vel_value) <= maxv:
        if np.abs(angular_vel_value) <= maxw:
            linear_vel_value_limit = linear_vel_value
            angular_vel_value_limit = angular_vel_value
        else:
            rd = linear_vel_value / angular_vel_value
            linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
            angular_vel_value_limit = maxw * np.sign(angular_vel_value)
    else:
        if np.abs(angular_vel_value) <= 0.001:
            linear_vel_value_limit = maxv * np.sign(linear_vel_value)
            angular_vel_value_limit = 0.0
        else:
            rd = linear_vel_value / angular_vel_value
            if np.abs(rd) >= maxv / maxw:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
            else:
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)

    # Publish a single velocity command derived from the chosen future
    # waypoint via the PD controller above, then re-plan on the next tick.
    if self._estop.is_set() or self._check_external_stop():
        return self.linear, self.angular
    self.node.publish_velocity(linear_vel_value_limit, angular_vel_value_limit)
