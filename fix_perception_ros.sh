#!/usr/bin/env bash
# Install ROS 2 Python packages for live stop_judge (--live) in perception env.
# Run once, then: bash fix_perception_numpy.sh  (ROS can change numpy/scipy)
set -eo pipefail

source /root/miniforge3/etc/profile.d/conda.sh
set +u
conda activate perception
set -u

echo "Installing ROS 2 packages into perception env..."
if [ -f /opt/ros/humble/setup.bash ]; then
  echo "System ROS found at /opt/ros/humble (run_robot_stack sources this for --live)."
fi
mamba install -y -c conda-forge -c robostack-humble \
  ros-humble-rclpy ros-humble-sensor-msgs ros-humble-geometry-msgs

echo "Verifying rclpy..."
python -c "import rclpy; from sensor_msgs.msg import Image; from geometry_msgs.msg import Twist; print('rclpy OK')"

echo "Re-pinning numpy/scipy (ROS install may change them)..."
bash "$(dirname "$0")/fix_perception_numpy.sh"
