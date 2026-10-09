#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_WS="${FRANKA_WS:-$HOME/franka_ros2_ws}"
ROS2_BIN="/opt/ros/jazzy/bin/ros2"
PYTHON_BIN="/usr/bin/python3"
ROBOT_IP="${1:-192.168.1.11}"

# # The demo crashes if it inherits a mixed conda/ROS Python overlay.
# unset PYTHONPATH
# unset AMENT_PREFIX_PATH
# unset COLCON_PREFIX_PATH
# unset CMAKE_PREFIX_PATH
# unset PYTHONHOME


set +u
source /opt/ros/jazzy/setup.bash
source "$FRANKA_WS/install/setup.bash"

set -u

echo "[1/4] Cleaning old ROS graph (best effort)..."
pkill -f "z_bounce_demo.py" 2>/dev/null || true
pkill -f "franka_fr3_moveit_config moveit.launch.py" 2>/dev/null || true
sleep 1

echo "[2/4] Starting MoveIt stack for robot ${ROBOT_IP} ..."
"$ROS2_BIN" launch franka_fr3_moveit_config moveit.launch.py robot_ip:=${ROBOT_IP} use_fake_hardware:=false > /tmp/franka_moveit_demo.log 2>&1 &
LAUNCH_PID=$!

echo "MoveIt launch PID: ${LAUNCH_PID}"
echo "Log: /tmp/franka_moveit_demo.log"


echo "[4/4] Running experiment..."
set +u
source /home/mhumais/ros2_py310_ws/install/setup.bash
set -u



"$CONDA_PREFIX/bin/python" - <<'PY'
import sys, rclpy, moveit_msgs, geometry_msgs
from moveit_msgs.srv import GetMotionPlan
print("exe:", sys.executable)
print("ver:", sys.version)
print("rclpy:", rclpy.__file__)
print("moveit_msgs:", moveit_msgs.__file__)
print("geometry_msgs:", geometry_msgs.__file__)
print("request module:", GetMotionPlan.Request.__module__)
print("request type:", GetMotionPlan.Request)
PY

"$CONDA_PREFIX/bin/python" /home/mhumais/Huang/DMP/llm_mani/test_mani_env_w_llm_exp.py --show-camera --subscribe-camera --subscriber-log-interval 5 --prompt "prepare breakfast with all fruits in plate" --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar --top_down_grasp --grasp-center-offset-z -0.115

# python ../../llm_mani/test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5 --prompt "prepare breakfast with all fruits in plate" --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar --top_down_grasp --grasp-center-offset-z -0.105 --num-runs 1  --scene-configs scene_4.json

echo "Stopping MoveIt launch..."
kill ${LAUNCH_PID} 2>/dev/null || true
