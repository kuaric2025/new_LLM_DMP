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

echo "[3/4] Waiting for planning services..."
for i in {1..40}; do
  if "$ROS2_BIN" service list | grep -q "/plan_kinematic_path" && "$ROS2_BIN" service list | grep -q "/compute_cartesian_path"; then
    echo "MoveIt services are up."
    break
  fi
  sleep 1
  if [[ $i -eq 40 ]]; then
    echo "ERROR: MoveIt services did not come up in time."
    echo "Check /tmp/franka_moveit_demo.log"
    exit 1
  fi
done

echo "[4/4] Running Z-bounce demo (type p + Enter to stop loop)..."
"$PYTHON_BIN" "$SCRIPT_DIR/franka_executor.py"

echo "Stopping MoveIt launch..."
kill ${LAUNCH_PID} 2>/dev/null || true
