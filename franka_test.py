import math
from scipy.spatial.transform import Rotation
from franky import (
    Robot, Affine, RobotPose, ElbowState,
    CartesianWaypoint, CartesianWaypointMotion, ReferenceType
)
from pylibfranka import ControllerMode, JointPositions, RealtimeConfig

robot = Robot("192.168.1.11") #RealtimeConfig.kIgnore
robot.recover_from_errors()
# robot.automatic_error_recovery()
robot.relative_dynamics_factor = 0.1

# Move in a small square around the current pose, relative motion
waypoints = [
    CartesianWaypoint(
        RobotPose(Affine([0.10, 0.0, 0.0]), ElbowState(0.0)),
        ReferenceType.Relative
    ),
    CartesianWaypoint(
        RobotPose(Affine([0.0, 0.10, 0.0]), ElbowState(0.0)),
        ReferenceType.Relative
    ),
    CartesianWaypoint(
        RobotPose(Affine([-0.10, 0.0, 0.0]), ElbowState(0.0)),
        ReferenceType.Relative
    ),
]

motion = CartesianWaypointMotion(waypoints)
import pdb; pdb.set_trace()
robot.move(motion)