# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float64MultiArray

# rclpy.init()

# node = Node("traj_test")
# pub = node.create_publisher(Float64MultiArray, "/test_traj_raw", 10)

# msg = Float64MultiArray()
# msg.data = [1.0, 7.0, 0.05, 0.5, 0.0, 0.5, 0.0, 3.14, 1.57, 0.08]  # [n, dof, dt, x,y,z,roll,pitch,yaw,gripper]

# pub.publish(msg)
# print("Published trajectory (Float64MultiArray) OK")

# node.destroy_node()
# rclpy.shutdown()

import numpy as np
import torch

segmap_list = torch.load("task_1.pt")
segmap = np.array(segmap_list['plate'][0].detach().cpu())

import matplotlib.pyplot as plt

plt.imsave('plate_segmap.png', segmap, cmap='gray')