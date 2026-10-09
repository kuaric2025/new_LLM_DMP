#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class TfLookupTester(Node):
    def __init__(self, target_frame: str, source_frame: str):
        super().__init__('tf_lookup_tester')

        self.target_frame = target_frame
        self.source_frame = source_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.lookup_once)

        self.get_logger().info(f'Target frame: {self.target_frame}')
        self.get_logger().info(f'Source frame: {self.source_frame}')
        self.get_logger().info('Waiting for TF...')

    def lookup_once(self):
        try:
            # Latest available transform
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,   # target
                self.source_frame,   # source
                rclpy.time.Time(),   # latest
                timeout=Duration(seconds=2.0)
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            self.get_logger().info('Transform found:')
            self.get_logger().info(
                f'{self.target_frame} <- {self.source_frame}'
            )
            self.get_logger().info(
                f'Translation: x={t.x:.6f}, y={t.y:.6f}, z={t.z:.6f}'
            )
            self.get_logger().info(
                f'Rotation (quat): x={q.x:.6f}, y={q.y:.6f}, z={q.z:.6f}, w={q.w:.6f}'
            )

            # Optional: convert quaternion to 4x4 matrix and print it
            matrix = self.transform_to_matrix(transform)
            self.get_logger().info('4x4 Transform Matrix:')
            for row in matrix:
                self.get_logger().info(
                    f'[{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}, {row[3]: .6f}]'
                )

        except LookupException as e:
            self.get_logger().error(f'LookupException: {str(e)}')
        except ConnectivityException as e:
            self.get_logger().error(f'ConnectivityException: {str(e)}')
        except ExtrapolationException as e:
            self.get_logger().error(f'ExtrapolationException: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {str(e)}')

    @staticmethod
    def transform_to_matrix(transform_stamped):
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        # Quaternion to rotation matrix
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)

        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)

        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)

        return [
            [r00, r01, r02, t.x],
            [r10, r11, r12, t.y],
            [r20, r21, r22, t.z],
            [0.0, 0.0, 0.0, 1.0],
        ]


def main(args=None):
    rclpy.init(args=args)

    # Default frames for your setup
    target_frame = 'camera_color_optical_frame'
    source_frame = 'base'

    # Allow CLI override:
    # python3 tf_lookup_tester.py base camera_color_optical_frame
    if len(sys.argv) >= 3:
        target_frame = sys.argv[1]
        source_frame = sys.argv[2]

    node = TfLookupTester(target_frame, source_frame)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()