#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from tf_transformations import euler_from_quaternion

PUBLISH_ERROR = True
SCALE_KP_BY = 0.6
SCALE_KD_BY = 0.6

class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        self.Kp = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.Kd = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.beta = 0.2

        self.last_time = self.get_clock().now()
        self.last_x_error = 0.0
        self.last_y_error = 0.0
        self.last_z_error = 0.0

        self.Kp_scale = 1.0
        self.Kd_scale = 1.0

        self.setpoint = Point()
        self.setpoint_timed_out = True
        self.pose_counter = 0

        # -- Parameters --
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        # -- Publishers --
        self.thrust_pub = self.create_publisher(
            ActuatorSetpoint,
            'thrust_setpoint',
            1
        )
        if PUBLISH_ERROR:
            self.error_pub = self.create_publisher(
                msg_type=Point,
                topic='~/error',
                qos_profile=1
            )
            self.filtered_error_pub = self.create_publisher(
                msg_type=Point,
                topic='~/filtered_error',
                qos_profile=1
            )
        
        # -- Subscribers --
        self.position_setpoint_sub = self.create_subscription(
            PointStamped,
            '~/setpoint',
            self.on_position_setpoint,
            1
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'vision_pose_cov',
            self.on_pose,
            1
        )

        # -- Services --
        self.reduce_K_srv = self.create_service(
            SetBool,
            '~/scale_K',
            self.srv_accel,
        )
        
        # -- Timers --
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)


    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('p_gain.x', rclpy.Parameter.Type.DOUBLE),
                ('p_gain.y', rclpy.Parameter.Type.DOUBLE),
                ('p_gain.z', rclpy.Parameter.Type.DOUBLE),
                ('d_gain.x', rclpy.Parameter.Type.DOUBLE),
                ('d_gain.y', rclpy.Parameter.Type.DOUBLE),
                ('d_gain.z', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        param = self.get_parameter('p_gain.x')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kp['x'] = param.value

        param = self.get_parameter('p_gain.y')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kp['y'] = param.value

        param = self.get_parameter('p_gain.z')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kp['z'] = param.value

        param = self.get_parameter('d_gain.x')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kd['x'] = param.value

        param = self.get_parameter('d_gain.y')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kd['y'] = param.value

        param = self.get_parameter('d_gain.z')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kd['z'] = param.value


    def srv_accel(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if request.data:
            self.Kp_scale = SCALE_KP_BY
            self.Kd_scale = SCALE_KD_BY
        else:
            self.Kp_scale = 1.0
            self.Kd_scale = 1.0

        response.success = True
        return response


    def on_params_changed(self, params) -> SetParametersResult:
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'p_gain.x':
                self.Kp['x'] = param.value
            elif param.name == 'p_gain.y':
                self.Kp['y'] = param.value
            elif param.name == 'p_gain.z':
                self.Kp['z'] = param.value
            elif param.name == 'd_gain.x':
                self.Kd['x'] = param.value
            elif param.name == 'd_gain.y':
                self.Kd['y'] = param.value
            elif param.name == 'd_gain.z':
                self.Kd['z'] = param.value
            else:
                self.get_logger().warning('Did not find parameter!')
        return SetParametersResult(successful= True, reason='Parameter set!')


    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('setpoint timed out. waiting for new setpoints.')
        self.setpoint_timed_out = True


    def on_position_setpoint(self, msg: PointStamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = msg.point


    def on_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, yaw)


    def apply_control(self, position: Point, yaw: float):
        now = self.get_clock().now()
        x_error = self.setpoint.x - position.x
        y_error = self.setpoint.y - position.y
        z_error = self.setpoint.z - position.z

        if PUBLISH_ERROR:
            error_msg = Point()
            error_msg.x = x_error
            error_msg.y = y_error
            error_msg.z = z_error
            self.error_pub.publish(error_msg)

        x_error = (x_error*self.beta) + ((1.0-self.beta) * self.last_x_error)
        y_error = (y_error*self.beta) + ((1.0-self.beta) * self.last_y_error)
        z_error = (z_error*self.beta) + ((1.0-self.beta) * self.last_z_error)

        dt = (now - self.last_time).nanoseconds * 1e-9
        try:
            dx_error = (x_error - self.last_x_error) / dt
            dy_error = (y_error - self.last_y_error) / dt
            dz_error = (z_error - self.last_z_error) / dt
        except ZeroDivisionError:
            self.get_logger().warning("dt is zero")
            dx_error = 0.0
            dy_error = 0.0
            dz_error = 0.0

        x = (self.Kp['x'] * self.Kp_scale) * x_error + (self.Kd['x'] * self.Kd_scale) * dx_error
        y = (self.Kp['y'] * self.Kp_scale) * y_error + (self.Kd['y'] * self.Kd_scale) * dy_error
        z = (self.Kp['z'] * self.Kp_scale) * z_error + (self.Kd['z'] * self.Kd_scale) * dz_error

        msg = ActuatorSetpoint()
        msg.header.stamp = now.to_msg()
        msg.x = math.cos(-yaw) * x - math.sin(-yaw) * y
        msg.x = min(0.5, max(-0.5, msg.x))
        msg.y = math.sin(-yaw) * x + math.cos(-yaw) * y
        msg.y = min(0.5, max(-0.5, msg.y))
        msg.z = z

        self.last_x_error = x_error
        self.last_y_error = y_error
        self.last_z_error = z_error
        self.last_time = now
        
        if PUBLISH_ERROR:
            filtered_msg = Point()
            filtered_msg.x = x_error
            filtered_msg.y = y_error
            filtered_msg.z = z_error
            self.filtered_error_pub.publish(filtered_msg)


        self.thrust_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
