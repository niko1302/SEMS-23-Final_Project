#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion

PUBLISH_ERROR = True


class YawController(Node):

    def __init__(self):
        super().__init__(node_name='yaw_controller')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # -- Parameter variables --
        self.Kp: float
        self.Kd: float
        self.Ki: float
        self.i_shutoff: float
        self.beta: float
        self.scale_Kp_by: float
        self.scale_Kd_by: float
        self.scale_Ki_by: float

        # -- internal class variables --
        self.Kp_scale = 1.0
        self.Kd_scale = 1.0
        self.Ki_scale = 1.0
        self.i_error = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()

        # default value for the yaw setpoint
        self.setpoint = math.pi / 2.0
        self.setpoint_timed_out = True

        # -- Parameters --
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)

        # -- Subscribers --
        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        self.setpoint_sub = self.create_subscription(
            Float64Stamped,
            topic='~/setpoint',
            callback=self.on_setpoint,
            qos_profile=qos
        )

        # -- Publishers --
        self.torque_pub = self.create_publisher(
            msg_type=ActuatorSetpoint,
            topic='torque_setpoint',
            qos_profile=1
        )
        if PUBLISH_ERROR:
            self.error_pub = self.create_publisher(
                msg_type=Float64Stamped,
                topic='~/error',
                qos_profile=1
            )
            self.filtered_error_pub = self.create_publisher(
                msg_type=Float64Stamped,
                topic='~/filtered_error',
                qos_profile=1
            )
        
        # -- Services --
        self.scale_K_srv = self.create_service(
            SetBool,
            '~/scale_K',
            self.srv_scale_K,
        )

    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('p_gain', rclpy.Parameter.Type.DOUBLE),
                ('d_gain', rclpy.Parameter.Type.DOUBLE),
                ('i_gain', rclpy.Parameter.Type.DOUBLE),
                ('i_shutoff', rclpy.Parameter.Type.DOUBLE),
                ('beta', rclpy.Parameter.Type.DOUBLE),
                ('scale_Kp_by', rclpy.Parameter.Type.DOUBLE),
                ('scale_Kd_by', rclpy.Parameter.Type.DOUBLE),
                ('scale_Ki_by', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        param = self.get_parameter('p_gain')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kp = param.value

        param = self.get_parameter('d_gain')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Kd = param.value

        param = self.get_parameter('i_gain')
        self.get_logger().info(f'{param.name}={param.value}')
        self.Ki = param.value

        param = self.get_parameter('i_shutoff')
        self.get_logger().info(f'{param.name}={param.value}')
        self.i_shutoff = param.value

        param = self.get_parameter('beta')
        self.get_logger().info(f'{param.name}={param.value}')
        self.beta = param.value

        param = self.get_parameter('scale_Kp_by')
        self.get_logger().info(f'{param.name}={param.value}')
        self.scale_Kp_by = param.value

        param = self.get_parameter('scale_Kd_by')
        self.get_logger().info(f'{param.name}={param.value}')
        self.scale_Kd_by = param.value

        param = self.get_parameter('scale_Ki_by')
        self.get_logger().info(f'{param.name}={param.value}')
        self.scale_Ki_by = param.value


    def on_params_changed(self, params) -> SetParametersResult:
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'p_gain':
                self.Kp = param.value
            elif param.name == 'd_gain':
                self.Kd = param.value
            elif param.name == 'i_gain':
                self.Ki = param.value
            elif param.name == 'i_shutoff':
                self.i_shutoff = param.value
            elif param.name == 'beta':
                self.beta = param.value
            elif param.name == 'scale_Kp_by':
                self.scale_Kp_by = param.value
            elif param.name == 'scale_Kd_by':
                self.scale_Kd_by = param.value
            elif param.name == 'scale_Ki_by':
                self.scale_Ki_by = param.value
            else:
                self.get_logger().warning('Did not find parameter!')
        return SetParametersResult(successful= True, reason='Parameter set!')


    def srv_scale_K(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if request.data:
            self.Kp_scale = self.scale_Kp_by
            self.Kd_scale = self.scale_Kd_by
            self.Ki_scale = self.scale_Ki_by
        else:
            self.Kp_scale = 1.0
            self.Kd_scale = 1.0
            self.Ki_scale = 1.0

        response.success = True
        return response


    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('Setpoint timed out. Waiting for new setpoints')
        self.setpoint_timed_out = True

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def on_setpoint(self, msg: Float64Stamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = self.wrap_pi(msg.data)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert the quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        #yaw = self.wrap_pi(yaw)

        control_output = self.compute_control_output(yaw)
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.publish_control_output(control_output, timestamp)

    def compute_control_output(self, yaw):
        # very important: normalize the angle error!
        error = self.wrap_pi(self.setpoint - yaw)

        if PUBLISH_ERROR:
            msg = Float64Stamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.data = error
            self.error_pub.publish(msg)

        error = (self.beta*error) + ((1-self.beta) * self.last_error)

        if PUBLISH_ERROR:
            msg = Float64Stamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.data = error
            self.filtered_error_pub.publish(msg)

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        try:
            d_error = (error - self.last_error) / dt
        except ZeroDivisionError:
            self.get_logger().warning("dt is zero!")
            d_error = 0.0

        self.last_error = error
        self.last_time = now

        if dt < 0.1 and error < self.i_shutoff:
            self.i_error += ((self.last_error + error) / 2) * dt
        else:
            self.i_error = 0

        yaw_thrust = (
            (self.Kp*self.Kp_scale) * error) + (
            (self.Kd*self.Kd_scale) * d_error) + (
            (self.Ki*self.Ki_scale) * self.i_error)

        return yaw_thrust

    def publish_control_output(self, control_output: float, timestamp: rclpy.time.Time):
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.ignore_x = True
        msg.ignore_y = True
        msg.ignore_z = False  # yaw is the rotation around the vehicle's z axis

        msg.z = control_output
        self.torque_pub.publish(msg)


def main():
    rclpy.init()
    node = YawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
