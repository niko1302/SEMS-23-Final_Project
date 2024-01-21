#!/usr/bin/env python3

from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scenario_msgs.msg import Viewpoints, Viewpoint
from scenario_msgs.srv import MoveToStart, SetPath
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()
    MOVEING = auto()


def occupancy_grid_to_matrix(grid: OccupancyGrid) -> np.ndarray:
    data = np.array(grid.data, dtype=np.uint8)
    data = data.reshape(grid.info.height, grid.info.width)
    return data


def world_to_matrix(x: float, y: float, grid_size: float) -> list[int]:
    """Converts position to the correspondig grid.

    Args:
        x (float): Real x position
        y (float): Real y position
        grid_size (float): Size of one grid square

    Returns:
        list[int]: x and y index of corresponding grid
    """
    return [round(x / grid_size), round(y / grid_size)]


def matrix_index_to_world(x, y, grid_size):
    return [x * grid_size, y * grid_size]


def multiple_matrix_indeces_to_world(points: list[list[int]], grid_size: float) -> list[list[float]]:
    """Converts discrete points to worl coordinates.

    Args:
        points (list[list[int]]): list of grid index pairs [x & y]
        grid_size (float): size of a single grid

    Returns:
        list[list[float]]: list of coordinate pairs [x & y]
    """
    world_points = []
    for point in points:
        world_points.append([point[0] * grid_size, point[1] * grid_size])
    return world_points


def compute_discrete_line(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """ Computes a discrete straight line between two points.

    Args:
        x0 (int): x position first point
        y0 (int): y position first point
        x1 (int): x position second point
        y1 (int): y position second point

    Returns:
        list[list[int]]: list of grid index pairs (x & y)
    """
    # NOTE ich verstehe diese funktion net
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x = x0
    y = y0
    points = []
    while True:
        points.append([int(x), int(y)])
        if x == x1 and y == y1: break
        doubled_error = 2 * error
        if doubled_error >= dy:
            if x == x1: break
            error += dy
            x += sx
        if doubled_error <= dx:
            if y == y1: break
            error += dx
            y += +sy
    return points


class PathPlanner(Node):

    # ----------------------------------------------
    # ---------- Initialization functions ----------
    # ----------------------------------------------
    def __init__(self) -> None:
        super().__init__(node_name='path_planner')
        
        # -- node settings --
        self.cell_size = 0.2

        # -- class variables --
        self.reset_internals()
        self.path_marker: Marker
        
        self.viewpoints = []
        self.waypoints = []
        self.orientations = []
        self.occupancy_grid: OccupancyGrid = None
        self.occupancy_matrix: np.ndarray = None
        
        self.progress = -1.0
        self.remaining_segments = []

        # -- call other initialisation functions --
        self.init_path_marker()
        self.init_clients()
        self.init_services()
        
        # ---- Create publishers and subscribers ----
        self.grid_map_sub = self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.on_occupancy_grid,
            1
        )
        self.viewpoints_sub = self.create_subscription(
            Viewpoints,
            'viewpoints',   # publishes at 50 Hz
            self.on_viewpoints,
            1
        )


    def init_services(self) -> None:
        """ Initialize services """
        self.move_to_start_service = self.create_service(
            MoveToStart,
            '~/move_to_start',
            self.serve_move_to_start
        )
        self.start_service = self.create_service(
            Trigger,
            '~/start',
            self.serve_start
        )
        self.stop_service = self.create_service(
            Trigger,
            '~/stop',
            self.serve_stop
        )


    def init_clients(self) -> None:
        """ Initialize clients """
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.set_path_client = self.create_client(
            SetPath,
            'path_follower/set_path',
            callback_group=cb_group
        )
        self.path_finished_client = self.create_client(
            Trigger,
            'path_follower/path_finished',
            callback_group=cb_group
        )


    def init_path_marker(self) -> None:
        """ Initializes the path marker """
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = 'path'
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.header.frame_id = 'map'
        msg.color.a = 1.0
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.02
        self.path_marker = msg


    def reset_internals(self) -> None:
        self.target_viewpoint_index = -1
        self.recomputation_required = True
        self.state = State.UNSET

    # -----------------------------------------
    # ---------- on subscirber input ----------
    # -----------------------------------------
    def on_occupancy_grid(self, msg: OccupancyGrid) -> None:
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        if msg.info.resolution != self.cell_size:
            self.get_logger().info('Cell size changed. Recomputation required.')
            self.recomputation_required = True
            self.cell_size = msg.info.resolution


    def on_viewpoints(self, msg: Viewpoints) -> None:
        if self.state == State.IDLE:
            return
        if self.state == State.UNSET:
            if not self.do_stop():
                self.get_logger().error('Failed to stop.')
            return
        if self.state == State.MOVE_TO_START:
            # Nothing to be done here. We already did the setup when the corresponding service was called.
            return
        if self.state == State.NORMAL_OPERATION:
            self.do_normal_operation(msg)

    # -------------------------------------
    # ---------- on service call ----------
    # -------------------------------------
    def serve_move_to_start(self, request: MoveToStart.Request, response: MoveToStart.Response) -> MoveToStart.Response:
        self.state = State.MOVE_TO_START
        self.start_pose = request.target_pose
        self.current_pose = request.current_pose
        # - We do not care for collisions while going to the start position. -
        response.success = self.move_to_start(
            request.current_pose,
            request.target_pose
        )
        return response


    def serve_start(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.state != State.NORMAL_OPERATION:
            self.get_logger().info('Starting normal operation.')
            self.reset_internals()
        self.state = State.NORMAL_OPERATION
        response.success = True
        return response


    def serve_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.state != State.IDLE:
            self.get_logger().info('Asked to stop. Going to idle mode.')
        response.success = self.do_stop()
        return response

    # --------------------------------
    # ---------- Operations ----------
    # --------------------------------
    def do_normal_operation(self, viewpoints: Viewpoints) -> None:
        # what we need to do:
        # - check if the viewpoints changed, if so, recalculate the path
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        
        # we completed our mission!
        if i < 0:
            self.handle_mission_completed()
            return

        if (not self.recomputation_required) or self.target_viewpoint_index == i:
            # we are still chasing the same viewpoint. Nothing to do.
            # TODO if we need to recompute than we keep going ???
            return

        self.get_logger().info('Computing new path segments')
        self.target_viewpoint_index = i
        if i == 0:
            p = viewpoints.viewpoints[0].pose
            if not self.move_to_start(p, p):
                self.get_logger().fatal(
                    'Could not move to first viewpoint. Giving up...')
                if not self.do_stop():
                    # Could not stop robot
                    self.state = State.UNSET
            return

        path_segments = self.compute_new_path(viewpoints)
        if not path_segments:
            self.get_logger().error(
                'This is a logic error. The cases that would have lead to '
                'no valid path_segments should have been handled before')
            return
        if path_segments[0]['collision_indices']:
            self.handle_no_collision_free_path()
            return
        self.set_new_path(path_segments[0]['path'])
        return


    def move_to_start(self, p0: Pose, p1: Pose) -> bool:
        path_segment = self.compute_simple_path_segment(p0, p1, check_collision=False)
        request = SetPath.Request()
        request.path = path_segment['path']
        answer: SetPath.Response = self.set_path_client.call(request)
        if answer.success:
            self.get_logger().info('Moving to start position')
            return True
        else:
            self.get_logger().info(
                'Asked to move to start position. '
                'But the path follower did not accept the new path.')
            return False
        

    def do_stop(self) -> bool:
        """Halts the robot.

        Returns:
            bool: has the path finished or not.
        """
        self.state = State.IDLE
        if self.path_finished_client.call(Trigger.Request()).success:
            self.reset_internals()
            return True
        return False
    
    # ------------------------------
    # ---------- Handlers ----------
    # ------------------------------
    def handle_mission_completed(self) -> None:
        """All viewpoints visited handler"""
        self.get_logger().info('Mission completed.')
        if not self.do_stop():
            self.get_logger().error(
                'All waypoints completed, but could not '
                'stop the path_follower. Trying again...',
                throttle_duration_sec=1.0)
            return
        self.state = State.IDLE


    def handle_no_collision_free_path(self) -> None:
        self.get_logger().fatal('We have a collision in our current segment! Giving up...')
        if not self.do_stop():
            self.state = State.UNSET

    # ----------------------------------
    # ---------- Computations ----------
    # ----------------------------------
    def compute_new_path(self, viewpoints: Viewpoints) -> list[dict[Path, list[list[int]]]]:
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # start position is treated differently.
        if i < 1:
            return []
        # complete them all.
        # We do nothing smart here. We keep the order in which we received
        # the waypoints and connect them by straight lines.
        # No collision avoidance.
        # We only perform a collision detection and give up in case that
        # our path of straight lines collides with anything.
        # Not very clever, eh?

        # now get the remaining uncompleted viewpoints. In general, we can
        # assume that all the following viewpoints are uncompleted, since
        # we complete them in the same order as we get them in the
        # viewpoints message. But it is not that hard to double-check it.
        viewpoint_poses = [
            v.pose for v in viewpoints.viewpoints[i:] if not v.completed
        ]
        # get the most recently visited viewpoint. Since we do not change
        # the order of the viewpoints, this will be the viewpoint right
        # before the first uncompleted viewpoint in the list, i.e. i-1
        p0 = viewpoints.viewpoints[i - 1].pose
        viewpoint_poses.insert(0, p0)

        # now we can finally call our super smart function to compute
        # the path piecewise between the viewpoints
        path_segments = []
        for i in range(1, len(viewpoint_poses)):
            segment = self.compute_simple_path_segment(
                viewpoint_poses[i - 1],
                viewpoint_poses[i]
            )
            # alternatively call your own implementation
            # segment = self.compute_a_star_segment(viewpoint_poses[i - 1],
            #                                       viewpoint_poses[i])
            path_segments.append(segment)
        return path_segments
    

    def compute_simple_path_segment(self, p0: Pose, p1: Pose, check_collision: bool = True) -> dict[Path, list[list[int]]]:
        p0_2d = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        p1_2d = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)
        # now we should/could apply some sophisticated algorithm to compute
        # the path that brings us from p0_2d to p1_2d. For this dummy example
        # we simply go in a straight line. Not very clever, but a straight
        # line is the shortes path between two points, isn't it?
        line_points_2d = compute_discrete_line(p0_2d[0], p0_2d[1], p1_2d[0], p1_2d[1])
        if check_collision:
            collision_indices = self.has_collisions(line_points_2d)
        else:
            collision_indices = []

        # Convert back our matrix/grid_map points to world coordinates. Since
        # the grid_map does not contain information about the z-coordinate,
        # the following list of points only contains the x and y component.
        xy_3d = multiple_matrix_indeces_to_world(line_points_2d, self.cell_size)

        # it might be, that only a single grid point brings us from p0 to p1.
        # in this duplicate this point. this way it is easier to handle.
        if len(xy_3d) == 1: xy_3d.append(xy_3d[0])
        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        q = quaternion_from_euler(0.0, 0.0, yaw0)
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}


    def compute_a_star_segment(self, p0: Pose, p1: Pose):
        # TODO: implement your algorithms
        # you probably need the gridmap: self.occupancy_grid
        pass

    # ---------------------------

    def find_first_uncompleted_viewpoint(self, viewpoints: Viewpoints) -> int:
        """Finds first completed viewpoint the robot has to go to.

        Args:
            viewpoints (Viewpoints): All viewpoints.

        Returns:
            int: Index of first uncompleted viewpoint or -1 if all have been visited.
        """
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if not viewpoint.completed:
                return i
        return -1


    def has_collisions(self, points_2d: list[list[int]]) -> list[list[int]]:
        """ Checks weather a given path collides with obsticles.

        Args:
            points_2d (list): given path as list of positions

        Returns:
            list: position where the path collides
        """
        if not self.occupancy_grid:
            return []
        collision_indices = [
            i for i, p in enumerate(points_2d)
            if self.occupancy_matrix[p[1], p[0]] >= 50  # NOTE if occupied should be 100
        ]
        return collision_indices


    def set_new_path(self, path):
        request = SetPath.Request()
        if not path:
            return False
        request.path = path
        self.set_new_path_future = self.set_path_client.call_async(request)
        return True

    # ---------- SELF ----------
    def self_on_viewpoints(self, msg: Viewpoints) -> None:
        if self.state == State.UNSET or self.state == State.IDLE:
            self.self_do_stop()
        elif self.state == State.MOVEING:
            self.self_do_move(viewpoint_msg=msg)
        else:
            self.get_logger().error('Robot is in an unknown state!')


    def self_do_stop(self) -> None:
        response: Trigger.Response = self.path_finished_client.call(Trigger.Request())
        if not response.success:
            self.get_logger().error('Path follower did NOT stop!')
        self.reset_internals()
        self.get_logger().info('----- BlueRov stopped! -----')


    def self_do_move(self, viewpoint_msg: Viewpoints) -> None:
        num_viewpoints = len(viewpoint_msg.viewpoints)
        viewpoint: Viewpoint
        last_viewpoint = Viewpoint()
        # TODO populate class
        for index, viewpoint in enumerate(viewpoint_msg.viewpoints, 1):
            if not viewpoint.completed: break

        # -- All viewpoints have been reached --
        if index == num_viewpoints:
            self.get_logger().info('All Viewpoints have been reached. Stopping BlueRov.')
            self.self_do_stop()
            return None
        
        # TODO starting point
        if index == 0:
            self.self_compute_path(Viewpoint(), viewpoint, ignore_obstacle=True)
        else:
            self.self_compute_path(Viewpoint(), viewpoint)
        
    def self_compute_path(self, start: Viewpoint, viewpoint: Viewpoint, ignore_obstacle: bool = False) -> None:
        pass
            


def main():
    rclpy.init()
    node = PathPlanner()
    exec = MultiThreadedExecutor()
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
