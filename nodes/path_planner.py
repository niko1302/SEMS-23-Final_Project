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


Z_PLAIN = -0.5

class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()


class AStarCell:
    def __init__(self, pos: list[int], start: list[int], finish: list[int], visited: bool, parent = None) -> None:
        # -- Position Data --
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]

        # -- Parent Data --
        self.parent = parent

        # -- Costs -> g: from start to here, h: from here to finish, f: g + h --
        if self.parent == None:
            self.g_cost = self.__calculate_distance(start, pos)
        else:
            self.g_cost = self.parent.g_cost + self.__calculate_distance(parent.pos, pos)
        self.h_cost = self.__calculate_distance(pos, finish)
        self.f_cost = self.g_cost + self.h_cost
        
        # -- Other --
        self.visited = visited


    def __calculate_distance(self, p0: list[int], p1: list[int]) -> float:
        dx = abs(p1[0] - p0[0])
        dy = abs(p1[1] - p0[1])
        return (np.sqrt(2)*dy + (dx-dy)) if dy <= dx else (np.sqrt(2)*dx + (dy-dx))


class PathPlanner(Node):

    # ----------------------------------------------
    # ---------- Initialization functions ----------
    # ----------------------------------------------
    def __init__(self) -> None:
        super().__init__(node_name='path_planner')
        
        # -- node settings --
        self.cell_size = 0.2
        self.orientation_method = 'const_change' # 'goal', 'start'

        # -- class variables --
        self.last_viewpoints: Viewpoints = None
        self.grid_changed: bool
        self.state = State.UNSET
        self._reset_internals()

        self.start_pose: Pose
        
        self.last_grid: OccupancyGrid = None
        self.occupancy_matrix: np.ndarray = None
        self.occupied_value = 50

        # -- call other initialisation functions --
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
            self.srv_move_to_start
        )
        self.start_service = self.create_service(
            Trigger,
            '~/start',
            self.srv_start
        )
        self.stop_service = self.create_service(
            Trigger,
            '~/stop',
            self.srv_stop
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

    # -------------------------------------
    # ---------- on service call ----------
    # -------------------------------------
    def srv_move_to_start(self, request: MoveToStart.Request, response: MoveToStart.Response) -> MoveToStart.Response:
        self.get_logger().info("---- MOVE TO START ----")
        self.get_logger().info(f"Start Position at: {request.target_pose.position.x}, {request.target_pose.position.y}")
        self.state = State.MOVE_TO_START
        self.start_pose = request.target_pose

        # -- just use the goal point as the last and only entry in our path --
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header = header
        path.poses.append(
            PoseStamped(
                header=header,
                pose=request.target_pose
            )
        )
        
        path_response: SetPath.Response = self.set_path_client.call(
            SetPath.Request(path=path))
        if path_response.success:
            self.get_logger().info("Set Path to start!")
            response.success = True
            return response
        else:
            self.get_logger().error("Could not set path to start.")
            response.success = False
            return response


    def srv_start(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.get_logger().info("---- NORMAL OPERATION ----")
        if self.state != State.NORMAL_OPERATION:
            self.get_logger().info('Starting normal operation.')
            self._reset_internals()
        self.state = State.NORMAL_OPERATION
        response.success = True
        return response


    def srv_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.state != State.IDLE:
            self.get_logger().info('Asked to stop. Going to idle mode.')
        response.success = self.do_stop()
        return response
    
    # -----------------------------------------
    # ---------- on subscirber input ----------
    # -----------------------------------------
    def on_viewpoints(self, msg: Viewpoints) -> None:
        self.get_logger().info("Viewpoints recieved!")
        self.get_logger().info(f"msg = {msg.viewpoints[0].pose.position.x}")
        if self.state == State.UNSET:
            self.self_do_stop()
        elif self.state == State.MOVE_TO_START:
            if self.grid_changed:
                # TODO sorting algorithm
                pass
            pass
        elif self.state == State.IDLE:
            pass
        elif self.state == State.NORMAL_OPERATION:
            self.self_do_move(viewpoint_msg=msg)
        else:
            self.get_logger().error('BlueRov is in an unknown state!')
    

    def on_occupancy_grid(self, msg: OccupancyGrid) -> None:
        if not self._grid_changed(grid=msg):
            return None
        
        self.grid_changed = True
        self.cell_size = msg.info.resolution
        # NOTE why transpose ??? doesn't work without
        self.occupancy_matrix = self._occupancy_grid_to_matrix(msg).transpose()
        self.get_logger().info("Received Occupancy grid:")
        self.get_logger().info(f"Resolution={msg.info.resolution}, height={msg.info.height}, width={msg.info.width}.")

    # --------------------------------
    # ---------- Operations ----------
    # --------------------------------
    def self_do_stop(self) -> bool:
        self.state = State.IDLE
        self._reset_internals()
        response: Trigger.Response = self.path_finished_client.call(Trigger.Request())
        if response.success:
            self.get_logger().info('----- BlueRov stopped! -----')
            return True
        else:
            self.get_logger().error('Path follower did NOT stop!')
            return False


    def self_do_move(self, viewpoint_msg: Viewpoints) -> None:
        if not self._viewpoints_changed(viewpoints=viewpoint_msg) and not self.grid_changed:
            return None
        
        self.grid_changed = False
        
        num_viewpoints = len(viewpoint_msg.viewpoints)
        viewpoint: Viewpoint
        last_viewpoint: Viewpoint
        index = 0
        for viewpoint in viewpoint_msg.viewpoints:
            if not viewpoint.completed: break
            last_viewpoint = viewpoint
            index += 1

        # -- All viewpoints have been reached --
        if index == num_viewpoints:
            self.get_logger().info('All Viewpoints have been reached. Stopping BlueRov.')
            self.self_do_stop()
            return None

        # -- First Viewpoint is handled by service move_to_start --
        if index > 0:
            path = self.self_compute_path(
                start=last_viewpoint.pose,
                goal=viewpoint.pose
            )
        else:
            self.get_logger().error("This should have been handled by the move to start service")
            self.self_do_stop()
            return None

        if not path:
            self.get_logger().error("Could not compute path. Giving up...")
            self.self_do_stop()
            return None
        
        response: SetPath.Response = self.set_path_client.call(
            SetPath.Request(path=path))
        if response.success:
            self.get_logger().info("Path has been successfully set.")
        else:
            self.get_logger().error("Could not set path")
            self.self_do_stop()

    # ----------------------------------
    # ---------- Computations ----------
    # ----------------------------------
    def sorting_algorithm(self, viewpoint_msg: Viewpoints, start_pose: Pose) -> list[Path]:
        points: list[Point] = []
        viewpoint: Viewpoint
        # -- Make sure start poition ist first in index
        for viewpoint in viewpoint_msg.viewpoints:
            if viewpoint.pose.position == start_pose.position:
                points.insert(0, viewpoint.pose.position)
            else:
                points.append(viewpoint.pose.position)
        
        # TODO


    def self_compute_path(self, start: Pose, goal: Pose, ignore_obstacles: bool = False) -> Path:
        self.get_logger().info(f"calculating path from ({start.position.x}|{start.position.y}) to ({goal.position.x}|{goal.position.y})")
        p0 = self._world_to_matrix(start.position.x, start.position.y, self.cell_size)
        p1 = self._world_to_matrix(goal.position.x, goal.position.y, self.cell_size)
        
        # -- Check the input data --
        try:
            self.occupancy_matrix[p0[0], p0[1]]
            self.occupancy_matrix[p1[0], p1[1]]
        except IndexError:
            self.get_logger().error(f"Grid shape is {self.occupancy_matrix.shape}.")
            self.get_logger().error(f"Start ({p0[0]}, {p0[1]}) or finish ({p1[0]}, {p1[1]}) is out of bounds!")
            return None

        # ---- Insert Method function here ----
        points_n = self.method_a_star(
            p0=p0,
            p1=p1,
            obstacles=np.copy(self.occupancy_matrix),
            ignore_obstacles=ignore_obstacles
        )

        # -- Method function returns None if failed --
        if not points_n: return None

        # -- Convert discrete points back to world coordinates --
        world_points = [
            self._matrix_to_world(p_n[0], p_n[1], self.cell_size)
            for p_n in points_n
        ]

        # -- How will the robot be orientated along the path --
        num_path_points = len(world_points)
        world_orientation: list[Quaternion] = []
        if self.orientation_method == 'const_change' and num_path_points > 1:
            # -- Robot will constantly rotate along path --
            _, _, yaw0 = euler_from_quaternion([
                start.orientation.x,
                start.orientation.y,
                start.orientation.z,
                start.orientation.w,
            ])
            _, _, yaw1 = euler_from_quaternion([
                goal.orientation.x,
                goal.orientation.y,
                goal.orientation.z,
                goal.orientation.w,
            ])

            d_yaw, dir = self._find_shortest_angle_and_dir(yaw0, yaw1)
            d_yaw_per_step = d_yaw / (num_path_points-1)    # num_path_points must be > 1
            for n in range(num_path_points):
                if dir == 'clockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 + (n*d_yaw_per_step))
                elif dir == 'anticlockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 - (n*d_yaw_per_step))
                world_orientation.append(Quaternion(x=qx, y= qy, z=qz, w=qw))
        elif self.orientation_method == 'start':
            # -- Robot will keep constant starting orientation and turn at the end --
            for n in range(num_path_points):
                world_orientation.append(start.orientation)
        else:
            # -- Robot wil turn at the start and keep constant goal orientation --
            for n in range(num_path_points):
                world_orientation.append(goal.orientation)
        
        # -- make sure the last orientation is the goal orientation --
        world_orientation[-1] = goal.orientation

        # -- Format output --
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        
        path = Path()
        path.header = header
        path.poses = [
            PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(x=p[0], y=p[1], z=Z_PLAIN),
                    orientation=q,
                )
            ) for p, q in zip(world_points, world_orientation) ]

        return path

    
    def method_a_star(self, p0: list[int, int], p1: list[int, int], obstacles: np.ndarray, ignore_obstacles: bool = False) -> list[list[int, int]]:
        
        def insert_in_stack(stack: list[AStarCell], new: AStarCell) -> list[AStarCell]:
            if not stack: return [new]
            i = 0
            for item in stack:
                if new.f_cost > item.f_cost: 
                    i+=1
                    continue
                elif new.f_cost == item.f_cost and new.h_cost > item.h_cost:
                    i+=1
                    continue
                else:
                    break
        
            stack.insert(i, new)
            return stack
        
        def find_and_del(stack: list[AStarCell], to_del: AStarCell) -> list[AStarCell]:
            for i, item in enumerate(stack):
                if item.pos == to_del.pos:
                    del stack[i]
                    return stack
            self.get_logger().error("A* Alg.: Could not find item to delete!")
        
        self.get_logger().info("---- A* is calculating a path ----")

        map = np.empty(obstacles.shape, dtype=AStarCell)
        map[p0[0], p0[1]] = AStarCell(pos=p0, start=p0, finish=p1, visited=True, parent=None)
        stack = []
        p: AStarCell = map[p0[0], p0[1]]
        while p.pos != p1:
            # ---- Check all surrounding cells ----
            for x in range(p.x - 1, p.x + 2):
                for y in range(p.y - 1, p.y + 2):
                    # -- Ignore itself --
                    if x == p.x and y == p.y: 
                        continue
                    
                    # -- Ignore obstacle cells --
                    try:
                        obstacles[x, y] # Enforce to check even if ignore_obstcales
                        value = obstacles[x, y] if not ignore_obstacles else 0
                        if value > self.occupied_value: continue
                    except IndexError:
                        continue

                    # -- Immediately use empty cells --
                    if map[x, y] == None:
                        map[x, y] = AStarCell(pos=[x, y], start=p0, finish=p1, visited=False, parent=p)
                        stack = insert_in_stack(stack=stack, new=map[x, y])
                        continue
                    
                    # -- Ignore cells we already visited --
                    if map[x,y].visited: 
                        continue

                    # -- Take the 'better' path --
                    new_pos = AStarCell(pos=[x,y],start=p0, finish=p1, visited=False, parent=p)
                    if new_pos.f_cost < map[x,y].f_cost:
                        map[x,y] = new_pos
                        stack = find_and_del(stack=stack, to_del=new_pos)
                        stack = insert_in_stack(stack=stack, new=new_pos)
                        continue
            
            # ---- Get next cell ---- 
            try:
                p = stack.pop(0)
            except IndexError:
                self.get_logger().error("A* Algorithm failed!")
                return None
            map[p.x, p.y].visited = True

        # ---- Return frfom goal to start ----
        path = [map[p1[0], p1[1]].pos]
        while p.pos != p0:
            path.append(p.parent.pos)
            p = p.parent
        
        return path[::-1]

    # ---------------------------
    # ---------- Other ----------
    # ---------------------------
    def _find_shortest_angle_and_dir(self, phi0: float, phi1: float) -> tuple[float, str]:
        diff = (phi1 - phi0 + np.pi) % (2*np.pi) - np.pi
        if diff < 0:
            return abs(diff), "anticlockwise"
        else:
            return diff, "clockwise"

    def _reset_internals(self) -> None:
        self.grid_changed = True

    def _grid_changed(self, grid: OccupancyGrid) -> bool:
        # -- If None type OccupancyGrid hasnt been initialized yet --
        if not self.last_grid:
            self.last_grid = grid
            self.get_logger().info(f'Initializing Occupancy grid')
            return True

        # -- Has Occupancy grid data changed? --
        if (grid.info.resolution != self.last_grid.info.resolution or
            grid.info.resolution != self.cell_size):
            self.get_logger().info(f'Cell size changed from {self.cell_size}/{self.last_grid.info.resolution} to {grid.info.resolution}.')
            self.last_grid = grid
            self.cell_size = grid.info.resolution
            return True
        
        if (grid.info.width != self.last_grid.info.width or
            grid.info.height != self.last_grid.info.height or
            grid.info.origin != self.last_grid.info.origin):
            self.get_logger().info(f'Occupancy grid Information changed!')
            self.last_grid = grid
            return True
        
        if grid.data != self.last_grid.data:
            self.get_logger().info(f'Occupancy grid data changed!')
            self.last_grid = grid
            return True
        
        self.last_grid = grid
        return False

    def _viewpoints_changed(self, viewpoints: Viewpoints) -> bool:
        """ Check weather or not the viewpoints have changed.
        The headers of each viewpoint are allowed to change, but the datas are not.

        Args:
            viewpoints (Viewpoints): new Viewpoints

        Returns:
            bool: True: viewpoints have changed, False: have not
        """
        # -- If None type Viewpoints havent been initialized yet --
        if not self.last_viewpoints:
            self.last_viewpoints = viewpoints
            self.get_logger().info(f"Viewpoints have been initialized")
            return True
        
        # -- Has the general structure changed? --
        if len(viewpoints.viewpoints) != len(self.last_viewpoints.viewpoints):
            self.last_viewpoints = viewpoints
            self.get_logger().info(f"Viewpoints have been added or deleted. Recalculating path.")
            return True
        
        # -- Have the viewpoints themselves changed? --
        viewpoint: Viewpoint
        last_viewpoint: Viewpoint
        for n, (viewpoint, last_viewpoint) in enumerate(zip(viewpoints.viewpoints, self.last_viewpoints.viewpoints), 1):
            if viewpoint.completed != last_viewpoint.completed:
                self.last_viewpoints = viewpoints
                self.get_logger().info(f"Viewpoint {n} is now completed. Recalculating path.")
                return True
            if viewpoint.pose.position != last_viewpoint.pose.position:
                self.last_viewpoints = viewpoints
                self.get_logger().info(f"Viewpoint {n}s position changed. Recalculating path.")
                return True
            if viewpoint.pose.orientation != last_viewpoint.pose.orientation:
                self.last_viewpoints = viewpoints
                self.get_logger().info(f"Viewpoint {n}s orientation changed. Recalculating path.")
                return True
        
        self.last_viewpoints = viewpoints
        return False

    def _occupancy_grid_to_matrix(self, grid: OccupancyGrid) -> np.ndarray:
        data = np.array(grid.data, dtype=np.uint8)
        data = data.reshape(grid.info.height, grid.info.width)
        return data

    def _world_to_matrix(self, x: float, y:float, grid_size: float) -> list[int, int]:
        return [round(x/grid_size), round(y/grid_size)]
    
    def _matrix_to_world(self, x: int, y: int, grid_size: float) -> list[float]:
        return [x*grid_size, y*grid_size]
            


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
