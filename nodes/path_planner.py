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
from std_srvs.srv import Trigger, SetBool
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
    def __init__(self, pos: Point, start: Point, finish: Point, visited: bool, parent = None) -> None:
        # -- Position Data --
        self.pos = pos
        self.x = int(pos.x)
        self.y = int(pos.y)

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


    def __calculate_distance(self, p0: Point, p1: Point) -> float:
        dx = abs(p1.x - p0.x)
        dy = abs(p1.y - p0.y)
        return (np.sqrt(2)*dy + (dx-dy)) if dy <= dx else (np.sqrt(2)*dx + (dy-dx))


class PathPlanner(Node):

    # ----------------------------------------------
    # ---------- Initialization functions ----------
    # ----------------------------------------------
    def __init__(self) -> None:
        super().__init__(node_name='path_planner')
        
        # -- node settings --
        self.cell_size = 0.2
        self.orientation_method = 'const_change' # 'first_3rd', 'const_change', 'goal', 'start'

        # -- class variables --
        self.last_viewpoints: Viewpoints = None
        self.grid_changed: bool
        self.viewpoints_changed: bool
        self.recalculate_paths: bool
        self.paths_set =  False
        self.state = State.UNSET

        self.start_pose: Pose
        self.paths: list[Path] = []
        
        self.last_grid: OccupancyGrid = None
        self.occupancy_matrix: np.ndarray = None
        self.occupied_value = 50

        self._reset_internals()

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
        self.reduce_K_client = self.create_client(
            SetBool,
            'position_controller/scale_K',
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
        
        # -- Reduce speed 
        if self.reduce_K_client.call(SetBool.Request(data=True)).success:
            self.get_logger().info("-> Reduced speed mode. <-")
        else:
            self.get_logger().info("-> Failed to set speed mode. <-")

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
        
        # -- Reduce speed 
        if self.reduce_K_client.call(SetBool.Request(data=False)).success:
            self.get_logger().info("-> Normal speed mode. <-")
        else:
            self.get_logger().info("-> Failed to set speed mode. <-")
        
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
        if self.state == State.UNSET:
            self.self_do_stop()
        elif self.state == State.MOVE_TO_START:
            if self.recalculate_paths and self.occupancy_matrix is not None:
                self.get_logger().info("---------- Sorting ----------")
                self.paths = self.sorting_algorithm(
                    viewpoint_msg=msg,
                    start_pose=self.start_pose,
                    occupancy_matrix=self.occupancy_matrix
                )
                self.recalculate_paths = False
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
        self.occupancy_matrix = self._occupancy_grid_to_matrix(msg).transpose()
        self.get_logger().info("Received Occupancy grid:")
        self.get_logger().info(f"Resolution={msg.info.resolution}, height={msg.info.height}, width={msg.info.width}.")

    # --------------------------------
    # ---------- Operations ----------
    # --------------------------------
    def self_do_stop(self) -> bool:
        self.state = State.IDLE
        self._reset_internals()
        self.paths = []
        self.paths_set = False
        response: Trigger.Response = self.path_finished_client.call(Trigger.Request())
        if response.success:
            self.get_logger().info('----- BlueRov stopped! -----')
            return True
        else:
            self.get_logger().error('Path follower did NOT stop!')
            return False


    def self_do_move(self, viewpoint_msg: Viewpoints) -> None:
        if not self._viewpoints_changed(viewpoints=viewpoint_msg):# and not self.grid_changed:
            return None
        
        self.viewpoints_changed = True

        if viewpoint_msg.viewpoints[0].completed:
            try:
                path = self.paths.pop(0)
            except IndexError:
                self.get_logger().info('All Viewpoints have been reached. Stopping BlueRov.')
                self.self_do_stop()
                return
            
            self.get_logger().info("---- Setting next path ----")

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
    def sorting_algorithm(self, viewpoint_msg: Viewpoints, start_pose: Pose, occupancy_matrix: np.ndarray) -> list[Path]:

        def traveling_salesman(used_points: list[Point], cost: float, current_point: int, costs: np.ndarray, viewpoints: Viewpoints) -> tuple[list[Point], float]:
            if len(used_points) == costs.shape[0]: 
                return (used_points, cost)
            
            best_cost = np.inf
            best_path: list[Path] = []
            viewpoint: Viewpoint
            for goal, viewpoint in enumerate(viewpoints.viewpoints):
                if viewpoint.pose.position in used_points:
                    continue
                path, total_cost = traveling_salesman(
                    used_points=used_points + [viewpoint.pose.position],
                    cost=cost+costs[current_point, goal],
                    current_point=goal,
                    costs=costs,
                    viewpoints=viewpoints
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path
            
            return (best_path, best_cost)
        
        self.paths_set = True

        self.get_logger().info("----- Findig optimal path -----")

        # -- Calculate all costst and paths in matrix --
        num_viewpoints = len(viewpoint_msg.viewpoints)
        costs = np.ones((num_viewpoints, num_viewpoints), dtype=float) * -1.0
        paths = np.empty((num_viewpoints, num_viewpoints), dtype=Path)
        for i in range(num_viewpoints):
            for j in range(num_viewpoints):
                if costs[i,j] != -1.0: continue
                if i == j:
                    costs[i,j] = 0.0
                    paths[i,j] = None
                    continue
                paths[i,j], costs[i,j] = self.method_a_star(
                    start_pose=viewpoint_msg.viewpoints[i].pose,
                    goal_pose=viewpoint_msg.viewpoints[j].pose,
                    obstacles=np.copy(self.occupancy_matrix)
                )
                """
                costs[j,i] = costs[i,j]
                paths[j,i] = paths[i,j]
                paths[j,i].poses = paths[j,i].poses[::-1]
                """

        # -- Calculate ideal order of points --
        points, total_cost = traveling_salesman(
            used_points=[viewpoint_msg.viewpoints[0].pose.position],
            cost=0.0,
            current_point=0,
            costs=costs,
            viewpoints=viewpoint_msg
        )

        # -- Get ideal path --
        path: list[Path] = []
        viewpoint: Viewpoint
        for start, goal in zip(points[:-1], points[1:]):
            for n, viewpoint in enumerate(viewpoint_msg.viewpoints):
                if viewpoint.pose.position == start:
                    i=n
                if viewpoint.pose.position == goal:
                    j=n
            path.append(paths[i,j])

        self.get_logger().info(f"----- Found optimal path containing {len(paths)} paths ({total_cost:.4f}m) -----")

        return path

    
    def method_a_star(self, start_pose: Pose, goal_pose: Pose, obstacles: np.ndarray, ignore_obstacles: bool = False) -> tuple[Path, float]:
        
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
        
        self.get_logger().info("-- A* is calculating a path --")

        start = self._world_point_to_matrix(start_pose.position, self.cell_size)
        goal = self._world_point_to_matrix(goal_pose.position, self.cell_size)

        map = np.empty(obstacles.shape, dtype=AStarCell)
        map[start.x, start.y] = AStarCell(pos=start, start=start, finish=goal, visited=True, parent=None)
        stack = []
        p: AStarCell = map[start.x, start.y]
        while p.pos != goal:
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
                        map[x, y] = AStarCell(pos=Point(x=x, y=y, z= Z_PLAIN), start=start, finish=goal, visited=False, parent=p)
                        stack = insert_in_stack(stack=stack, new=map[x, y])
                        continue
                    
                    # -- Ignore cells we already visited --
                    if map[x,y].visited: 
                        continue

                    # -- Take the 'better' path --
                    new_pos = AStarCell(pos=Point(x=x, y=y, z=Z_PLAIN), start=start, finish=goal, visited=False, parent=p)
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

        # ---- Return from goal to start ----
        points = [goal]
        while p.pos != start:
            points.append(p.parent.pos)
            p = p.parent

        orientations = self.compute_orientation(
            start=start_pose.orientation,
            goal=goal_pose.orientation,
            len_points=len(points),
        )
        
        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='n_map'
        )
        path = Path(
            header=header,
            poses=[PoseStamped(
                header=header,
                pose=Pose(
                    position=self._matrix_point_to_world(point, self.cell_size),
                    orientation=orientation
                )
            ) for point, orientation in zip(points[::-1], orientations) ]
        )
        
        cost = map[goal.x, goal.y].f_cost * 0.05
        self.get_logger().info(f"-- A* finished: {len(path.poses)} tiles ({cost:.4f}m)")
        return (path, cost)


    def compute_orientation(self, start: Quaternion, goal: Quaternion, len_points: int) -> list[Quaternion]:
        orientations: list[Quaternion] = []
        if self.orientation_method == 'const_change' and len_points > 1:
            # -- Robot will constantly rotate along path --
            _, _, yaw0 = euler_from_quaternion([start.x, start.y, start.z, start.w])
            _, _, yaw1 = euler_from_quaternion([goal.x, goal.y, goal.z, goal.w])
            
            d_yaw, dir = self._find_shortest_angle_and_dir(yaw0, yaw1)
            yaw_per_step = d_yaw / (len_points-1)
            
            for n in range(len_points):
                if dir == 'clockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 + (n*yaw_per_step))
                elif dir == 'anticlockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 - (n*yaw_per_step))
                orientations.append(Quaternion(x=qx, y=qy, z=qz, w=qw))
        elif self.orientation_method == 'first_3rd':
            _, _, yaw0 = euler_from_quaternion([start.x, start.y, start.z, start.w])
            _, _, yaw1 = euler_from_quaternion([goal.x, goal.y, goal.z, goal.w])

            len_turning = round(len_points - ((1/3) * len_points))
            len_const = len_points - len_turning

            d_yaw, dir = self._find_shortest_angle_and_dir(yaw0, yaw1)
            yaw_per_step = d_yaw / (len_turning-1)

            for n in range(len_points):
                if dir == 'clockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 + (n*yaw_per_step))
                elif dir == 'anticlockwise':
                    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw0 - (n*yaw_per_step))
                orientations.append(Quaternion(x=qx, y=qy, z=qz, w=qw))
                self.get_logger().info(f"Orientation = {yaw0 - (n*yaw_per_step)}")
            
            for n in range(len_const):
                orientations.append(goal)
                self.get_logger().info(f"Orientation = {goal.z}")

        elif self.orientation_method == 'start':
            # -- Robot will keep constant starting orientation and turn at the end --
            for n in range(len_points):
                orientations.append(start)
        else:
            # -- Robot wil turn at the start and keep constant goal orientation --
            for n in range(len_points):
                orientations.append(goal)
                self.get_logger().info(f"Orientation = {goal.z}")
        
        # -- make sure the last orientation is the goal orientation --
        orientations[-1] = goal
        return orientations

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
        self.viewpoints_changed = True
        self.recalculate_paths = True

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
        
        self.viewpoints_changed = False
        self.last_viewpoints = viewpoints
        return False

    def _occupancy_grid_to_matrix(self, grid: OccupancyGrid) -> np.ndarray:
        data = np.array(grid.data, dtype=np.uint8)
        data = data.reshape(grid.info.height, grid.info.width)
        return data

    def _world_point_to_matrix(self, p: Point, grid_size: float) -> Point:
        return Point(x=round(p.x/grid_size), y=round(p.y/grid_size), z=Z_PLAIN)

    def _matrix_point_to_world(self, p: Point, grid_size: float) -> Point:
        return Point(x=p.x*grid_size, y=p.y*grid_size, z=Z_PLAIN)


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
