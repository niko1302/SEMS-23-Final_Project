import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatch
import timeit

Z_PLAIN = -0.5

class Point():
    def __init__(self, x: int = 0, y: int = 0, z: int = 0) -> None:
        self.x = x
        self.y = y
        self.z = z


MAP = np.array([
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 100, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 100, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 100, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 100, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 100, 100, 100, 000, 000, 000],
    [000, 000, 100, 100, 100 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 100, 100, 100 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 100, 100, 100 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 100, 100, 100 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000, 000, 000, 000, 000, 000],
])

START = [14, 0]
GOAL = [6, 11]


OCCUPIED_VALUE = 50


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


def method_a_star(p0: list[int], p1: list[int], obstacles: np.ndarray, ignore_obstacles: bool = False) -> list[list[int]]:
        
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
            print("ERROR: Could not find item to delete!")
        
        def print_stack(stack: list[AStarCell]) -> None:
            print("--- Print Stack ---")
            for item in stack:
                print(f"\t{item.x}, {item.y}: f = {item.f_cost}, h = {item.h_cost}, g = {item.g_cost}")
            print("-------------------")

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
                        if value > OCCUPIED_VALUE: continue
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
            p = stack.pop(0)
            map[p.x, p.y].visited = True

        # ---- Reverse path ----
        path = [map[p1[0], p1[1]].pos]
        while p.pos != p0:
            path.append(p.parent.pos)
            p = p.parent
        
        return path[::-1]


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x0 != x1:
            if (x0, y0) != (x1, y1):
                yield x0, y0
            err -= dy
            if err < 0:
                y0 += sy
                err += dx
            x0 += sx
    else:
        err = dy / 2.0
        while y0 != y1:
            if (x0, y0) != (x1, y1):
                yield x0, y0
            err -= dx
            if err < 0:
                x0 += sx
                err += dy
            y0 += sy
    yield x1, y1

def compute_discrete_line(p0: Point, p1: Point):
    
    dx = abs(p1.x - p0.x)
    sx = 1 if p0.x < p1.x else -1
    dy = -abs(p1.y - p0.y)
    sy = 1 if p0.y < p1.y else -1
    error = dx + dy

    x = p0.x
    y = p0.y
    points: list[Point] = []
    while True:
        points.append(Point(x=int(x), y=int(y)))
        if x == p1.x and y == p1.y:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            if x == p1.x:
                break
            error += dy
            x += sx
        if doubled_error <= dx:
            if y == p1.y:
                break
            error += dx
            y += +sy
    return points


def obstacle_on_path(p0: Point, p1: Point, obstacles: np.ndarray) -> list[Point]:
    dx = abs(p1.x - p0.x)
    sx = 1 if p0.x < p1.x else -1
    dy = -abs(p1.y - p0.y)
    sy = 1 if p0.y < p1.y else -1
    error = dx + dy

    x = p0.x
    y = p0.y
    points: list[Point] = []
    while True:
        if obstacles[int(x), int(y)] > 50:
            return None
        else:
            points.append(Point(x=int(x), y=int(y)))
        
        if x == p1.x and y == p1.y:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            if x == p1.x:
                break
            error += dy
            x += sx
        if doubled_error <= dx:
            if y == p1.y:
                break
            error += dx
            y += +sy
    return points


def niko_star(start: Point, goal: Point, obstacles: np.ndarray):
    new_map_1: np.ndarray = np.copy(obstacles)
    cells: list[Point] = []
    
    # -- All cells around obstacles --
    for x in range(obstacles.shape[0]): 
        for y in range(obstacles.shape[1]):
            try:
                obstacles[x+1, y+1]
                obstacles[x-1, y-1]
            except IndexError:
                continue

            visited_free = False
            visited_occpied = False
            for x_near in range(x-1, x + 2):
                for y_near in range(y-1, y+2):
                    if x_near == x and y_near == y: continue

                    if obstacles[x_near, y_near] > 50: visited_occpied = True
                    if obstacles[x_near, y_near] <= 50: visited_free = True

                    if visited_free and visited_occpied:
                        if obstacles[x, y] <= 50:
                            new_map_1[x, y] = 999
                            cells.append(Point(x=x, y=y))
                        break
                else:
                    continue
                break
    
    print(new_map_1)

    # -- only at corners --
    new_map_2 = np.copy(new_map_1)    
    for x in range(new_map_1.shape[0]):
        for y in range(new_map_1.shape[1]):
            try:
                new_map_1[x+1, y+1]
                new_map_1[x-1, y-1]
            except IndexError:
                continue

            if new_map_1[x-1, y] == 999 and new_map_1[x, y] == 999 and new_map_1[x+1, y] == 999:
                new_map_2[x, y] = 0
                cells = [cell for cell in cells if (cell.x != x and cell.y != y)]
            if new_map_1[x, y-1] == 999 and new_map_1[x, y] == 999 and new_map_1[x, y+1] == 999:
                new_map_2[x, y] = 0
                cells = [cell for cell in cells if (cell.x != x and cell.y != y)]
    
    print(new_map_2)
    
    # -- in cells list --
    cells = []
    for x in range(new_map_1.shape[0]):
        for y in range(new_map_1.shape[1]):
            try:
                new_map_1[x+1, y+1]
                new_map_1[x-1, y-1]
            except IndexError:
                continue

            if new_map_2[x, y] == 999: cells.append(Point(x=x, y=y))
    
    for cell in cells:
        print(f"({cell.x}, {cell.y})")
    
    print("\npath:\n")

    path = compute_discrete_line(p0=start, p1=Point(x=8, y=1))
    for cell in path:
        new_map_2[cell.x, cell.y] = 555
        print(f"({cell.x}, {cell.y})")
    
    print(new_map_2)


    print(obstacle_on_path(p0=start, p1=Point(x=8, y=1), obstacles=np.copy(obstacles)))
    print(obstacle_on_path(p0=start, p1=Point(x=8, y=2), obstacles=np.copy(obstacles)))
    print(obstacle_on_path(p0=start, p1=Point(x=8, y=3), obstacles=np.copy(obstacles)))
    path = [start]
    while True:
        if not obstacle_on_path(p0=path[-1], p1=goal, obstacles=np.copy(obstacles)):
            break




def calc_cost(p0: Point, p1: Point) -> float:
    return np.sqrt(np.power(p1.x - p0.x, 2) + np.power(p1.y - p1.x, 2))

class NikoStarCell:
    def __init__(self, pos: Point, start: Point, goal: Point, visited: bool, parent: object = None) -> None:
        self.pos = pos
        self.x = int(pos.x)
        self.y = int(pos.y)

        self.parent = parent

        if self.parent is None:
            self.g_cost = self.__calc_cost(p0=start, p1=pos)
        else:
            self.g_cost = self.parent.g_cost + self.__calc_cost(p0=parent.pos, p1=pos)
        self.h_cost = self.__calc_cost(p0=pos, p1=goal)
        self.f_cost = self.g_cost + self.h_cost

        self.visited = visited
        
    
    def __calc_cost(self, p0: Point, p1: Point) -> float:
        return np.sqrt(np.power(p1.x - p0.x, 2) + np.power(p1.y - p0.y, 2))
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y}): {self.g_cost:.4f} + {self.h_cost:.4f} = {self.f_cost:.4f}"
        



def method_niko_star(start_pose: Point, goal_pose: Point, obstacles: np.ndarray, ignore_obstacles: bool = False) -> tuple[list[Point], float]:
    
    def insert_in_stack(stack: list[NikoStarCell], new: NikoStarCell) -> list[NikoStarCell]:
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
        
    def find_and_del(stack: list[NikoStarCell], to_del: NikoStarCell) -> list[NikoStarCell]:
        for i, item in enumerate(stack):
            if item.x == to_del.x and item.y == to_del.y:
                del stack[i]
                return stack
        print("A* Alg.: Could not find item to delete!")
    
    def get_cells(obstacles: np.ndarray) -> list[Point]:
        # -- Get all cells that are around a obstacle
        around_obstacles: np.ndarray = np.zeros((obstacles.shape))
        for x in range(around_obstacles.shape[1]):
            for y in reversed(range(around_obstacles.shape[0])):
                try:
                    obstacles[x+1, y+1]
                    obstacles[x-1, y-1]
                except IndexError:
                    continue

                visited_free = False
                visited_occpied = False
                for x_neighb in range(x-1, x + 2):
                    for y_neighb in range(y-1, y+2):
                        if x_neighb == x and y_neighb == y: continue

                        if obstacles[x_neighb, y_neighb] > 50: visited_occpied = True
                        if obstacles[x_neighb, y_neighb] <= 50: visited_free = True

                        if visited_free and visited_occpied:
                            if obstacles[x, y] <= 50:
                                around_obstacles[x, y] = -1.0
                            break
                    else:
                        continue
                    break
        
        # -- Only keep corner cells --
        corners = np.zeros(obstacles.shape)
        cells: list[Point] = []
        for x in range(around_obstacles.shape[1]):
            for y in reversed(range(around_obstacles.shape[0])):
                try:
                    obstacles[x+1, y+1]
                    obstacles[x-1, y-1]
                except IndexError:
                    continue
                
                if not(around_obstacles[x-1, y] == -1.0 and around_obstacles[x, y] == -1.0 and around_obstacles[x+1, y] == -1.0) and not(around_obstacles[x, y-1] == -1.0 and around_obstacles[x, y] == -1.0 and around_obstacles[x, y+1] == -1.0):
                    if around_obstacles[x, y] == -1.0:
                        cells.append(Point(x=x, y=y))
                        corners[x, y] = -1.0
        
        print(corners)
        return cells

    # start: to matrix
    # goal: to matrix
    start = start_pose
    goal = goal_pose

    map = np.empty(obstacles.shape, dtype=NikoStarCell)
    map[start.x, start.y] = NikoStarCell(pos=start, start=start, goal=goal, visited=True, parent=None)
    stack: list[NikoStarCell] = []
    cells = get_cells(np.copy(obstacles))
    cells.insert(0, goal)
    p: NikoStarCell = map[start.x, start.y]
    while p.x != goal.x or p.y != goal.y:
        
        for cell in cells:
            if cell.x == p.x and cell.y == p.y: 
                continue

            if obstacle_on_path(p0=p, p1=cell, obstacles=np.copy(obstacles)) is None:
                continue

            if map[cell.x, cell.y] is None:
                map[cell.x, cell.y] = NikoStarCell(
                    pos=Point(x=cell.x, y=cell.y, z=Z_PLAIN),
                    start=start,
                    goal=goal,
                    visited=False,
                    parent=p
                )
                stack = insert_in_stack(stack=stack, new=map[cell.x, cell.y])
                continue

            if map[cell.x, cell.y].visited:
                continue

            new_pos = NikoStarCell(
                pos=Point(x=cell.x, y=cell.y, z=Z_PLAIN),
                start=start,
                goal=goal,
                visited=False,
                parent=p
            )
            if new_pos.f_cost < map[cell.x, cell.y].f_cost:
                map[cell.x, cell.y] = new_pos
                stack = find_and_del(stack=stack, to_del=new_pos)
                stack = insert_in_stack(stack=stack, new=new_pos)
                continue
        
        # -- Get next cell --
        print("---- Stack ----")
        for point in stack:
            print(point)  
        
        try:
            p = stack.pop(0)
        except IndexError:
            print("Niko* failed")
            return ([], 0.0)
        map[p.x, p.y].visited = True
    
    points = [goal]
    while p.x != start.x and p.y != start.y:
        points.append(p.parent.pos)
        p = p.parent
    
    return (points[::-1], map[goal.x, goal.y].f_cost)



def new_niko_star(start: Point, goal: Point, obstacles: np.ndarray):
    
    def find_shortest_path(p0: Point, goal: Point, cost: float, path: list[Point], points: list[Point]) -> tuple[list[Point], float]:

        if p0.x == goal.x and p0.y == goal.y:
            return (path, cost)

        path_to_goal = obstacle_on_path(p0=p0, p1=goal)
        if path_to_goal is not None:
            return (path.append(goal), cost + calc_cost(p0=p0, p1=goal))

        best_cost = np.inf
        best_path:list[Point] = []
        for point in points:
            next_path = obstacle_on_path(p0=p0, p1=point)
            if next_path is not None:
                
                # check cost (straight line) and append if lower
                pass

    
    around_obstacles: np.ndarray = np.zeros((obstacles.shape))
    for x in range(around_obstacles.shape[1]):
        for y in reversed(range(around_obstacles.shape[0])):
            try:
                obstacles[x+1, y+1]
                obstacles[x-1, y-1]
            except IndexError:
                continue

            visited_free = False
            visited_occpied = False
            for x_neighb in range(x-1, x + 2):
                for y_neighb in range(y-1, y+2):
                    if x_neighb == x and y_neighb == y: continue

                    if obstacles[x_neighb, y_neighb] > 50: visited_occpied = True
                    if obstacles[x_neighb, y_neighb] <= 50: visited_free = True

                    if visited_free and visited_occpied:
                        if obstacles[x, y] <= 50:
                            around_obstacles[x, y] = -1.0
                        break
                else:
                    continue
                break

    corners = np.zeros(obstacles.shape)
    cells: list[Point] = []
    for x in range(around_obstacles.shape[1]):
        for y in reversed(range(around_obstacles.shape[0])):
            try:
                obstacles[x+1, y+1]
                obstacles[x-1, y-1]
            except IndexError:
                continue
            
            if not(around_obstacles[x-1, y] == -1.0 and around_obstacles[x, y] == -1.0 and around_obstacles[x+1, y] == -1.0) and not(around_obstacles[x, y-1] == -1.0 and around_obstacles[x, y] == -1.0 and around_obstacles[x, y+1] == -1.0):
                if around_obstacles[x, y] == -1.0:
                    cells.append(Point(x=x, y=y))
                    corners[x, y] = -1.0




    print(around_obstacles)
    print(corners)
    for cell in cells:
        print(f"({cell.x}, {cell.y})")

    p = start
    while True:
        if p.x == goal.x and p.y == goal.y:
            break

        for cell in cells:
            pass


def draw_path(path: list[list[int]], map: np.ndarray) -> None:
    str_map = np.empty(map.shape, dtype=str)
    str_map[map == 0] = '.'
    str_map[map == 100] = '#'
    
    for pos in path:
        x = pos[0]
        y = pos[1]
        str_map[x,y] = '+'

    x = START[0]
    y = START[1]
    str_map[x, y] = 'S'

    x = GOAL[0]
    y = GOAL[1]
    str_map[x, y] = 'E'

    print(str_map)         


def main():
    path_no_obstacles = method_a_star(
        p0=START,
        p1=GOAL,
        obstacles=MAP,
        ignore_obstacles=True,
    )
    path = method_a_star(
        p0=START,
        p1=GOAL,
        obstacles=MAP,
    )

    draw_path(path=path_no_obstacles, map=np.copy(MAP))
    print('\n\n')
    draw_path(path=path, map=np.copy(MAP))


def find_shortest_path_and_direction(angle1, angle2):
    diff = (angle2 - angle1 + 180) % 360 - 180
    if diff < 0:
        return abs(diff), "anticlockwise"
    else:
        return diff, "clockwise"


def weired_stuff():
    print(find_shortest_path_and_direction(180, (270+45)))
    print(find_shortest_path_and_direction(270+45, 0))


    start = 2
    stop = 10
    arr = [2,4,2,4,6,8,10]
    lenght = len(arr)
    d = stop - start
    print(d)
    print(lenght)
    for i in range(lenght):
        print(start + (i*(d/(lenght-1))))



if __name__ == "__main__":
    path, cost = method_niko_star(
        start_pose=Point(x=START[0], y=START[1]),
        goal_pose=Point(x=GOAL[0], y=GOAL[1]),
        obstacles=np.copy(MAP),
    )
    print(cost)
    map = np.copy(MAP)
    for point in path:
        map[point.x, point.y] = 999
    
    print(map)


    """
    niko_star(
        start=Point(x=START[0], y=START[1]),
        goal=Point(x=GOAL[0], y=GOAL[1]),
        obstacles=np.copy(MAP)
    )"""