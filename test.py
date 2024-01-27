import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatch
import timeit

MAP = np.array([
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000],
    [000, 000, 000, 000, 100 ,100, 100, 100, 100, 000],
    [000, 000, 100, 000, 000 ,000, 000, 000, 100, 000],
    [000, 000, 100, 100, 100 ,100, 100, 000, 100, 000],
    [000, 000, 000, 000, 000 ,000, 100, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 100, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000],
    [000, 000, 000, 000, 000 ,000, 000, 000, 000, 000],
])

START = [6, 3]
GOAL = [2, 3]

START = [9, 9]
GOAL = [0, 0]

START = [3, 4]
GOAL = [0, 9]

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
    start = timeit.default_timer()
    path = method_a_star(
        p0=START,
        p1=GOAL,
        obstacles=MAP,
    )
    end = timeit.default_timer()

    draw_path(path=path_no_obstacles, map=np.copy(MAP))
    print('\n\n')
    draw_path(path=path, map=np.copy(MAP))

    print(end-start)


def find_shortest_path_and_direction(angle1, angle2):
    diff = (angle2 - angle1 + 180) % 360 - 180
    if diff < 0:
        return abs(diff), "anticlockwise"
    else:
        return diff, "clockwise"






if __name__ == "__main__":
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


