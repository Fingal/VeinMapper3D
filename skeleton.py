import numpy as np
from pyrr import Vector3
import copy
from typing import Tuple,Dict,Set,List,FrozenSet,Optional

Point = Tuple[float,float,float]
Graph = Dict[Point,Set[Point]]
Line = Tuple[Point,Point]
Path = List[Point]

def tsum(*args):
    return tuple(map(sum, zip(*args)))


def tdot(a, b):
    return sum(map(lambda r: r[0] * r[1], zip(a, b)))


def tdif(a, b):
    return sum(map(lambda r: r[0] - r[1], zip(a, b)))


def graph_to_lines(graph :  Graph) -> List[Line]:
    result : List[Line]= []
    for key, items in graph.items():
        for item in items:
            if (key, item) not in result and (item, key) not in result:
                result.append((key, item))
    return result


def move_point(skeleton_graph : Graph, original : Point, new : Point):
    skeleton_graph[new]=skeleton_graph[original]
    del skeleton_graph[original]
    for k,v  in skeleton_graph.items():
        if original in v:
            v.remove(original)
            v.add(new)

def find_path_between_nodes(skeleton_graph : Graph, start : Point, end : Point, find_only_primary : bool=False):
    #visited : Set[Point]= {start} | skeleton_graph[start]
    visited : Set[FrozenSet[Point]]= set(frozenset({start, node}) for node in skeleton_graph[start])
    paths : List[List[List[Point]]]= [[[start, node]] for node in skeleton_graph[start]]
    working : bool = True
    while working:
        working = False
        found : Optional[int]= None
        new_paths : List[List[List[Point]]]= []
        for i, path in enumerate(paths):
            new_path : List[List[Point]] = []
            for p in path:
                if end in p:
                    if find_only_primary:
                        return p
                    new_path.append(p)
                    continue
                neighbours = {
                    node
                    for node in skeleton_graph[p[-1]]
                    if frozenset({p[-1], node}) not in visited
                }
                if neighbours:
                    working = True
                else:
                    new_path.append(p)
                for node in neighbours:
                    visited.add(frozenset({p[-1], node}))
                    new_path.append(p + [node])
                if end in neighbours:
                    found = i
            new_paths.append(new_path)
        if found == None:
            paths = new_paths
        else:
            paths = [new_paths[found]]
            found = None
    # print(paths)
    if not paths or not [path for path in paths[0] if end in path]:
        return []
    return paths[0]


def find_secondary(skeleton_graph : Graph, start : Point, end : Point):
    paths = find_path_between_nodes(skeleton_graph, start, end)
    # print(len(paths))
    primary = [x for x in paths if end in x][0]
    primary_path = set(frozenset({a, b}) for a, b in zip(primary[:-1], primary[1:]))
    result = []
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            if a not in skeleton_graph or b not in skeleton_graph:
                print("aaaabbbb")
            if (
                frozenset({a, b}) not in primary_path
                and (a, b) not in result
                and (b, a) not in result
            ):
                result.append((a, b))
    return result


def find_primary(skeleton_graph : Graph, start : Point, end : Point) -> List[Line]:
    primary = shortest_path(skeleton_graph, start, end)
    primary_path = [(a, b) for a, b in zip(primary[:-1], primary[1:])]
    return primary_path


def find_all(skeleton_graph : Graph, start : Point, end : Point) -> List[List[Line]]:
    paths = find_path_between_nodes(skeleton_graph, start, end)
    result = [] 
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) not in result and (b, a) not in result:
                result.append((a, b))
    return result

def find_compact_component(skeleton_graph : Graph,point : Point) -> Dict[Point,Set[Point]]:
    to_visit=[point]
    visited=set()
    result={}
    while to_visit:
        v=to_visit.pop(0)
        visited.add(v)
        result[v]=copy.copy(skeleton_graph[v])
        for n in skeleton_graph[v]:
            if n not in visited:
                to_visit.append(n)
    return result

def get_all_paths(skeleton_graph : Graph,start : Optional[Point]=None) -> List[Path]:
    all_nodes=list(filter(lambda x: len(skeleton_graph[x])>2 or len(skeleton_graph[x])==1,skeleton_graph.keys()))
    result=[]
    while all_nodes:
        to_visit={all_nodes[0]}
        while to_visit:
            new_to_visit = set()
            for node in to_visit:
                paths=_paths_to_junction(skeleton_graph,node)
                for path in paths:
                    if path[-1] == path[0] and list(reversed(path)) not in result:
                        result.append(path)
                    elif path[-1] in all_nodes:
                        result.append(path)
                        if len(skeleton_graph[path[-1]])>2:
                            new_to_visit.add(path[-1])
                try:
                    all_nodes.remove(node)
                except:
                    pass
            to_visit=new_to_visit
    return result



def remove_from_graph(graph : Graph, to_remove : Point) -> Graph:
    for start, end in to_remove:
        try:
            graph[start].remove(end)
            if not graph[start]:
                del graph[start]
            graph[end].remove(start)
            if not graph[end]:
                del graph[end]
        except:
            print(f"{(start,end)} doesn't exist")
    return graph


def shortest_path(graph, start, end):
    return find_path_between_nodes(graph, start, end, find_only_primary=True)


def shortest_path_length(graph, start, end):
    points = shortest_path(graph, start, end)
    if not points:
        return -1

    length = 0
    for a, b in zip(points[:-1], points[1:]):
        length += sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    return length


def paths_to_junction(graph, point):
    paths = [[point, i] for i in graph[point]]
    visited = {frozenset({point, i}) for i in graph[point]}
    working = True
    while working:
        working = False
        for path in paths:
            point = path[-1]
            neighbours = [
                i for i in graph[point] if frozenset({point, i}) not in visited
            ]
            if len(neighbours) == 1:
                working = True
                neighbour = neighbours[0]
                path.append(neighbour)
                visited.add(frozenset({point, neighbour}))
            else:
                continue
    result = []
    for path in paths:
        for p, p2 in zip(path[:-1], path[1:]):
            result.append((p, p2))

    return result

def _paths_to_junction(graph : Graph, starting_point : Point) -> List[List[Point]]:
    paths = [[starting_point, i] for i in graph[starting_point]]
    visited = {frozenset({starting_point, i}) for i in graph[starting_point]}
    working = True
    while working:
        working = False
        for path in paths:
            point = path[-1]
            neighbours = [
                i for i in graph[point] if i != path[-2]
            ]
            if len(neighbours) == 1 and point!=starting_point:
                working = True
                neighbour = neighbours[0]
                path.append(neighbour)
            else:
                continue

    return paths


def add_cylinder(
    arr,
    points,
    currents=[],
    size=0,
    direction=None,
    min_size=3,
    max_size=10,
    dont_visit=[],
):
    cylinders = []
    old_len = -1
    xxx = 0
    while points:
        # print(len(points))
        # print(len(currents))
        if len(points) == old_len:
            xxx += 1
        else:
            xxx = 0
        if xxx > 50:
            import pdb

            pdb.set_trace()
        old_len = len(points)
        if not currents:
            # start,current,size,direction,dont_visit
            currents.append((points[0], []))
        # TODO turn to BFS
        # import pdb; pdb.set_trace()
        new_currents = []
        for current, last_visited in currents:
            neighbours = [
                i
                for i in get_neighbours(arr, current)
                if i in points and i not in last_visited
            ]
            last_visited = neighbours
            for neighbour in neighbours:
                cylinders.append((Vector3(current), Vector3(neighbour)))
                new_currents.append((neighbour, last_visited))

            try:
                points.remove(current)
            except:
                pass

        currents = new_currents
    return cylinders


def get_nonzero_coordinates(x):
    return list(map(lambda x: tuple(x), zip(*np.nonzero(x))))


def remove_from_array(l, test_array):
    for index in range(len(l)):
        if np.array_equal(l[index], test_array):
            l.pop(index)
            break


def step_node(graph, skeleton, nodes, node):
    if node not in graph:
        graph[node] = []
    x, y, z = node
    for neighbour in get_nonzero_coordinates(
        skeleton[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
    ):
        neighbour = tuple(np.array(neighbour) + np.array(node))
        visited = False
        try:
            nodes.remove(neighbour)
        except:
            visited = True
        graph[node].append(neighbour)
        if not visited:
            step_node(graph, skeleton, nodes, neighbour)


def gen_graph(skeleton):
    nodes = get_nonzero_coordinates(skeleton)
    graph = {}
    while nodes:
        node = nodes.pop(0)
        step_node(graph, skeleton, nodes, node)
    return graph


def get_neighbours(skeleton, node):
    x, y, z = node
    center = [1, 1, 1]
    if x == 0:
        center[0] = 0
    if y == 0:
        center[1] = 0
    if z == 0:
        center[1] = 0
    center = tuple(center)
    result = []
    for i in get_nonzero_coordinates(
        skeleton[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2, max(0, z - 1) : z + 2]
    ):
        coor = tuple(np.array(i) - np.array(center))
        # if coor !=(0,0,0) and abs(coor[0])+abs(coor[1])+abs(coor[2])!=3:
        if coor != (0, 0, 0):
            result.append(tuple(np.array(coor) + np.array(node)))
    return result
    # return [tuple(np.array(i)+np.array(node)-np.array(center)) for i in get_nonzero_coordinates(skeleton[max(0,x-1):x+2,max(0,y-1):y+2,max(0,z-1):z+2]) if tuple(i) != tuple(center) and abs(i[0])+abs(i[1])+abs(i[2])!=3]


def count_length(skeleton, visited, current, min_length, to_remove=[], length=0):
    if min_length <= 0:
        return [], visited
    neighbours = [x for x in get_neighbours(skeleton, current) if x not in visited]
    visited.add(current)
    # print(visited)
    # print("neighbours size {}".format(len(neighbours)))
    if len(neighbours) == 0:
        to_remove.append(current)
        return to_remove, visited
    if len(neighbours) == 1:
        to_remove.append(current)
        return count_length(skeleton, visited, neighbours[0], min_length - 1, to_remove)
    if len(neighbours) > 1:
        return to_remove, visited


def prune_nodes(skeleton, min_length):
    nodes = get_nonzero_coordinates(skeleton)
    to_remove = []
    while nodes:
        node = nodes.pop(0)
        x, y, z = node
        # print(node)
        neighbours = get_neighbours(skeleton, node)
        # print(skeleton[max(0,x-1):x+2,max(0,y-1):y+2,max(0,z-1):z+2])
        # print(neighbours)
        # if np.sum(skeleton[max(0,x-1):x+2,max(0,y-1):y+2,max(0,z-1):z+2])-1!=len(neighbours):
        #     import pdb; pdb.set_trace()
        if len(neighbours) == 0:
            to_remove.append(node)
        if len(neighbours) == 1:
            neighbour = neighbours[0]
            r, visited = count_length(skeleton, {node}, neighbour, min_length, [node])
            to_remove = to_remove + r
            for point in visited:
                try:
                    node.remove(point)
                except:
                    pass
    for point in to_remove:
        try:
            # print(point)
            skeleton[point] = 0
        except:
            print("error", point)


def closestDistanceBetweenLines(
    a0,
    a1,
    b0,
    b1,
    clampAll=False,
    clampA0=False,
    clampA1=False,
    clampB0=False,
    clampB1=False,
):

    """ Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    """

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = b0 - a0
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)
