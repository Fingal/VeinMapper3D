# distutils: language=c++
# cython: language_level=3

from libc.math cimport sqrt,cos,ceil,round
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
cimport cython
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint16

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

from libcpp.set cimport set
from libcpp.queue cimport queue

ctypedef np.uint16_t DTYPE_t

cdef struct Point:
    float x
    float y
    float z 



cdef _RAYS_NUMBER=26
cdef Point[26] _RAYS=[
    Point(1, 0, 0),
    Point(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
    Point(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    Point(0.7071067811865475, -0.7071067811865475, 0.0),
    Point(-0.7071067811865475, 0.7071067811865475, 0.0),
    Point(0.0, 0.7071067811865475, -0.7071067811865475),
    Point(0.7071067811865475, 0.0, -0.7071067811865475),
    Point(0.0, -0.7071067811865475, 0.7071067811865475),
    Point(0, 0, -1),
    Point(-1, 0, 0),
    Point(0, 0, 1),
    Point(-0.7071067811865475, -0.7071067811865475, 0.0),
    Point(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
    Point(0.7071067811865475, 0.0, 0.7071067811865475),
    Point(-0.7071067811865475, 0.0, 0.7071067811865475),
    Point(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    Point(0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
    Point(0, -1, 0),
    Point(0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    Point(0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
    Point(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    Point(-0.7071067811865475, 0.0, -0.7071067811865475),
    Point(0.7071067811865475, 0.7071067811865475, 0.0),
    Point(0.0, -0.7071067811865475, -0.7071067811865475),
    Point(0.0, 0.7071067811865475, 0.7071067811865475),
    Point(0, 1, 0),
]

cdef RAYS_NUMBER=98
cdef Point[98] RAYS=[
Point(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
Point(0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
Point(0.9238795325112867, 0.0, -0.3826834323650897),
Point(0, 0, -1),
Point(0.8880738339771151, 0.32505758367186816, 0.32505758367186816),
Point(-0.3826834323650897, 0.9238795325112867, 0.0),
Point(0.0, -0.3826834323650897, 0.9238795325112867),
Point(0.6845503194452978, -0.6845503194452978, -0.2505628070857316),
Point(-0.6845503194452978, 0.6845503194452978, -0.2505628070857316),
Point(0.2505628070857316, 0.6845503194452978, -0.6845503194452978),
Point(0.6845503194452978, -0.2505628070857316, 0.6845503194452978),
Point(0.2505628070857316, -0.6845503194452978, -0.6845503194452978),
Point(-0.6845503194452978, -0.2505628070857316, 0.6845503194452978),
Point(-0.6845503194452978, -0.2505628070857316, -0.6845503194452978),
Point(0.2505628070857316, -0.6845503194452978, 0.6845503194452978),
Point(0.6845503194452978, 0.6845503194452978, -0.2505628070857316),
Point(-0.6845503194452978, -0.6845503194452978, -0.2505628070857316),
Point(-0.7071067811865475, -0.7071067811865475, 0.0),
Point(-0.7071067811865475, 0.0, -0.7071067811865475),
Point(-0.2505628070857316, 0.6845503194452978, -0.6845503194452978),
Point(0.2505628070857316, 0.6845503194452978, 0.6845503194452978),
Point(0.0, 0.3826834323650897, -0.9238795325112867),
Point(0, 1, 0),
Point(-0.7071067811865475, 0.0, 0.7071067811865475),
Point(0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
Point(0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
Point(0.9238795325112867, 0.0, 0.3826834323650897),
Point(0.9238795325112867, -0.3826834323650897, 0.0),
Point(0.32505758367186816, -0.8880738339771151, 0.32505758367186816),
Point(-0.32505758367186816, -0.8880738339771151, 0.32505758367186816),
Point(-1, 0, 0),
Point(-0.32505758367186816, 0.32505758367186816, -0.8880738339771151),
Point(0, 0, 1),
Point(-0.32505758367186816, -0.8880738339771151, -0.32505758367186816),
Point(-0.9238795325112867, 0.0, 0.3826834323650897),
Point(-0.3826834323650897, 0.0, -0.9238795325112867),
Point(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
Point(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
Point(-0.9238795325112867, 0.3826834323650897, 0.0),
Point(0.3826834323650897, 0.9238795325112867, 0.0),
Point(-0.3826834323650897, -0.9238795325112867, 0.0),
Point(-0.9238795325112867, 0.0, -0.3826834323650897),
Point(0.3826834323650897, -0.9238795325112867, 0.0),
Point(-0.9238795325112867, -0.3826834323650897, 0.0),
Point(0, -1, 0),
Point(0.0, 0.9238795325112867, -0.3826834323650897),
Point(0.3826834323650897, 0.0, -0.9238795325112867),
Point(0.32505758367186816, 0.8880738339771151, -0.32505758367186816),
Point(0.0, -0.3826834323650897, -0.9238795325112867),
Point(-0.8880738339771151, -0.32505758367186816, -0.32505758367186816),
Point(-0.32505758367186816, -0.32505758367186816, 0.8880738339771151),
Point(0.32505758367186816, -0.8880738339771151, -0.32505758367186816),
Point(-0.3826834323650897, 0.0, 0.9238795325112867),
Point(0.9238795325112867, 0.3826834323650897, 0.0),
Point(0.0, -0.9238795325112867, 0.3826834323650897),
Point(0.8880738339771151, 0.32505758367186816, -0.32505758367186816),
Point(-0.6845503194452978, 0.2505628070857316, -0.6845503194452978),
Point(-0.6845503194452978, -0.6845503194452978, 0.2505628070857316),
Point(-0.2505628070857316, 0.6845503194452978, 0.6845503194452978),
Point(-0.2505628070857316, -0.6845503194452978, -0.6845503194452978),
Point(0.6845503194452978, 0.2505628070857316, -0.6845503194452978),
Point(0.6845503194452978, -0.6845503194452978, 0.2505628070857316),
Point(-0.6845503194452978, 0.6845503194452978, 0.2505628070857316),
Point(-0.6845503194452978, 0.2505628070857316, 0.6845503194452978),
Point(0.6845503194452978, 0.6845503194452978, 0.2505628070857316),
Point(0.6845503194452978, 0.2505628070857316, 0.6845503194452978),
Point(0.32505758367186816, -0.32505758367186816, -0.8880738339771151),
Point(-0.8880738339771151, 0.32505758367186816, 0.32505758367186816),
Point(0.32505758367186816, -0.32505758367186816, 0.8880738339771151),
Point(-0.2505628070857316, -0.6845503194452978, 0.6845503194452978),
Point(0.6845503194452978, -0.2505628070857316, -0.6845503194452978),
Point(0.0, -0.9238795325112867, -0.3826834323650897),
Point(0.3826834323650897, 0.0, 0.9238795325112867),
Point(1, 0, 0),
Point(0.32505758367186816, 0.32505758367186816, 0.8880738339771151),
Point(-0.32505758367186816, 0.8880738339771151, -0.32505758367186816),
Point(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
Point(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
Point(0.8880738339771151, -0.32505758367186816, 0.32505758367186816),
Point(0.8880738339771151, -0.32505758367186816, -0.32505758367186816),
Point(-0.8880738339771151, -0.32505758367186816, 0.32505758367186816),
Point(0.7071067811865475, -0.7071067811865475, 0.0),
Point(-0.7071067811865475, 0.7071067811865475, 0.0),
Point(0.0, 0.7071067811865475, -0.7071067811865475),
Point(0.7071067811865475, 0.0, -0.7071067811865475),
Point(0.0, -0.7071067811865475, 0.7071067811865475),
Point(-0.8880738339771151, 0.32505758367186816, -0.32505758367186816),
Point(0.32505758367186816, 0.8880738339771151, 0.32505758367186816),
Point(0.32505758367186816, 0.32505758367186816, -0.8880738339771151),
Point(-0.32505758367186816, -0.32505758367186816, -0.8880738339771151),
Point(0.0, 0.3826834323650897, 0.9238795325112867),
Point(0.7071067811865475, 0.0, 0.7071067811865475),
Point(0.7071067811865475, 0.7071067811865475, 0.0),
Point(0.0, -0.7071067811865475, -0.7071067811865475),
Point(0.0, 0.7071067811865475, 0.7071067811865475),
Point(0.0, 0.9238795325112867, 0.3826834323650897),
Point(-0.32505758367186816, 0.8880738339771151, 0.32505758367186816),
Point(-0.32505758367186816, 0.32505758367186816, 0.8880738339771151),
]
ctypedef (float, float, float) tPoint

cdef float dot(Point a,Point b):
    return a.x*b.x+a.y*b.y+a.z*b.z

cdef Point add(Point a, Point b):
    return Point(a.x+b.x,a.y+b.y,a.z+b.z)

cdef float length(Point a):
    return sqrt(a.x*a.x+a.y*a.y+a.z*a.z)

cdef Point normalize(Point a):
    cdef float l=length(a)
    return Point(a.x/l,a.y/l,a.z/l)

cdef Point mul(float n,Point a):
    return Point(a.x*n,a.y*n,a.z*n)

cdef Point cross(Point a, Point b):
    return Point(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x)

cdef float tdot(tPoint a,tPoint b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

cdef tPoint tadd(tPoint a, tPoint b):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

cdef float tlength(tPoint a):
    return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

cdef tPoint tnormalize(tPoint a):
    cdef float l=tlength(a)
    return (a[0]/l,a[1]/l,a[2]/l)

cdef tPoint tmul(float n,tPoint a):
    return (a[0]*n,a[1]*n,a[2]*n)

cdef tPoint tcross(tPoint a, tPoint b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def tests():
    assert(dot(Point(1,0,0),Point(0.5,1,0))==0.5)
    assert((add(Point(1,0,0),Point(0.5,1,0))).x==1.5)
    assert((add(Point(1,0,0),Point(0.5,1,0))).y==1.0)
    assert((add(Point(1,0,0),Point(0.5,1,0))).z==0.0)
    print((-0.001<(length(Point(3,4,0))-5)))
    print((length(Point(3,4,0))-5))

cdef _calculate_cone(Point direction,float distance, float angle):
    cdef vector[Point] result
    cdef int size = int(ceil(distance))
    cdef float c = cos(angle)
    cdef int i,j,k
    cdef Point point
    for i in range(-size,size+1):
        for j in range(-size,size+1):
            for k in range(-size,size+1):
                point = Point(i,j,k)
                if i!=0 and j!=0 and k!=0 and length(point)<distance+0.0001 and dot(normalize(point),direction)>c:
                    result.push_back(point)
    return result

def calculate_cone(direction,float distance,float angle):
    return _calculate_cone(normalize(Point(direction[0],direction[1],direction[2])),distance,angle)


# def array_speed_test_1(np.ndarray[DTYPE_t, ndim=2] a):
#     cdef int xmax = a.shape[0]
#     cdef int ymax = a.shape[1]
#     cdef float s=0
#     for i in range(xmax):
#         for j in range(ymax):
#             s=s+a[i,j]
#     return s

# def array_speed_test_2(np.ndarray[DTYPE_t, ndim=2] a):
#     cdef int i
#     cdef int j
#     cdef int xmax = a.shape[0]
#     cdef int ymax = a.shape[1]
#     cdef DTYPE_t s=0
#     for i in range(xmax):
#         for j in range(ymax):
#             s=s+a[i,j]
#     return s

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def array_speed_test_3(np.ndarray[DTYPE_t, ndim=2] a):
#     cdef int i
#     cdef int j
#     cdef int xmax = a.shape[0]
#     cdef int ymax = a.shape[1]
#     cdef DTYPE_t s=0
#     for i in range(xmax):
#         for j in range(ymax):
#             s=s+a[i,j]
#     return s

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef aprox_value(Point point,np.ndarray[DTYPE_t, ndim=3] array):
    cdef int x=int(round(point.x))
    cdef int y=int(round(point.y))
    cdef int z=int(round(point.z))
    if x<0 or y<0 or z< 0 or x>array.shape[0] or y>array.shape[1] or z>array.shape[2]:
        #print(x,y,z)
        return 0
    
    return array[x,y,z]


cdef _calculate_initial_direction(Point pos,np.ndarray[DTYPE_t, ndim=3] array):
    cdef int initial=aprox_value(pos,array)
    print("initial",initial)
    if initial==0:
        return Point(0,0,0)
    cdef int found=0
    cdef float distance = 0.0
    cdef Point[2] found_rays
    cdef Point ray
    while found<2:
        distance+=1.0
        for ray in RAYS:
            if aprox_value(add(mul(distance,ray),pos),array)<=0.2*initial and aprox_value(add(mul(-distance,ray),pos),array)<=0.2*initial:
                if found==0:
                    found_rays[0]=ray
                    found+=1
                else:
                    if length(cross(found_rays[0],ray))>0.2:
                        found_rays[1]=ray
                        found+=1
    print(distance,found_rays[0],found_rays[1])
    return normalize(cross(found_rays[0],found_rays[1])),found_rays[0],found_rays[1]


cdef (long,long,long) decode(long code):
    cdef SIZE =1200
    return (code%SIZE,(code//SIZE)%SIZE,(code//SIZE**2)%SIZE)
cdef long encode((long,long,long) pos):
    cdef SIZE =1200
    return (pos[0]+pos[1]*SIZE+pos[2]*SIZE**2)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _calculate_new_direction(Point direction,int posx, int posy,int posz,float distance, float angle,np.ndarray[DTYPE_t, ndim=3] array):
    cdef Point new_direction=Point(0.0,0.0,0.0)
    cdef int size = int(ceil(distance))
    cdef float c = cos(angle)
    cdef int x,y,z
    cdef float value

    cdef Point point
    cdef Point normalized_point

    cdef minx=max(0,posx-size-1)
    cdef miny=max(0,posy-size-1)
    cdef minz=max(0,posz-size-1)

    cdef maxx=min(array.shape[0],posx+size+1)
    cdef maxy=min(array.shape[1],posy+size+1)
    cdef maxz=min(array.shape[2],posz+size+1)
    #cdef (int,int,int) tep_vox

    current_value = np.sum(array[max(0,posx-1):posx+2,max(0,posy-1):posy+2,max(0,posz-1):posz+2])/2700.

    #cdef set[long] points_of_interest
#
    #for x in range(minx,maxx):
    #    for y in range(miny,maxy):
    #        for z in range(minz,maxz):
    #            if x!=posx and y!=posy and z!=posz:
    #                point = Point(x-posx,y-posy,z-posz)
    #                normalized_point=normalize(point)
    #                if length(point)<distance+0.0001 and dot(normalized_point,direction)>c:
    #                    value=(array[x,y,z])/100.0
    #                    if value>current_value*1.3:
    #                        continue
    #                    if value<current_value*0.5:
    #                        continue
    #                    #tep_vox = ((x),(y),(z))
    #                    points_of_interest.insert(encode((x,y,z)))
    #                    value=(array[int(_point[0]),int(_point[1]),int(_point[2])])/100.0
    #                    new_direction.x+=value*normalized_point.x
    #                    new_direction.y+=value*normalized_point.y
    #                    new_direction.z+=value*normalized_point.z
#
    #cdef set[long] visited
    #cdef vector[long] visited_vector
    #print(len(points_of_interest))
    #cdef queue[long] to_visit 
    #to_visit.push(encode((int(posx),int(posy),int(posz))))
    #while not (to_visit.empty()):
    #    _point = to_visit.front()
    #    to_visit.pop()
    #    if visited.count(_point):
    #        continue
    #    visited.insert(_point)
    #    visited_vector.push_back(_point)
    #    for a in (-1,0,1):
    #        for b in (-1,0,1):
    #            for c in (-1,0,1): 
    #                new_point = _point+encode((a,b,c))
    #                if points_of_interest.count(new_point) and not visited.count(new_point):
    #                    to_visit.push(new_point)
    #visited.erase(encode((int(posx),int(posy),int(posz))))
    #print(len(visited))
    #if len(visited)>0:
    #    new_direction=Point(0.0,0.0,0.0)
    #    for encoded in visited_vector:
    #        _point = decode(encoded)
    #        normalized_point=normalize(Point(_point[0]-posx,_point[1]-posy,_point[2]-posz))
    #        value=(array[int(_point[0]),int(_point[1]),int(_point[2])])/100.0
    #        new_direction.x+=value*normalized_point.x
    #        new_direction.y+=value*normalized_point.y
    #        new_direction.z+=value*normalized_point.z
    #else:
#
    #
    #if length(new_direction)>0.01:
    #    return normalize(new_direction),length(new_direction)
    #else:
    #    return Point(0.0,0.0,0.0),0
        
    for x in range(minx,maxx):
        for y in range(miny,maxy):
            for z in range(minz,maxz):
                if x!=posx and y!=posy and z!=posz:
                    point = Point(x-posx,y-posy,z-posz)
                    normalized_point=normalize(point)
                    if length(point)<distance+0.0001 and dot(normalized_point,direction)>c:
                        value=(array[x,y,z])/100.0
                        if value>current_value*1.3:
                            continue
                        if value<current_value*0.5:
                            continue
                        new_direction.x+=value*normalized_point.x
                        new_direction.y+=value*normalized_point.y
                        new_direction.z+=value*normalized_point.z
    if length(new_direction)>0.01:
        return normalize(new_direction),length(new_direction)
    else:
        return Point(0.0,0.0,0.0),0

cdef _calculate_max_direction(int posx, int posy,int posz,float distance,np.ndarray[DTYPE_t, ndim=3] array):
    cdef Point direction,min_direction,max_direction
    cdef float value,min_value,max_value
    while True:
        direction=_RAYS[0]
        _,value = _calculate_new_direction(normalize(direction),posx,posy,posz,distance,0.8,array)
        max_direction=min_direction=direction
        max_value=min_value=value
        for i in range(1,_RAYS_NUMBER):
            direction=_RAYS[i]
            _,value = _calculate_new_direction(normalize(direction),posx,posy,posz,distance,0.8,array)
            if value>max_value:
                max_direction=direction
                max_value=value
            elif value<min_value:
                min_direction=direction
                min_value=value
        if min_value<0.5*max_value:
            return max_direction
        else:
            distance=distance+3


##cdef _generate_line(Point start,Point end,float distance):
##    cdef result = []

def calculate_new_direction(np.ndarray[DTYPE_t, ndim=3] array, (int,int,int) pos, (float,float,float) direction, float distance, float angle):
    p,_ = _calculate_new_direction(normalize(Point(direction[0],direction[1],direction[2])),pos[0],pos[1],pos[2],distance,angle,array)
    return (p['x'],p['y'],p['z'])
def calculate_cone(direction,float distance,float angle):
    return _calculate_cone(normalize(Point(direction[0],direction[1],direction[2])),distance,angle)

def calculate_initial_direction(np.ndarray[DTYPE_t, ndim=3] array, (int,int,int) pos):
    x,a,b=_calculate_initial_direction(Point(pos[0],pos[1],pos[2]),array)
    return (x['x'],x['y'],x['z']),(a['x'],a['y'],a['z']),(b['x'],b['y'],b['z'])

def calculate_max_direction(np.ndarray[DTYPE_t, ndim=3] array, (int,int,int) pos, float distance=20.0):
    p=_calculate_max_direction(pos[0],pos[1],pos[2],distance,array)
    print(p)
    return (p['x'],p['y'],p['z'])


##def generate_line(start,end,distance):
##    result=_generate_line(Point(*start),Point(*end),distance)

def test_tadd():
    tadd((0,1,2),(5,3,2))

def test_add():
    add(Point(0,1,2),Point(5,3,2))