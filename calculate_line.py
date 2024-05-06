from time import time
from math import ceil
from itertools import  product
import numpy as np
import cone_calculation

directions=[
    ([-0.57735027, -0.57735027, -0.57735027]),
    ([-0.70710678, -0.70710678,  0.        ]),
    ([-0.57735027, -0.57735027,  0.57735027]),
    ([-0.70710678,  0.        , -0.70710678]),
    ([-1.,  0.,  0.]),
    ([-0.70710678,  0.        ,  0.70710678]),
    ([-0.57735027,  0.57735027, -0.57735027]),
    ([-0.70710678,  0.70710678,  0.        ]),
    ([-0.57735027,  0.57735027,  0.57735027]),
    ([ 0.        , -0.70710678, -0.70710678]),
    ([ 0., -1.,  0.]),
    ([ 0.        , -0.70710678,  0.70710678]),
    ([ 0.,  0., -1.]),
    ([0., 0., 1.]),
    ([ 0.        ,  0.70710678, -0.70710678]),
    ([0., 1., 0.]),
    ([0.        , 0.70710678, 0.70710678]),
    ([ 0.57735027, -0.57735027, -0.57735027]),
    ([ 0.70710678, -0.70710678,  0.        ]),
    ([ 0.57735027, -0.57735027,  0.57735027]),
    ([ 0.70710678,  0.        , -0.70710678]),
    ([1., 0., 0.]),
    ([0.70710678, 0.        , 0.70710678]),
    ([ 0.57735027,  0.57735027, -0.57735027]),
    ([0.70710678, 0.70710678, 0.        ]),
    ([0.57735027, 0.57735027, 0.57735027])
 ]



direction_cones={}
def normalize(vec):
    return vec/np.linalg.norm(vec)

def calculate_cone(direction,distance,angle):
    cos=np.cos(angle)
    possible=product(range(-ceil(distance),ceil(distance)+1),repeat=3)
    result=[]
    for vec in possible:
        if vec!=(0,0,0) and np.linalg.norm(np.array(vec))<=distance and direction @ normalize(vec)>cos:
            result.append(vec)
    return result



# def precalculate_cones(distance,angle):
#     direction_cones={}
#     for direction in directions:
#         if tuple(direction) not in direction_cones:
#             direction_cones[tuple(direction)]=cone.calculate_cone(direction,distance,angle)
#     return direction_cones


def get_aprox_direction(direction):
    return sorted(directions,key=lambda x: x@direction)[-1]

# def calculate_new_direction(array,pos,aprox_direction,direction_cones):
#     pos_x,pos_y,pos_z=pos
#     new_direction=np.array([0.0,0.0,0.0])
#     values = [array[x+pos_x,y+pos_y,z+pos_z] for x,y,z in direction_cones[tuple(aprox_direction)]]
#     for x,y,z in direction_cones[tuple(aprox_direction)]:
#         new_direction=new_direction+array[x+pos_x,y+pos_y,z+pos_z]/100*normalize(np.array([x,y,z]))
#     if np.linalg.norm(new_direction)>0.01:
#         return normalize(new_direction),np.linalg.norm(new_direction)
#     else: 
#         return np.array([0,0,0]),0
        
# def calculate_new_direction(array,pos,direction_cone):
#     pos_x,pos_y,pos_z=pos
#     new_direction=np.array([0.0,0.0,0.0])
#     values = [array[x+pos_x,y+pos_y,z+pos_z] for x,y,z in direction_cone]
    
#     result = cone.calculate_direction(values,direction_cone)
#     #print('direction:',result)
#     if result[3]>0.01:
#         return np.array(result[:3]),result[3]
#     else: 
#         return np.array([0,0,0]),0

# def calculate_step(array,pos,direction_cones,angle=np.pi/4,distance=5,direction=np.array([-1,0,0])):
#     return calculate_new_direction(array,pos,direction_cones)

def calculate_line(array,start_pos,angle=180.0,distance=5,direction=np.array([0,0,-1]),step_size=3,precalculated_cones=None,inertia=1.0,steps_number=10,flood_fill=False):
    direction=normalize(direction)
    angle=angle/360*np.pi
    start_time = time()
    if precalculated_cones:
        direction_cones=precalculated_cones
    else:
        #direction_cones=precalculate_cones(distance,angle)
        #print("cone time",time()-start_time)
        pass
    positions=[start_pos]
    start_time = time()
    deltas =[]
    while len(positions)<=steps_number:
        last_position=positions[-1]
        #TODO compare speed
        #direction_cone=cone.calculate_cone(tuple(direction),distance,angle)
        t = time()
        new_direction=cone_calculation.calculate_new_direction(array,last_position,tuple(direction),distance,angle,flood_fill)
        delta = time()-t
        deltas.append(delta)
        print("single time",delta)
        #print('angle',np.arccos(get_aprox_direction(direction)@new_direction))
        #print('cos',(get_aprox_direction(direction)@new_direction))
        #print('values',(get_aprox_direction(direction),new_direction))
        #print(np.linalg.norm(new_direction))
        new_direction=np.array(new_direction)
        print(new_direction)
        if np.linalg.norm(new_direction)>0:
            new_pos=tuple((np.array(last_position)+new_direction*step_size).astype(int))
        else:
            new_pos=last_position
        if new_pos[0]<0 or new_pos[1]<0 or new_pos[2]<0 or new_pos[0]>array.shape[0] or new_pos[1]>array.shape[1] or new_pos[2]>array.shape[2]:
            direction=None
            break
        #print(last_position,new_pos)
        positions.append(new_pos)
        direction=normalize(direction*inertia+new_direction)
    print("calculation time",time()-start_time)
    print("average delta", sum(deltas)/len(deltas))
    return positions,direction
    

def iter_calculate_line(array,start_pos,angle=np.pi/4,distance=5,direction=np.array([0,0,-1]),step_size=3,precalculated_cones=None,inertia=1.0,flood_fill=False):
    direction=normalize(direction)
    start_time = time()
    if precalculated_cones:
        direction_cones=precalculated_cones
    else:
        #direction_cones=precalculate_cones(distance,angle)
        #print("cone time",time()-start_time)
        pass
    positions=[start_pos]
    start_time = time()
    while True:
        last_position=positions[-1]
        #TODO compare speed
        #direction_cone=cone.calculate_cone(tuple(direction),distance,angle)
        new_direction=cone_calculation.calculate_new_direction(array,last_position,tuple(direction),distance,angle,flood_fill)
        #print('angle',np.arccos(get_aprox_direction(direction)@new_direction))
        #print('cos',(get_aprox_direction(direction)@new_direction))
        #print('values',(get_aprox_direction(direction),new_direction))
        #print(np.linalg.norm(new_direction))
        new_direction=np.array(new_direction)
        if np.linalg.norm(new_direction)>0:
            new_pos=tuple((np.array(last_position)+new_direction*step_size).astype(int))
        else:
            new_pos=last_position
        #print(last_position,new_pos)
        positions.append(new_pos)
        direction=normalize(direction*inertia+new_direction)
        yield new_pos
    #print("calculation time",time()-start_time)
