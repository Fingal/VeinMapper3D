from math import sqrt,cos
import numpy as np

def vexel_position(pos: (int,int)) -> (float,float):
    #TODO this function is supposed to calculate vexel position in continous simulation space
    assert False,'Not implemented'

def dif(a: (float,float),b: (float,float)) -> (float,float):
    return (a[0]-b[0],a[1]-b[1])

def add(a: (float,float),b: (float,float)) -> (float,float):
    return (a[0]+b[0],a[1]+b[1])

def mul(A: float,b: (float,float)) -> (float,float):
    return (A*b[0],A*b[1])

def length(a: (float,float)) -> float:
    return sqrt(a[0]**2+a[1]**2)

def dot(a: (float,float),b: (float,float)) -> float:
    return a[0]*b[0]+a[1]*b[1]

def normalize(a: (float,float)) -> (float,float):
    return (a[0]/length(a),a[1]/length(a))


def is_in_cone(agent_position: (float,float), cone_length: float, cone_angle: float, direction: (float,float), point: (float,float)) -> bool:
    point_vector : (float,float) = dif(point,agent_position)
    distance : float = length(point_vector)
    #print(f'distance comparison {distance}, {cone_length}')
    #print(f'angle comparospm {dot(normalize(direction),normalize(point_vector))}, {cos(cone_angle/2)}')
    return distance<0.001 or (distance<cone_length and dot(normalize(direction),normalize(point_vector))>cos(cone_angle/2))


def calculate_direciton(agent_position: (float,float), cone_length: float, cone_angle: float, initial_direction: (float,float),concentration_map: np.array) -> (float,float):
    #TODO limit checked vexels to bounding rectangle
    new_direction : (float,float) = (0,0)
    for i in range(concentration_map.shape[0]):
        for j in range(concentration_map.shape[1]):
            point = vexel_position((i,j))
            if is_in_cone(agent_position,cone_length,cone_angle,initial_direction,point):
                weight = concentration_map[i,j]
                vector = dif(point,agent_position)
                new_direction = add(new_direction,mul(weight,vector))
    
    new_direction = normalize(new_direction)
    return new_direction

if __name__ == "__main__":
    calculate_direciton((0,0),10,1,(1,0),np.array(((0,0),(1,1))))

