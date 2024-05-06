# distutils: language=c++
# cython: language_level=3
# cython: embedsignature=True


from libc.math cimport sqrt,cos,ceil,round,floor
from libcpp.vector cimport vector
from libcpp.map cimport map as _map
from libcpp.utility cimport pair

import numpy as np
cimport numpy as np
import time
cimport cython

#np.import_array()

# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = float



DTYPEI = np.uint16

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

ctypedef np.uint16_t DTYPE_tI
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _opengl_coord_to_stack(np.ndarray[DTYPE_t,ndim=1] point,np.ndarray[DTYPE_t,ndim=1] scaling_ratio, np.ndarray[DTYPE_t,ndim=1] stack_shape):
    cdef np.ndarray[DTYPE_t,ndim=1] new_point =np.array((point[2],point[0],point[1]))
    cdef np.ndarray[DTYPE_t,ndim=1] _scaling_ratio = np.array((scaling_ratio[1],scaling_ratio[0],scaling_ratio[2]))
    return (new_point/_scaling_ratio)*stack_shape + 0.5*stack_shape

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _stack_coord_to_opengl(np.ndarray[DTYPE_t,ndim=1] point,np.ndarray[DTYPE_t,ndim=1] scaling_ratio, np.ndarray[DTYPE_t,ndim=1] stack_shape):
    cdef np.ndarray[DTYPE_t,ndim=1] new_point
    cdef np.ndarray[DTYPE_t,ndim=1] _scaling_ratio = np.array((scaling_ratio[1],scaling_ratio[0],scaling_ratio[2]))
    new_point = (point*_scaling_ratio)/stack_shape - 0.5*scaling_ratio
    return np.array((new_point[1],new_point[2],new_point[0]))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float _distance_point_line_segment(np.ndarray[DTYPE_t,ndim=1] point,np.ndarray[DTYPE_t,ndim=2] line):
    cdef np.ndarray[DTYPE_t,ndim=1] start,end
    cdef np.ndarray[DTYPE_t,ndim=1] closest_point = np.array((0,0,0)).astype(DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] temp_point = np.array((0,0,0)).astype(DTYPE)
    cdef float distance = 9999
    start = line[0]
    end = line[1]
    a = point-start
    b = end-start
    c = point-end
    if np.linalg.norm(b)<0.001:
        return np.linalg.norm(a)
    proj_len = np.dot(a, b) / np.linalg.norm(b)
    if 0 < proj_len < np.linalg.norm(b):
        distance = (np.dot(a, a) - np.dot(a, b) ** 2 / np.dot(b, b)) ** 0.5
    elif proj_len <=0:
        distance = np.linalg.norm(a)
    else:
        distance = np.linalg.norm(c)
    #print(distance,new_distance)
    return distance

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (int,int,int) _point_to_index(np.ndarray[DTYPE_t,ndim=1] point,size):
    return tuple((np.ceil(point/size)*size).astype(int))

import itertools

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _convert_graph_to_tiff(np.ndarray[DTYPE_t,ndim=3] lines,(int,int,int) stack_shape,float unit_to_microns_coeff,np.ndarray[DTYPE_t,ndim=1] scaling_ratio, float width):
    partitioning={}
    shape_arr = np.array(stack_shape).astype(DTYPE)
    SIZE=2+int(np.ceil(np.max(np.abs(_opengl_coord_to_stack(np.array((0.0,0.0,0.0)),scaling_ratio,shape_arr)-_opengl_coord_to_stack(np.array((1.0,1.0,1.0))/unit_to_microns_coeff,scaling_ratio,shape_arr)))))
    box_dimentions=(_stack_coord_to_opengl(np.array((1.,1.,1.))*SIZE,scaling_ratio,shape_arr)*unit_to_microns_coeff-_stack_coord_to_opengl(np.array((0.,0.,0.)),scaling_ratio,shape_arr)*unit_to_microns_coeff)
    diag_len = np.sum(box_dimentions*box_dimentions)**0.5
    print(SIZE)
    options = [np.array(x) for x in itertools.product(range(-1,2),repeat=3)]
    #for x in range(0,stack_shape[0],10):
    #    for y in range(0,stack_shape[1],10):
    #        for z in range(0,stack_shape[2],10):
    for line in lines:
        indices = set()
        lowest = _point_to_index(_opengl_coord_to_stack(line[0]+(width+1)*np.array((-1.,-1.,-1.))/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE)
        highest = _point_to_index(_opengl_coord_to_stack(line[0]+(width+1)*np.array((1.,1.,1.))/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE)
        lowest2 = _point_to_index(_opengl_coord_to_stack(line[1]+(width+1)*np.array((-1.,-1.,-1.))/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE)
        highest2 = _point_to_index(_opengl_coord_to_stack(line[1]+(width+1)*np.array((1.,1.,1.))/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE)
        for x in range(min(lowest[0],highest[0],lowest2[0],highest2[0]),max(lowest[0],highest[0],lowest2[0],highest2[0])+1,SIZE):
            for y in range(min(lowest[1],highest[1],lowest2[1],highest2[1]),max(lowest[1],highest[1],lowest2[1],highest2[1])+1,SIZE):
                for z in range(min(lowest[2],highest[2],lowest2[2],highest2[2]),max(lowest[2],highest[2],lowest2[2],highest2[2])+1,SIZE):
                    pos_micron = _stack_coord_to_opengl(np.array((x,y,z)).astype(DTYPE)-SIZE/2,scaling_ratio,shape_arr)*unit_to_microns_coeff
                    distance3 = _distance_point_line_segment((pos_micron),line*unit_to_microns_coeff)
                    if distance3<width+0.5+diag_len/2:
                        indices.add((x,y,z))
        # for option in options:
        #     indices.add(_point_to_index(_opengl_coord_to_stack(line[0]+(width+1)*option/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE))
        #     indices.add(_point_to_index(_opengl_coord_to_stack(line[1]+(width+1)*option/unit_to_microns_coeff,scaling_ratio,shape_arr),SIZE))
        for index in indices:
            #if index[2]<50 or index[2]>70 or index[1]>500 or index[1]<40 or index[0]<800 or index[0]>1000:
            v = partitioning.get(index,[])
            v.append(line)
            partitioning[index]=v
    #print([len(v) for k,v in partitioning.items()])
    #print(max([len(v) for k,v in partitioning.items()]))
    #print(len([len(v) for k,v in partitioning.items()]))
    #print(stack_shape[0]*stack_shape[1]*stack_shape[2])


    cdef np.ndarray[DTYPE_tI,ndim=3] stack = np.zeros(stack_shape,dtype=DTYPEI)
    zeros=np.array((0,0,0)).astype(DTYPE)
    i=0
    import time
    t=time.time()
    steps = len(partitioning)
    for (X,Y,Z),llines in partitioning.items():
        i+=1
        if i%100==0:
            print((X,Y,Z),i, "out of",steps)
            print(time.time()-t)
        #if i==300:
        #    break
        for x in range(max(0,X-SIZE),min(stack_shape[0],X)):
            for y in range(max(0,Y-SIZE),min(stack_shape[1],Y)):
                for z in range(max(0,Z-SIZE),min(stack_shape[2],Z)):
                    pos = (x,y,z)
                    arr_pos=np.array(pos).astype(DTYPE)
                    pos_micron = _stack_coord_to_opengl(arr_pos,scaling_ratio,shape_arr)*unit_to_microns_coeff
                    distance = 99999
                    #stack[pos]=4000
                    for line in llines:
                        distance = min(distance,_distance_point_line_segment((pos_micron),line*unit_to_microns_coeff))
                    if distance<width*0.3:
                        stack[pos]=4000
                    elif distance<width+0.2:
                        coeff = min(width+0.1,(distance-width*0.3))
                        stack[pos]=max(stack[pos],4000*(1-coeff/(width+0.3)))
    print(time.time()-t)
    return stack
                    

    #return stack
    #for line in lines:
    #    i+=1
    #    if (i%10):
    #        print("time", time.time()-t)
    #        t=time.time()
    #    #i-=1
    #    low_corner = np.minimum(line[0],line[1])-width/unit_to_microns_coeff-1/unit_to_microns_coeff
    #    high_corner = np.maximum(line[0],line[1])+width/unit_to_microns_coeff+1/unit_to_microns_coeff
    #    low_corner_stack=_opengl_coord_to_stack(low_corner,scaling_ratio,shape_arr)
    #    high_corner_stack=_opengl_coord_to_stack(high_corner,scaling_ratio,shape_arr)
    #    distances = 9999
    #    poses = []
    #    # for a in range(int(floor(min(low_corner_stack[0],high_corner_stack[0]))),int(ceil(max(low_corner_stack[0],high_corner_stack[0])))):
    #    #     for b in range(int(floor(min(low_corner_stack[1],high_corner_stack[1]))),int(ceil(max(low_corner_stack[1],high_corner_stack[1])))):
    #    #         for c in range(int(floor(min(low_corner_stack[2],high_corner_stack[2]))),int(ceil(max(low_corner_stack[2],high_corner_stack[2])))):
    #    #             poses.append((a,b,c))
    #    for a in range(int(floor(low_corner_stack[0])),int(ceil(high_corner_stack[0]))):
    #        for b in range(int(floor(low_corner_stack[1])),int(ceil(high_corner_stack[1]))):
    #            for c in range(int(floor(low_corner_stack[2])),int(ceil(high_corner_stack[2]))):
    #                pos = (a,b,c)
    #                arr_pos=np.array(pos).astype(DTYPE)
    #                if not all(arr_pos<shape_arr) or any(arr_pos<zeros):
    #                    continue
    #                pos_micron = _stack_coord_to_opengl(arr_pos,scaling_ratio,shape_arr)*unit_to_microns_coeff
    #                distance = _distance_point_line_segment((pos_micron),line*unit_to_microns_coeff)
    #                if distance<width*0.3:
    #                    stack[pos]=4000
    #                elif distance<width+0.5:
    #                    coeff = min(width+0.3,(distance-width*0.3))
    #                    stack[pos]=max(stack[pos],4000*(1-coeff/(width+0.3)))
    #        #self.context.graphic_context.mark_temporary_points(positions)
    #        #print(low_corner_stack,high_corner_stack)
    #return stack

def convert_graph_to_tiff(np.ndarray[DTYPE_t,ndim=3] lines,(int,int,int) stack_shape,float unit_to_microns_coeff,np.ndarray[DTYPE_t,ndim=1] scaling_ratio, float width=0.4):
    return _convert_graph_to_tiff(lines,stack_shape,unit_to_microns_coeff,scaling_ratio,width)