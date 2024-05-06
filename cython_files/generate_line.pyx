# distutils: language=c++
# cython: language_level=3
# cython: embedsignature=True

from libc.math cimport sqrt,cos,ceil,round,floor,sqrt
from libcpp.vector cimport vector
from libcpp.map cimport map as _map
from libcpp.utility cimport pair

import numpy as np
cimport numpy as np
import time
cimport cython
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


ctypedef (float, float, float) Point
ctypedef (int,int,int) IndexPoint

cdef float dot(Point a,Point b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

cdef Point add(Point a, Point b):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

cdef Point dif(Point a, Point b):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2])

cdef float length(Point a):
    return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

cdef Point normalize(Point a):
    cdef float l=length(a)
    return (a[0]/l,a[1]/l,a[2]/l)

cdef Point mul(float n,Point a):
    return (a[0]*n,a[1]*n,a[2]*n)

cdef Point cross(Point a, Point b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

cdef float distance2d(Point a, Point b):
    cdef c = dif(a,b)
    return sqrt((c[0]**2+c[2]**2))

cdef float distance(Point a, Point b):
    return length(dif(a,b))

def tests():
    assert(dot((1,0,0),(0.5,1,0))==0.5)
    assert((add((1,0,0),(0.5,1,0))).x==1.5)
    assert((add((1,0,0),(0.5,1,0))).y==1.0)
    assert((add((1,0,0),(0.5,1,0))).z==0.0)
    print((-0.001<(length((3,4,0))-5)))
    print((length((3,4,0))-5))

cdef _generate_line(Point start,Point end,float variation, float step,float min_distance):
    cdef result = [start]
    cdef Point direction,normal,binormal,side
    cdef float dist
    cdef float angle
    distance = length(dif(start,end))
    while distance>min_distance:
        direction=normalize(dif(end,start))
        dist = np.random.normal(scale=variation/2)
        angle = np.random.uniform(np.pi)

        normal=cross(direction,(0.,0.,1.))
        if length(normal)<0.001:
            normal=cross(direction,(0.,1.,0.))
        normal=normalize(normal)
        
        binormal=normalize(cross(normal,direction))

        side = add(mul(np.cos(angle),normal),mul(np.sin(angle),binormal))
        start = add(start,mul(step,normalize(add(mul(dist,side),direction))))
        result.append(start)
        distance=length(dif(start,end))
    result.append(end)
    return result
    

cdef _calculate_distances(points,line):
    cdef result = []
    cdef float distance, new_distance
    cdef Point point,start,end
    for point in points:
        distance = 9999
        for start, end in line:
            a = dif(point,start)
            b = dif(end,start)
            c = dif(point,end)
            proj_len = dot(a, b) / length(b)
            if 0 < proj_len < length(b):
                new_distance = sqrt((dot(a, a) - dot(a, b) ** 2 / dot(b, b)))
            else:
                new_distance = min(length(a), length(c))
            #print(distance,new_distance)
            distance=min(distance,new_distance)
        result.append(distance)
    return result

cdef _calculate_distance_(points,line):
    cdef result = []
    cdef float distance, new_distance
    cdef Point point,start,end
    for point in points:
        distance = 9999
        for start, end in line:
            a = dif(point,start)
            b = dif(end,start)
            c = dif(point,end)
            proj_len = dot(a, b) / length(b)
            if 0 < proj_len < length(b):
                new_distance = (dot(a, a) - dot(a, b) ** 2 / dot(b, b)) ** 0.5
            else:
                new_distance = min(length(a), length(c))
            #print(distance,new_distance)
            distance=min(distance,new_distance)
        result.append(distance)
    return result

cdef (float,Point) _calculate_closest(Point point,line):
    cdef Point start,end
    cdef Point closest_point = (0,0,0)
    cdef Point temp_point = (0,0,0)
    cdef float new_distance, distance = 9999
    for start, end in line:
        a = dif(point,start)
        b = dif(end,start)
        c = dif(point,end)
        if length(b)==0:
            continue
        proj_len = dot(a, b) / length(b)
        if 0 < proj_len < length(b):
            proj_len=proj_len/length(b)
            new_distance = (dot(a, a) - dot(a, b) ** 2 / dot(b, b)) ** 0.5
            temp_point=add(mul(proj_len,end),mul(1-proj_len,start))
        elif proj_len <=0:
            new_distance = length(a)
            temp_point=start
        else:
            new_distance = length(c)
            temp_point=end
        #print(distance,new_distance)
        if new_distance<distance:
            distance=new_distance
            closest_point=temp_point
    return distance,closest_point  

cdef (float,Point) _calculate_closest_point_line(Point point,line):
    cdef Point start,end
    cdef Point closest_point = (0,0,0)
    cdef Point temp_point = (0,0,0)
    cdef float new_distance, distance = 9999
    for start, end in line:
        a = dif(point,start)
        b = dif(end,start)
        c = dif(point,end)
        if length(b)==0:
            continue
        proj_len = dot(a, b) / length(b)
        if 0 < proj_len < length(b):
            proj_len=proj_len/length(b)
            new_distance = (dot(a, a) - dot(a, b) ** 2 / dot(b, b)) ** 0.5
            temp_point=add(mul(proj_len,end),mul(1-proj_len,start))
            if length(dif(temp_point,start))<length(dif(temp_point,end)):
                temp_point=start
            else:
                temp_point=end
        elif proj_len <=0:
            new_distance = length(a)
            temp_point=start
        else:
            new_distance = length(c)
            temp_point=end
        #print(distance,new_distance)
        if new_distance<distance:
            distance=new_distance
            closest_point=temp_point
    return distance,closest_point  

cdef (float,Point) _calculate_closest_point(Point point,line):
    cdef Point start,end
    cdef Point closest_point = (0,0,0)
    cdef Point temp_point = (0,0,0)
    cdef float distance = 9999
    for start, end in line:
        dist1=length(dif(point,start))
        if dist1<distance:
            distance=dist1
            closest_point=start
        dist1=length(dif(point,end))
        if dist1<distance:
            distance=dist1
            closest_point=start
    return distance,closest_point

cdef float _calculate_strength(float distance,float age, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float peak_coef):
    cdef float result=0.0
    #modified for testing
    #pos_range=pos_range*(1+age_coef*(age-4))
    age = max(15,age)
    coef = sqrt(max(0,(1+age_coef*(age-age_cut_off_coef))))
    neg_range=max(0,neg_range*coef)
    pos_range=pos_range+neg_range
    peak_coef=max(0.01,min(0.99,peak_coef))
    if distance<neg_range:
        #result= -(distance-30*(0.9+age*0.05))*(0.5+age*0.2)
        result = -(inhibition_coef/neg_range)*distance+inhibition_coef
    elif distance< neg_range*(1-peak_coef)+pos_range*peak_coef:
        result=-(distance-neg_range)*(attraction_coef/(pos_range*peak_coef-neg_range*peak_coef))
    else:
        result= min(0,-attraction_coef+(distance-(neg_range*(1-peak_coef)+pos_range*peak_coef))*(attraction_coef/(pos_range*(1-peak_coef)-neg_range*(1-peak_coef))))
    if result >0:
        return result
    else:
        return result

#cdef float _calculate_strength_old(float distance,float age, float _range, float strongest, float fallof, float inhibition_coef, float attraction_coef,float age_coef):
#    cdef float result=0.0
#    # cdef float _range=150
#    # cdef float strongest=20
#    # cdef float fallof=0.5
#    if distance<_range:
#        #result= -(distance-30*(0.9+age*0.05))*(0.5+age*0.2)
#        result= -(distance*fallof-strongest)*(0.5+age*age_coef)
#    else:
#        result= min(0,(-(_range*fallof-strongest)*(0.5+age*age_coef))+(distance-_range)*(0.5))
#    if result >0:
#        return result*inhibition_coef
#    else:
#        return result*attraction_coef

cdef _predict_step_mark_points(Point pos,Point target,points, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    cdef Point direction=(0,0,0)
    cdef float distance, strength,_distance
    _distance = 100
    cdef temp_dir,other_pos
    for line,age in lines:
        distance,other_pos = _calculate_closest(pos,line)
        temp_dir=normalize(dif(pos,other_pos))
        points.append(other_pos)
        strength = _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
        direction=add(direction,mul(strength,temp_dir))
    direction=normalize(add(direction,mul(straight_coef*(1+300/(length(dif(pos,target))*length(dif(pos,target))+1)),normalize(dif(target,pos)))))
    #direction=normalize(dif(target,pos))
    return direction,other_pos,distance



cdef _predict_step_get_distance(Point pos,Point target, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    cdef Point direction=(0,0,0)
    cdef float distance, strength,_distance
    _distance = 100
    cdef temp_dir,other_pos
    for line,age in lines:
        distance,other_pos = _calculate_closest(pos,line)
        temp_dir=normalize(dif(pos,other_pos))
        strength = _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
        direction=add(direction,mul(strength,temp_dir))
        _distance=min(_distance,distance)
    direction=normalize(add(direction,mul(straight_coef*(1+300/(length(dif(pos,target))*length(dif(pos,target))+1)),normalize(dif(target,pos)))))
    #direction=normalize(dif(target,pos))
    return direction,_distance

cdef _predict_step(Point pos,Point target, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    cdef Point direction=(0,0,0)
    cdef float distance, strength
    cdef temp_dir,other_pos
    for line,age in lines:
        distance,other_pos = _calculate_closest(pos,line)
        temp_dir=normalize(dif(pos,other_pos))
        strength = _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
        direction=add(direction,mul(strength,temp_dir))
    direction=normalize(add(direction,mul(straight_coef*(1+300/(length(dif(pos,target))*length(dif(pos,target))+1)),normalize(dif(target,pos)))))
    #direction=normalize(dif(target,pos))
    #return direction,other_pos,distance
    return direction

cdef _predict_step_global_simulation(Point pos,Point target, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    cdef Point direction=(0,0,0)
    cdef float distance, strength
    cdef temp_dir,other_pos
    cdef float lowest_distance = 10000.0
    cdef Point closest_point = (0,0,0)
    for line,age in lines:
        distance,other_pos = _calculate_closest(pos,line)
        if lowest_distance>distance:
            lowest_distance=distance
            closest_point= other_pos
        temp_dir=normalize(dif(pos,other_pos))
        strength = _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
        direction=add(direction,mul(strength,temp_dir))
    direction=normalize(add(direction,mul(straight_coef*(1+300/(length(dif(pos,target))*length(dif(pos,target))+1)),normalize(dif(target,pos)))))
    #direction=normalize(dif(target,pos))
    #return direction,other_pos,distance
    return direction,closest_point

def predict_step_global_simulation(Point pos,Point target, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    return _predict_step_global_simulation(pos,target, lines, pos_range, neg_range, inhibition_coef, attraction_coef,age_coef, age_cut_off_coef, straight_coef, peak_coef)

def predict_step(Point pos,Point target, lines, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float straight_coef, float peak_coef):
    return _predict_step(pos,target, lines, pos_range, neg_range, inhibition_coef, attraction_coef,age_coef, age_cut_off_coef, straight_coef, peak_coef)

#def generate_model_line(start,target,lines,inertia,pos_range,strongest,neg_range,inhibition_coef,attraction_coef,age_coef,straight_coef):
#    cdef Point step=start
#    cdef Point direction, old_direction, other_pos
#    cdef float distance=0,max_distance=0
#    cdef result = []
#    cdef points= []
#    cdef int i=0
#    old_direction=normalize(dif(target,start))
#    cdef int MAX_STEP=1000
#    while length(dif(target,step))>10:
#        direction,other_pos,distance = _predict_step(step,target,points,lines,pos_range,strongest,neg_range,inhibition_coef,attraction_coef,age_coef,straight_coef)
#        step=add(step,mul(1/(1.+inertia),add(direction,mul(inertia,old_direction))))
#        old_direction=mul(1/(1.+inertia),add(direction,mul(inertia,old_direction)))
#        result.append(step)
#        max_distance=max(distance,max_distance)
#        i+=1
#        if i>MAX_STEP:
#            #print('distance:',length(dif(step,target)))
#            break
#    if i<=MAX_STEP:
#        result.append(target)
#    return result,points
def generate_model_line(start,target,original_line,lines,inertia,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef, peak_coef,center):
    cdef float SIMSTEP = 4
    cdef Point step=start
    cdef Point direction, old_direction, other_pos,original_pos
    cdef float distance=0,max_distance=0,_distance
    cdef result = []
    cdef points= []
    cdef int i=0
    old_direction=normalize(dif(target,start))
    cdef int MAX_STEP=1000
    #while length(dif(target,step))>10:
    #while length(dif(target,step))>10 or length(dif(step,center))>length(dif(target,center))*0.95:
    while length(dif(step,center))>length(dif(target,center))*0.95:
        #direction,other_pos,distance = _predict_step(step,target,points,lines,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,straight_coef, peak_coef)
        direction,other_pos,distance = _predict_step_mark_points(step,center,points,lines,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef, peak_coef)
        direction=mul(SIMSTEP,normalize(add(direction,mul(inertia,old_direction))))
        step=add(step,direction)


        #setting height 
        _distance,original_pos = _calculate_closest(step,original_line)
        step=(step[0],original_pos[1],step[2])

        old_direction=normalize(direction)
        result.append(step)
        max_distance=max(distance,max_distance)
        i+=1
        if i>MAX_STEP:
            #print('distance:',length(dif(step,target)))
            break
    if length(dif(target,step))<=3:
         result.append(target)
    return result,points

def calculate_distance(points,line):
    return _calculate_distances(points,line)

def generate_line_random(start,end,variation,step, min_distance=None):
    if min_distance is None:
        min_distance=step*2
    result=_generate_line((start),(end),variation,step,min_distance)
    return result
    
def calculate_strength(float distance,float age, float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef,float age_cut_off_coef, float peak_coef):
    return _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef, peak_coef)

#cdef float calculate_square_distance(float old_distance,float time_step, float a = 0.2, float b = 6, float c = 62.6):
cdef float calculate_square_distance(float old_distance,float time_step, float a = 0.114, float b = 2.842, float c = 50.03):
    # cdef float a = 0.2098 
    # cdef float b = 6.3424
    # cdef float c = 68.8
    cdef float delta = b**2-4*a*(c-old_distance)
    if delta < 0:
        return old_distance+b*time_step
    cdef float time = (- b + sqrt(delta))/(2*a)
    cdef float new_time = time+time_step
    cdef float new_distance = a*new_time**2+b*new_time+c
    return new_distance

def test_scquare(float old_distance,float time_step, float a = 0.2, float b = 6, float c = 62.6):
    return calculate_square_distance(old_distance, time_step, a, b, c)

cdef Point grow_point(Point point, Point center, float time_step):
    cdef Point vector = dif(point,center)
    cdef float old_distance = length(vector)
    cdef float new_distance = calculate_square_distance(old_distance,time_step)
    cdef new_vector = mul(new_distance/old_distance,vector)
    return add(center,new_vector)

def grow_points(points,center,float time_step):
    return [grow_point(point,center,time_step) for point in points]


cdef _project_to_cone(Point point, Point _center, float a1, float a2,float c,float tan):
    cdef Point center,_dir,b,new_point,center_dir
    cdef float distance
    _dir = normalize((a1,-1,a2))
    center = add(_center , mul(c,_dir))
    a = dif(point,center)
    proj_len = dot(a,_dir)
    distance = (dot(a,a) - proj_len ** 2) ** 0.5
    expected_len = distance/tan
    new_point = add(point,mul(expected_len-proj_len,_dir))
    #TODO projection by normal
    #center_dir = normalize(dif(center,new_point))
    #center_dir = mul(dot(mul(-1,dif(new_point,point)),center_dir),center_dir)
    #new_point = add(new_point,center_dir)

    return new_point


def project_to_cone(Point point, Point _center, float a1, float a2,float c,float tan):
    return _project_to_cone(point, _center, a1 , a2 ,c, tan)


 
cdef _grow_point_cone(Point point, Point _center, float a1, float a2,float c,float tan, float time_step, float A = 0.2, float B = 6, float C = 62.6):
    cdef Point projected_point,vector,center
    projected_point = _project_to_cone(point, _center, a1 , a2 ,c, tan)
    _dir = normalize((a1,-1,a2))
    center = add(_center , mul(c,_dir))

    vector = normalize(dif(projected_point,center))
    #cdef float old_distance = distance2d(point,center)
    cdef float old_distance = length(dif(point,center))
    cdef float new_distance = calculate_square_distance(old_distance,time_step,A,B,C)
    return add(point,mul(new_distance-old_distance,vector))


 
#def grow_points_cone(points,Point _center, float time_step, float a1=0, float a2=0, float c=0, float tan = 2.97, float A = 0.2, float B = 6, float C = 62.6):
def grow_points_cone(points,Point _center, float time_step, float a1=0, float a2=0, float c=0, float tan = 2.97, float A = 0.114, float B = 2.842, float C = 50.03):
    #print(a1,a2,c,tan,A,B,C)
    return [_grow_point_cone(point,_center,a1,a2,c,tan,time_step,A,B,C) for point in points]

def graph_to_lines(graph,points):
    cdef list result = []
    for key, items in graph.items():
        for item in items:
            if (points[key], points[item]) not in result and (points[item], points[key]) not in result:
                result.append((points[key], points[item]))
    return result

def gen_prediction_lines(global_simulation):
    prediction_lines = []
    for v in global_simulation.lines:
        if v.matured_age<global_simulation.max_adulted_age:
            prediction_lines.append((graph_to_lines(v.skeleton,global_simulation.points),(v.matured_age)))
    return prediction_lines
def simulation_vector_in_pos(prediction_lines,pos,cone_coeffs,center,
                            inertia,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef, peak_coef):
        inertia = 0                    
        direction,_closest_point = _predict_step_global_simulation(pos,center, prediction_lines, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef,age_cut_off_coef, straight_coef, peak_coef)
            
        #closest_direction = dif(_closest_point,pos)
        closest_direction = dif(_project_to_cone(_closest_point,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan']),pos)

        step=add(pos,mul(1,direction))
        rate = 1
        #set position on cone
        step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
        _dir = dif(step,pos)
        _l = length(_dir)
        i=0
        #print(i,_l/rate)
        while (_l/rate<0.999 or _l/rate>1.001) and i<10:
            step = add(pos,mul(rate/_l,_dir))
            step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
            _dir = dif(step,pos)
            _l = length(_dir)
            i+=1

        return direction

def global_simulation_step(steps,global_simulation,cone_coeffs,
                            inertia,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef, peak_coef,
                            growth_rate=30,distance_treshold=(50,4),connection_distance=5,young_attraction_distance=8,
                            young_attraction_strength=0.2,time=1.,do_maximum_attraction=False,maximum_attraction_treshold=35,max_adulted_age=10):
    '''steps,center,young_lines,old_lines,points,**coef,growth_rate=40,distance_treshold=(50,4),connection_distance=5
    young_lines as dict with active: bool, initial_direction as old_direction, line containing at leat one Point, empty list of connected. 
    old_lines dict key-label/age value-skeleton_dict with indices, points === self.points everything should be in micron coords'''
    cdef float time_step = time/steps
    cdef Point center = global_simulation.center
    young_lines = global_simulation.in_progres
    old_lines = global_simulation.lines
    cdef Point pos,old_direction,direction,direction2,step,attr_dir,closest_direction
    a = 1
    print('steps:', steps, 'time_step: ',time_step)

    for _i in range(steps):
        distance_treshold=global_simulation.distance_treshold
        young_lines = global_simulation.in_progres
        old_lines = global_simulation.lines
        growth_rate = global_simulation.growth_rate
        #generate lines
        prediction_lines = []
        adult_lines=[]
        for v in global_simulation.lines:
            if v.matured_age<max_adulted_age:
                prediction_lines.append((graph_to_lines(v.skeleton,global_simulation.points),(v.matured_age)))
                adult_lines.append(prediction_lines[-1])
            else:
                adult_lines.append((graph_to_lines(v.skeleton,global_simulation.points),(v.matured_age)))
        #young growth step
        #print([p.active for p in young_lines])
        for i,line in enumerate(global_simulation.in_progres):
            if not line.active:
                continue
            if distance2d(center,line.line[-1])<distance_treshold[0]-2:
                consumed = global_simulation.surface_points.check_if_connected(line.line[-1])
                if consumed:
                    age = (global_simulation.MATURED_AGE)*global_simulation.growth_coeff_development.calculate_surface_frequency(global_simulation.simulation_time)/12-consumed.stage
                    age = (global_simulation.MATURED_AGE-1)*global_simulation.growth_coeff_development.calculate_surface_frequency(global_simulation.simulation_time)/12-consumed.stage
                    if age is not None:
                        line.active=False
                        line.set_primordium_label(consumed.primordium_label)
                        line.start_maturing(maturing_adult_time=age)
                continue
                if line.reach_ring_age<0.:
                    line.reach_ring_age=global_simulation.simulation_time

            pos=line.line[-1]
            old_direction = line.old_direction

            # growing_primordium = global_simulation.surface_points.find_closest(pos)
            # if growing_primordium is not None:
            #     _center = mul(0.5,add(growing_primordium,center))
            # else:
            #     _center = center

            _center = center
            #not used generally 
            if do_maximum_attraction:
                if distance2d(center,line.line[-1])<distance_treshold[0]+10:
                    result=global_simulation.surface_points.find_closest(line.line[-1])
                    if result:
                        maximum_pos,angle_distance=result
                        if angle_distance<maximum_attraction_treshold:
                            print("atracted to maximum")
                            print("atracted to maximum", line.get_label())
                            print("atracted to maximum")
                            _center=add(mul(0.7,maximum_pos),mul(0.3,center))


            direction,_closest_point = _predict_step_global_simulation(pos,_center, prediction_lines, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef,age_cut_off_coef, straight_coef, peak_coef)
            _,_closest_point =         _predict_step_global_simulation(pos,_center, adult_lines, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef,age_cut_off_coef, straight_coef, peak_coef)
            direction = normalize(add(direction,mul(inertia/(time_step**2*400),old_direction)))
            #young attraction pass
            #TODO more complex function
            if line.age<0 or line.age>1.0:
                for second in global_simulation.in_progres:
                    if second!=line and second not in line.connected_secondary and (second.age<0 or second.age>1.0):
                        second_line = list(map(tuple,zip(second.line,second.line[1:])))
                        center_diff=length(dif(center,second.line[-1]))-length(dif(center,pos))
                        if center_diff<4:
                            distance,closest_point = _calculate_closest(pos,second_line)
                            if distance<young_attraction_distance:
                                attr_dir=normalize(dif(closest_point,pos))
                                direction=normalize(add(mul(young_attraction_strength,attr_dir),direction))
                
            #closest_direction = dif(_closest_point,pos)
            closest_direction = dif(_project_to_cone(_closest_point,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan']),pos)

            step=add(pos,mul(time_step*growth_rate,direction))
            rate = time_step*growth_rate
            #set position on cone
            step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
            _dir = dif(step,pos)
            _l = length(_dir)
            i=0
            #print(i,_l/rate)
            while (_l/rate<0.999 or _l/rate>1.001) and i<10:
                step = add(pos,mul(rate/_l,_dir))
                step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
                _dir = dif(step,pos)
                _l = length(_dir)
                i+=1

            direction=_dir

            #print("!!!!!!!!!!!!!!!!!!\n",time_step*growth_rate*2)
            #if (length(closest_direction)<time_step*growth_rate*4):
            if length(closest_direction)<time_step*growth_rate*4:
                projection_coef = dot(direction,normalize(closest_direction))
                if projection_coef > 0:
                    #print("!!!!\nreached_line_bouncing\n")
                    import context
                    #c = context.Context()
                    #c.graphic_context.mark_selection_points([c.canvas._micron_coord_to_stack(p) for p in [_closest_point,pos]])
                    direction=normalize(dif(direction,mul(projection_coef*1.0001,normalize(closest_direction))))
            #else:
            #    closest_direction = dif(_project_to_cone(_closest_point,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan']),pos)
            #    if (length(closest_direction)<7):
            #        projection_coef = dot(direction,normalize(closest_direction))
            #        print("!!!!\nreached_line_bouncing projected",projection_coef,"\n")
            #        if projection_coef > 0:
            #            direction=normalize(dif(direction,mul(projection_coef*1.3,closest_direction)))

                step=add(pos,mul(time_step*growth_rate,direction))
                rate = time_step*growth_rate
                #set position on cone
                step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
                _dir = dif(step,pos)
                _l = length(_dir)
                i=0
                #print(i,_l/rate)
                while (_l/rate<0.999 or _l/rate>1.001) and i<10:
                    step = add(pos,mul(rate/_l,_dir))
                    step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
                    _dir = dif(step,pos)
                    _l = length(_dir)
                    i+=1
                direction=_dir
                #print(i,_l/rate)
            
            # direction2=dif(step,pos)
            # step=add(pos,mul(time_step*growth_rate,normalize(direction2)))
            # step = _project_to_cone(step,center,cone_coeffs['a1'],cone_coeffs['a2'],cone_coeffs['c'],cone_coeffs['tan'])
            
            
            #direction=normalize(dif(step,pos))
            #step = add(pos,mul(time_step*growth_rate,direction))

            line.old_direction=direction
            line.line.append(step)
            if distance2d(center,line.line[-1]) <= distance_treshold[0]*1.0:
                consumed = global_simulation.surface_points.check_if_connected(line.line[-1])
                if consumed:
                    age = (global_simulation.MATURED_AGE)*global_simulation.growth_coeff_development.calculate_surface_frequency(global_simulation.simulation_time)/12-consumed.stage
                    #should not work like that
                    age = (global_simulation.MATURED_AGE-1)*global_simulation.growth_coeff_development.calculate_surface_frequency(global_simulation.simulation_time)/12-consumed.stage
                    line.active=False
                    line.primordium_label=consumed.primordium_label
                    line.start_maturing(maturing_adult_time=age)
        #connection step
        for first in global_simulation.in_progres:
            for second in global_simulation.in_progres:
                if first.active:
                    if not first.active or first==second or second in first.connected_secondary:
                        continue
                    pos=first.line[-1]
                    second_line = list(map(tuple,zip(second.line,second.line[1:])))
                    if length(dif(center,pos))-length(dif(center,second.line[-1]))<0:
                        continue
                    distance,closest_point = _calculate_closest_point_line(pos,second_line)
                    if distance<connection_distance:
                        if closest_point not in second.line:
                            distance = 1000
                            closest_point=min(second.line,key = lambda x: length(dif(x,closest_point)))
                        #choosing best conneciton point
                        #first.line.append(closest_point)
                        index = second.line.index(closest_point)
                        connection_index=-1
                        
                        if length(dif(center,pos))-length(dif(center,closest_point))>0:
                            connection_index=second.line.index(closest_point)
                        else:
                            connection_index=min(len(second.line)-1,second.line.index(closest_point)+1)
                        
                        if connection_index==len(second.line)-1:
                            connection_index=len(second.line)-2
                            _i=-1
                            while length(dif(center,first.line[_i]))-length(dif(center,closest_point))<0:
                                _i-=1
                            if _i<-1:
                                first.line=first.line[:_i+1]
                        
                        print('connecting:',first.age,second.age,'!!!',sep='\n')
                        first.active=False
                        second.connect_secondary(first,connection_index)

                        index=first.connection
                        weight = min(20,max(1,length(dif(second.line[-1],first.line[-1]))/5))
                        #second.old_direction=normalize(add(mul(weight,second.old_direction),first.old_direction))
        #grow from center step
        #points=grow_points(points,center,time_step)
        #grow cone
        global_simulation.points=grow_points_cone(global_simulation.points,center,time_step,**cone_coeffs,**global_simulation.growth_coeffs)
        #consider young pass
        for line in global_simulation.in_progres:
            #all young lines 
            #line.line=grow_points(line.line,center,time_step)
            #grow cone
            line.line=grow_points_cone(line.line,center,time_step,**cone_coeffs,**global_simulation.growth_coeffs)


            #if line.adult:
            #    line.line=grow_points(line.line,center,time_step)
            #    for secondary_line in line.connected_secondary:
            #        secondary_line.line=grow_points(secondary_line.line,center,time_step)
        
        #aging_step
        global_simulation.update_age(time_step)
        global_simulation.make_lines_adult()
#
        

cdef distance_cone_rotated(Point point,Point _center, float a1=0, float a2=0, float c=0, float tan = 2.97):
    cdef Point center, dir
    cdef float distance
    dir = normalize((a1,-1,a2))
    center = add(_center,mul(c,dir))
    a = dif(point,center)
    b = (dir)
    proj_len = dot(a,b)/ length(b)
    distance = (dot(a,a) - dot(a,b) ** 2 / dot(b,b)) ** 0.5
    return abs(proj_len-distance/tan)

def calculate_error(points, center = (14.785861525386506, 24.7665, -17.67426238150849),a1=0,a2=0,c=0,tan = 2.97):
    return sqrt(sum([distance_cone_rotated(point,center,a1,a2,c,tan)**2 for point in points])/len(points))

def calculate_error_rotated(points_weighted, center,a1=0,a2=0,c=0,tan = 2.97):
    return sqrt(sum([distance_cone_rotated(point,center,a1,a2,c,tan)**2*weight for point,weight in points_weighted])/len(points_weighted))

def calculate_closest_point_line(point,line):
    return _calculate_closest_point_line(point,line)

def test_distance2d(Point a, Point b):
    return distance2d(a,b)

cdef _fast_distance(Point point,line, int skip):
    cdef float dist = length(dif(point,line[-1][1]))
    cdef index=-1

    for i,(a,b) in enumerate(line):
        if i%skip==0:
            new_distance=length(dif(a,point))
            if new_distance<dist:
                dist=new_distance
                index=i
    
    return dist,index

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _fill_array(np.ndarray[DTYPE_t, ndim=2] array, age, line, float  pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef, float age_cut_off_coef, float peak_coef):
    cdef Point direction=(0,0,0)
    cdef Point point,other_pos
    cdef IndexPoint point_index
    cdef float distance, strength, _strength, r
    #cdef np.ndarray[DTYPE_t, ndim=3] distance_array
    #cdef _map[IndexPoint,float] in_progres
    _time = time.time()
    cdef temp_dir,

    cdef int _i,j,i,size,index
    _i=0
    size=len(line)


    for i in range(0,array.shape[0]):
        if i%10==0:
            print(i,time.time()-_time)
            _time=time.time()
        for j in range(0,array.shape[1]):
            point_index=(i,0,j)
            point=(float(i),0,float(j))

            distance,index = _fast_distance(point,line,10)
            if distance<(neg_range+pos_range+20+age):
                distance,other_pos = _calculate_closest(point,line)
                
            strength=-_calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
            array[point_index[0],point_index[2]]=strength
    

def fill_array((int,int) shape, age, line, float pos_range=0, float neg_range=0, float inhibition_coef=0, float attraction_coef=0,float age_coef=0, float age_cut_off_coef=0, float straight_coef=0, float peak_coef=0,inertia=0):
    cdef np.ndarray[DTYPE_t, ndim=2] array
    array = np.zeros(shape,dtype='float')
    _fill_array(array,age,line,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
    return array


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _fill_array_cone(np.ndarray[DTYPE_t, ndim=3] array,lines,young_lines, float start_x,float start_y, float step,
                      float pos_range, float neg_range, float inhibition_coef, float attraction_coef,float age_coef,float age_cut_off_coef, float straight_coef, float peak_coef, 
                      Point center,float a1, float a2,float c,float tan):
    cdef int i,j
    cdef Point pos,to_center,direction,_center
    cdef int x,y
    cdef float _step

    
    _center = (center[0],0,center[2])
    _time = time.time()
    for i in range(array.shape[0]):
        if i%10==0:
            print(i,time.time()-_time)
            _time=time.time()
        for j in range(array.shape[1]):
            pos = (i*step+start_x,0,j*step+start_y)
            # pos=_project_to_cone(pos, center, a1 , a2 ,c, tan)
            direction = (0,0,0)
            # for line,age in lines:
                # distance,other_pos = _calculate_closest(pos,line)
                # other_pos=_project_to_cone(other_pos, center, a1 , a2 ,c, tan)
                # temp_dir=normalize(dif(pos,other_pos))
                # strength = _calculate_strength(distance,age,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,peak_coef)
                # if strength>0:
                #     array[i,j,2]+=strength
                # else:
                #     array[i,j,0]-=strength
                # if length(dif(pos,other_pos))<5:
                #     array[i,j,1]+=1
            direction,distance=_predict_step_get_distance(pos,_center,lines,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef, peak_coef)
            if direction[1]>0.001:
                print(direction[1])



            array[i,j,0]=direction[0]
            array[i,j,2]=direction[2]
            if distance<2:
                array[i,j,1]=1

            #direction = (_predict_step(pos,center, lines, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef,age_cut_off_coef, straight_coef, peak_coef))
            #to_center=normalize(dif(center,pos))
            #to_center=normalize((to_center[0],0,to_center[2]))
            #cos = dot(to_center,(1,0,0))
            #sin = cross(to_center,(1,0,0))[2]
            #center_coef = dot(to_center,direction)

            #direction = (dif(direction,mul(center_coef,to_center)))
        for young_line in young_lines:
            for a,b in young_line:
                l=length(dif(a,b))
                for _ in range(int(ceil(l+1))):
                    _step = float(_)/ceil(l)
                    x = ((a[0]*_step+(1-_step)*b[0])-start_x)/step
                    y = ((a[2]*_step+(1-_step)*b[2])-start_y)/step
                    for i in range(max(0,x-2),min(array.shape[0],x+2)):
                        for j in range(max(0,y-2),min(array.shape[0],y+2)):
                            array[i,j,1]=-1
            distance = min(distance,_calculate_closest(pos,young_line)[0])




def fill_array_cone_from_lines(int scale, prediction_lines,young_lines,float margin=50.,
                    float pos_range=0, float neg_range=0, float inhibition_coef=0, float attraction_coef=0,float age_coef=0, float age_cut_off_coef=0, float peak_coef=0,float straight_coef=0,float inertia=0,
                    center=0,float a1=0, float a2=0,float c=0,float tan=0):
    cdef np.ndarray[DTYPE_t, ndim=3] array
    array = np.zeros((scale,scale,3),dtype='float')
    
    prediction_lines = [ ([((a[0],0,a[2]),(b[0],0,b[2])) for a,b in line],age) for line,age in prediction_lines]
    young_lines = [ [((a[0],0,a[2]),(b[0],0,b[2])) for a,b in line] for line in young_lines]


    c_min = [10000,1000]
    c_max = [-1000,-1000]
    for line,_ in prediction_lines:
        for a,b in line:
            c_min[0]=min(c_min[0],a[0],b[0])
            c_min[1]=min(c_min[1],a[2],b[2])

            c_max[0]=max(c_max[0],a[0],b[0])
            c_max[1]=max(c_max[1],a[2],b[2])
    for line in young_lines:
        for a,b in line:
            c_min[0]=min(c_min[0],a[0],b[0])
            c_min[1]=min(c_min[1],a[2],b[2])

            c_max[0]=max(c_max[0],a[0],b[0])
            c_max[1]=max(c_max[1],a[2],b[2])
    c_min_x=c_min[0]-margin
    c_min_y=c_min[1]-margin

    c_max_x=c_max[0]+margin
    c_max_y=c_max[1]+margin

    c_max=max(c_max)+margin
    c_min=min(c_min)-margin
    step_size=max(c_max_x-c_min_x,c_max_y-c_min_y)/scale
    print(c_min_x,c_min_y,c_max_x,c_max_y)
    print("range",c_min,c_max)
    _fill_array_cone(array, prediction_lines,young_lines, c_min_x,c_min_y, step_size,
                      pos_range, neg_range, inhibition_coef, attraction_coef, age_coef, age_cut_off_coef, straight_coef, peak_coef, 
                      center, a1, a2, c, tan)
    return (array,c_min_x,c_min_y,c_max_x,c_max_y)
    

def fill_array_cone(int scale, global_simulation,float margin=50,
                    float pos_range=0, float neg_range=0, float inhibition_coef=0, float attraction_coef=0,float age_coef=0, float age_cut_off_coef=0, float peak_coef=0,float straight_coef=0,float inertia=0,
                    center=0,float a1=0, float a2=0,float c=0,float tan=0):
    cdef np.ndarray[DTYPE_t, ndim=3] array
    array = np.zeros((scale,scale,3),dtype='float')        
    prediction_lines = []
    young_lines = []
    for v in global_simulation.lines:
        prediction_lines.append(([((a[0],0,a[2]),(b[0],0,b[2])) for a,b in graph_to_lines(v.skeleton,global_simulation.points)],(v.age)))
    for young_line in global_simulation.in_progres:
        young_lines.append([((a[0],0,a[2]),(b[0],0,b[2])) for a,b in zip(young_line.line,young_line.line[1:])])
        # prediction_lines.append(([(_project_to_cone(a, center, a1 , a2 ,c, tan),_project_to_cone(b, center, a1 , a2 ,c, tan)) for a,b in graph_to_lines(v.skeleton,global_simulation.points)],(v.age)))

    c_min = [0,0]
    c_max = [0,0]
    for line,_ in prediction_lines:
        for a,b in line:
            c_min[0]=min(c_min[0],a[0],b[0])
            c_min[1]=min(c_min[1],a[2],b[2])

            c_max[0]=max(c_max[0],a[0],b[0])
            c_max[1]=max(c_max[1],a[2],b[2])
    c_min=min(c_min)-margin
    c_min_x=c_min[0]-margin
    c_min_y=c_min[1]-margin

    c_max_x=c_max[0]-margin
    c_max_y=c_max[1]-margin
    c_max=max(c_max)+margin
    step_size=max(c_max_x-c_min_x,c_max_y-c_min_y)/scale
    _fill_array_cone(array, prediction_lines,young_lines, c_min_x,c_min_y, step_size,
                      pos_range, neg_range, inhibition_coef, attraction_coef, age_coef, age_cut_off_coef, straight_coef, peak_coef, 
                      center, a1, a2, c, tan)
    return array


cdef float _line_segment_length(line):
    cdef result = 0
    cdef Point p1, p2
    for p1,p2 in line:
        result += length(dif(p1,p2))
    
    return result

#line: list[(Point,Point)]
def line_segment_length(line):
    return _line_segment_length(line)

cdef _calculate_average_distances(lines):
    result = []
    min_size=min(len(line) for line in lines)
    for i in range(min_size):
        c=[]
        for x in lines:
            c.append(x[int(round(i/min_size*len(x)))])
        result.append(sum(c)/len(c))
    return result

cdef _calculate_average_line(lines):
    result = []
    min_size=min(len(line) for line in lines)
    for i in range(min_size):
        c=[]
        for x in lines:
            c.append(x[int(round(i/min_size*len(x)))])
        point = []
        for j in range(len(c[0])):
            point.append(sum(p[j] for p in c)/len(c))
        result.append(tuple(point))
    return result

def calculate_average_distances(lines):
    return _calculate_average_distances(lines)

def calculate_average_line(lines):
    return _calculate_average_line(lines)

def _distance_lines_iv(a,b):
    result = []
    sign = 1
    if len(a)<len(b):
        sign = -1
        a,b=b,a 

    min_size=len(a)
    max_size=len(b)
    for i,p in enumerate(a):
        index_f = int(floor(i/min_size*max_size))
        index_c = min(int(ceil(i/min_size*max_size)),max_size-1)
        w = i/min_size*max_size-index_f
        p2 = b[index_f]*(1-w)+b[index_c]*w 
        result.append(sign*(p-p2))
    return result


def distance_lines_iv(a,b):
    return _distance_lines_iv(a,b)


def point_on_path_distance_reversed(line,start,distance):
    index = start
    curr_distance = 0
    while distance>curr_distance and index>0:
        curr_distance+=length(dif(line[index],line[index-1]))
        index-=1
    return line[index]

