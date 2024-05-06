from contextvars import Context
from typing import List,Optional,Dict,Tuple,Set
from numpy.lib.function_base import average
from pyrr import Vector3, Matrix33,vector3,matrix33
from itertools import product
import skeleton
import numpy as np
import generate_line
import context


def implicit_elipsoid(x,y,z,x1=0,y1=0,z1=0,a=1,b=1,c=1):
    return (x-x1)**2 + (y-y1)**2*a/b + (z-z1)**2*a/c - a

# def distance_cone(x,y,z,x1=0,y1=0,z1=0,alpha = np.pi/2):
#     vec = np.array((x-x1,y-y1,z-z1))
#     l = np.linalg.norm(vec)
#     vec = vec/l
#     c = vec@np.array((0,-1,0))
#     actual_angle = np.arccos(c)
#     dist = abs(np.sin(alpha-actual_angle)*l)
#     return dist
    
# def draw_cone(coeffs,size=300,v=20):
#     point = Vector3((coeffs['x1'],coeffs['y1'],coeffs['z1']))
#     aaa = c.canvas._micron_coord_to_stack((coeffs['x1'],coeffs['y1'],coeffs['z1']))
#     x=Vector3((0,-size,0))
#     Matrix33.from_x_rotation(coeffs['alpha'])
#     x=x@Matrix33.from_x_rotation(coeffs['alpha'])
#     ob = [tuple(x@Matrix33.from_y_rotation(np.pi*2/v*i))+Vector3(point) for i in range(v)]
#     ob2=list(map(c.canvas._micron_coord_to_stack,ob))
#     c.graphic_context.mark_temporary_paths([ob2+[ob2[0]] ]+[[aaa,i] for i in ob2])

def distance_cone(x,y,z,x1=0,y1=0,z1=0,tan = 2.97):
    r = ((x-x1)**2+(z-z1)**2)**0.5
    if tan == 0:
        return 10_000
    return abs((y1-r/tan)-y)

def distance_cone_rotated(x,y,z,center,a1=0,a2=0,c=0,tan = 2.97):
    #center = Vector3((3.369800998809006, 35.510999999999996, -58.9715174791578))
    point = Vector3((x,y,z))
    dir = Vector3(vector3.normalise(Vector3((a1,-1,a2))))
    center = Vector3(center) + dir*c
    a = (point-center)
    b = (dir)
    proj_len = (a | b)/ b.length
    distance = ((a| a) - (a | b) ** 2 / (b | b)) ** 0.5
    return abs(proj_len-distance/tan)

def create_micron_skeleton():
    c = context.Context()
    global micron_skeleton
    canvas = c.canvas
    micron_skeleton = {canvas._stack_coord_to_micron(k):set(canvas._stack_coord_to_micron(x) for x in v) for k,v in canvas.skeleton_graph.items()}
    return  micron_skeleton
def calculate_weight(p,micron_skeleton):
    weight = 0
    point = Vector3(p)
    for x in micron_skeleton[p]:
        weight += (point-Vector3(x)).length
    weight = weight/2
    return weight

def distance_cone_rotated_weighted(x,y,z,center,a1=0,a2=0,c=0,tan = 2.97):
    create_micron_skeleton()
    #center = Vector3((3.369800998809006, 35.510999999999996, -58.9715174791578))
    weight = calculate_weight((x,y,z))/10
    point = Vector3((x,y,z))
    dir = Vector3(vector3.normalise(Vector3((a1,-1,a2))))
    center = center + dir*c
    a = (point-center)
    b = (dir)
    proj_len = (a | b)/ b.length
    distance = ((a| a) - (a | b) ** 2 / (b | b)) ** 0.5
    return abs(proj_len-distance/tan)*weight

def draw_cone(c,coeffs,size=300,v=20):
    point = Vector3((coeffs['x1'],coeffs['y1'],coeffs['z1']))
    aaa = c.canvas._micron_coord_to_stack((coeffs['x1'],coeffs['y1'],coeffs['z1']))
    x=Vector3((size,-size/coeffs['tan'],0))
    ob = [tuple(x@Matrix33.from_y_rotation(np.pi*2/v*i))+Vector3(point) for i in range(v)]
    ob2=list(map(c.canvas._micron_coord_to_stack,ob))
    c.graphic_context.mark_temporary_paths([ob2+[ob2[0]] ]+[[aaa,i] for i in ob2])

def draw_cone_rotation(c,coeffs,center,size=300,v=20):
    _center=center
    center = Vector3(center)
    dir = vector3.normalise(Vector3((coeffs['a1'],-1,coeffs['a2'])))
    center = center + dir*coeffs['c']
    side = vector3.normalise(Vector3((1,0,0))^dir)
    print(side)
    x = size * vector3.normalise(side*coeffs['tan']+dir)
    ob = [tuple(x@matrix33.create_from_axis_rotation(dir,np.pi*2/v*i))+center for i in range(v)]
    ob2=list(map(c.canvas._micron_coord_to_stack,ob))
    _center = c.canvas._micron_coord_to_stack(tuple(_center))
    aaa = c.canvas._micron_coord_to_stack(tuple(center))
    bbb = c.canvas._micron_coord_to_stack(tuple(dir*size+center))
    c.graphic_context.mark_temporary_paths([ob2+[ob2[0]] ]+[[aaa,i] for i in ob2]+[[aaa,bbb]]+[[_center,aaa]])

def translate_from_cone_coordinate(coeffs,center,r,alpha):
    _center=center
    center = Vector3(center)
    dir = vector3.normalise(Vector3((coeffs['a1'],-1,coeffs['a2'])))
    center = center + dir*coeffs['c']
    side = vector3.normalise(Vector3((1,0,0))^dir)
    #print(side)
    result = vector3.normalise(side*coeffs['tan']+dir)*r
    result = result@matrix33.create_from_axis_rotation(dir,alpha/180*np.pi)+center
    return result

def translate_to_cone_coordinate(coeffs,center,point):
    pp = Vector3(generate_line.project_to_cone(tuple(point),center,**coeffs))
    dir = vector3.normalise(Vector3((coeffs['a1'],-1,coeffs['a2'])))
    center = center + dir*coeffs['c']
    pp = pp-Vector3(center)
    side = vector3.normalise(Vector3((1,0,0))^dir)
    parallel = vector3.normalise(pp-vector3.dot(pp,dir)*dir)
    
    import math
    angle = math.acos(vector3.dot(parallel,side))/np.pi*180
    if vector3.dot(vector3.cross(side,parallel),dir)>0:
        angle = angle
    else:
        angle = 360 - angle
    return pp.length,angle

def generate_cone_points(coeffs,center,radius_step,angular_density,start,end):
    points=[]
    r=start
    step=0
    while r<=end:
        points_number = max(1,int(2*np.pi*r/angular_density))
        angle_step = 360/points_number
        for i in range(points_number):
            #angle=np.pi**10*step+angle_step/2+i*angle_step
            angle=angle_step/2+i*angle_step
            points.append(translate_from_cone_coordinate(coeffs,center,r,angle))
        r+=radius_step
        step+=1
    return points




    

# usefull call
# for i in range(3,9):
#     points=points_from_indices(c,range(-3,i))
#     find_coeff_shape(points,distance_cone_rotated,center,coeffs)

def calculate_error(points: List[Tuple[float,float,float]],function,coeffs,center):
        return sum((function(*point,center,**coeffs)**2 for point in points))/len(points)

def find_coeff_shape(points: List[Tuple[float,float,float]],center,coeffs,steps=1000,MAX_TRIES=30,
#                        min_step={'x1':0.1,'y1':0.1,'z1':0.1,'a':0.03,'b':0.03,'c':0.03}):
                        min_step={'a1':.1,'a2':.1,'c':30,'tan':0.1},use_weights = True
                        ):
    micron_skeleton = create_micron_skeleton()

    if use_weights:
        points_weighted = [(point,calculate_weight(point,micron_skeleton)) for point in points]
    else:
        points_weighted = [(point,1) for point in points]

    tries_table=list([(1/abs(i)**2) for i in range(1,MAX_TRIES+3,1) if i!=0])
    
    tries_values=[]
    #error=calculate_error(points,function,coeffs)
    print(coeffs)
    last_tries=1
    error=generate_line.calculate_error_rotated(points_weighted,center,**coeffs)
    for i in range(steps):
        tries=max(0,last_tries-1)
        while tries<MAX_TRIES:
            new_errors=[]
            for direction in product((-1,0,1), repeat=len(coeffs)):
                new_coeffs={}
                for (k,v),d in zip(min_step.items(),direction):
                    new_coeffs[k]=v*d*tries_table[tries]+coeffs[k]
                #new_errors.append((calculate_error(points,function,new_coeffs),new_coeffs))
                new_errors.append((generate_line.calculate_error_rotated(points_weighted,center,**new_coeffs),new_coeffs))
            tries+= 1
            
            test = min(new_errors,key=lambda x: x[0])
            if test[0]<error:
                error=test[0]
                coeffs=test[1]
                last_tries = tries
                break
        #print('error:',error)
        if tries==MAX_TRIES:
            print("Max tries reached")
            break
        else:
            tries_values.append(tries_table[tries])
    if tries_values:
        print('tries results', min(tries_values), max(tries_values))
    #print(coeffs)
    print('error:',error)
    return coeffs,error

def prepare_points(c):
    _points=c.skeleton_distances.get_points()
    points = list(map(c.canvas._stack_coord_to_micron,_points))
    return points

def prepare_paths(c):
    _points=c.skeleton_distances.get_points()
    points=[]
    for a,b in product(_points,repeat=2):
        points.extend(skeleton.find_path_between_nodes(c.canvas.skeleton_graph,a,b,True))
    return list(map(c.canvas._stack_coord_to_micron,points))

def points_from_labels(c,labels):
    points=[]
    for label in labels:
        point = c.data_context.get_point_from_label(label)
        points.extend(skeleton.find_compact_component(c.canvas.skeleton_graph,point))
    return list(map(c.canvas._stack_coord_to_micron,points))


def points_from_indices(c,indices):
    points=[]
    for index in indices:
        label = f"{index}"
        point = c.data_context.get_point_from_label(label)
        points.extend(skeleton.find_compact_component(c.canvas.skeleton_graph,point))
    return list(map(c.canvas._stack_coord_to_micron,points))
    
def get_center(c):
    names = ["c","center","CENTER","C"]
    center = None
    for name in names:
        center  = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label(name))
        if center is not None:
            return center 




# {'x1': 6.600000000000032, 'y1': -41.60000000000004, 'z1': -46.90000000000054, 'a': 11490.99999999735, 'b': 8544.759999998, 'c': 11490.99999999735}
# coeffs = {'x1':1,'y1':1,'z1':1,'a':1,'b':1,'c':1}
# coeffs = dict(x1=0,y1=0,z1=0,alpha = np.pi/2.3 )

# coeffs = find_coeff_shape(points,distance_cone,coeffs,steps=10)
# coeffs = find_coeff_shape(points2,distance_cone,coeffs,steps=10)
# {'x1': -0.5999999999999864, 'y1': 13.8, 'z1': -54.4000000000005, 'alpha': 7.529714635730041}
# coeffs = {'x1': -0.5999999999999864, 'y1': 13.8, 'z1':  -28.88400856122013, 'tan': 2.9750237084819764}
# {'x1': 4.800000000000015, 'y1': 13.8, 'z1': -70.48400856122063, 'tan': 2.8550237084819767}
# all error 55 {'x1': 2.2000000000000135, 'y1': 13.600000000000001, 'z1': -74.08400856122071, 'tan': 2.9550237084819764}

# coeffs = dict(a1=0,a2=0,c=0,tan = 2.97)
# coeffs = find_coeff_shape(points,distance_cone_rotated,coeffs,min_step={'a1':0.005,'a2':0.005,'c':0.1,'tan':0.05},steps=100,MAX_TRIES=10)
# points = [c.canvas._stack_coord_to_micron(point) for point in c.canvas.skeleton_graph.keys()]
# points = [(-39.956211843021194, -10.573499999999996, 18.2932054221061), (-150.67824466103175, -34.1145, -156.93644651596276), (-10.109402996427056, -30.922499999999992, 56.805216837066254), (-28.88400856122013, -56.0595, 108.79643224726254), (51.02841512482225, -18.952499999999993, 28.402608418533127), (82.80082454216436, -53.26649999999999, 82.8008245421644), (47.65861412601327, -39.70049999999999, 92.42882739590445), (58.24941726512726, -55.66049999999999, 120.83143581443758), (104.9452311057665, -32.9175, 2.4070007134349822), (151.15964480371875, -50.074499999999986, 17.81180527941907), (175.22965193806883, -71.62049999999999, 75.57982240185939), (133.82923966698667, -50.074499999999986, 66.43321969080631), (155.97364623058877, -35.7105, -72.21002140305035), (203.15086021391505, -52.86749999999999, -72.21002140305035), (79.91242368604242, -11.770499999999995, -79.91242368604239), (139.60604137923067, -30.5235, -104.9452311057665), (182.93205422106087, -39.3015, -118.90583524368957), (62.58201854931032, -28.528499999999994, -164.6388487989548), (102.53823039233144, -39.3015, -187.2646555052439), (38.512011414960185, -34.51349999999999, -213.26026321034203), (24.070007134350103, -40.897499999999994, -258.9932767656073), (-7.702402282992011, -36.1095, -214.70446363840304), (-27.921208275846126, -7.3814999999999955, -100.1312296788965), (-43.32601284183021, -20.5485, -137.68044080848267), (-51.509815267509225, -37.705499999999994, -212.77886306765504), (0.9628002853740054, -52.06949999999999, -257.06767619485925), (-109.27783238994955, -19.750500000000006, -111.68483310338453), (-147.78984380490974, -38.503499999999995, -142.0130420926657), (-54.398216123631244, -15.361499999999998, -98.6870292508355), (-77.50542297260738, -24.937499999999993, -137.19904066579568), (-93.87302782396546, -17.7555, -50.54701498213525), (-137.68044080848267, -28.9275, -86.17062554097342), (-230.59066834707414, -56.857499999999995, -51.50981526750925), (-223.8510663494561, -64.43849999999999, -22.144406563602093), (-96.76142868008748, -14.563500000000001, 11.553603424488065), (-153.56664551715377, -56.857499999999995, 27.921208275846155), (-193.04145721748796, -58.45349999999999, 40.919012128395195), (93.39162768127842, -10.174500000000002, -33.698009988090156), (149.23404423297075, -41.29649999999999, -21.66300642091509), (19.256005707480078, -3.7905, -109.27783238994952), (19.73740585016708, -10.174500000000002, -138.16184095116967), (-106.8708316765145, -59.25149999999999, 64.98901926274533)]

coeffs = dict(a1=0,a2=0,c=0,tan = 2.97)
# coeffs = {'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}
# coeffs = find_coeff_shape(points,distance_cone_rotated,coeffs,min_step={'a1':0.005,'a2':0.005,'c':0.1,'tan':0.05},steps=10000,MAX_TRIES=10)
# xp48_90 coeffs = {'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}
# xp41_dr35 coeffs = {'a1': 0.034999999999999865, 'a2': 0.040000000000000306, 'c': 22.800000000000228, 'tan': 2.4199999996166066}
# points = [c.canvas._stack_coord_to_micron(point) for point in c.canvas.skeleton_graph.keys()]


# batch shape calculation
# step 1 trimming
def get_points_skeleton(c:context.Context):
    result = {}
    for i in range(-10,20):
        end = c.data_context.get_point_from_label(f"{i}")
        if end is not None:
            result[i]=set(skeleton.find_compact_component(c.canvas.skeleton_graph,end).keys())
            start = c.data_context.get_point_from_label(f"{i}x")
            if start is None:
                start = c.data_context.get_point_from_label(f"{i}e")
            if start is not None:
                result[i] = result[i] - set(x[1] for x in skeleton.find_primary(c.canvas.skeleton_graph,start,end))
                result[i].add(start)
            result[i] = set(c.canvas._stack_coord_to_micron(x) for x in result[i])
    return result

def remove_extra_points(c):
    for i in range(-10,20):
        end = c.data_context.get_point_from_label(f"{i}")
        if end is not None:
            start = c.data_context.get_point_from_label(f"{i}x")
            if start is None:
                start = c.data_context.get_point_from_label(f"{i}e")
            if start is not None and end!=start:
                print(i)
                c.skeleton_distances.remove_path_pass_label(start,end)
    

def get_points_skeleton_marking():
    c=context.Context()
    result = []
    for i in range(-10,20):
        end = c.data_context.get_point_from_label(f"{i}")
        if end is not None:
            r=set(skeleton.find_compact_component(c.canvas.skeleton_graph,end).keys())
            start = c.data_context.get_point_from_label(f"{i}x")
            if start is not None:
                r = r - set(x[1] for x in skeleton.find_primary(c.canvas.skeleton_graph,start,end))
                r.add(start)
            result.extend(list(r))
    return result

def perform_calculation(c,points_dict,**kwargs):
    center = get_center(c)
    start = min(points_dict.keys())
    end = max(points_dict.keys())
    result = {}

    for i in range(min(max(3,start+1),end),end+1):
        print(i)
        indices = list(range(start,i+1))
        if i not in points_dict:
            continue
        points = sum((list(points_dict.get(x,[])) for x in indices),start=[])
        coeffs = dict(a1=0.1,a2=0.1,c=1,tan = 4)
        result[i]=find_coeff_shape(points,center,coeffs,**kwargs)
    # i=5
    # print(i)
    # indices = list(range(start,i+1))
    # points = sum((list(points_dict.get(x,[])) for x in indices),start=[])
    # coeffs = dict(a1=0.1,a2=0.1,c=1,tan = 4)
    # result=find_coeff_shape(points,center,coeffs,**kwargs)
    return result

import pickle
import glob

def run_cone_calculation(c,path,**kwargs):
    import time
    t = time.time()
    create_micron_skeleton()
    save = pickle.load(open(path,"rb"))
    c.data_context.load_data(save)
    pd = get_points_skeleton(c)
    result = perform_calculation(c,pd,**kwargs)
    print("time passed:", time.time()-t)
    return result
#run_cone_calculation(c,"C:\\Users\\Andrzej\\Desktop\\2send_1\\3_CONEshape\\mediumSAM\\xp13_dr9_cone.ske",steps=300,MAX_TRIES=40,min_step={'a1':.1,'a2':.1,'c':30,'tan':0.1})
#({'a1': 0.17142716006313477, 'a2': 0.09029554358740732, 'c': 59.498313096425015, 'tan': 3.745230392567691}, 18.84927887244293)
# pd = get_points_skeleton(c)
# perform_calculation(c,pd)

def cone_calculation():
    c=context.Context()
    for folder in ["smallSAM","mediumSAM","bigSAM"]:
        path = f'C:\\Users\\Andrzej\\Desktop\\2send_1\\3_CONEshape\\{folder}\\**'
        results = {}
        for filepath in glob.iglob(path):
            name = filepath.split("\\")[-1]
            print()
            print(name)
            if ".ske" not in filepath:
                continue
            result=run_cone_calculation(c,filepath,steps=300,MAX_TRIES=40,min_step={'a1':.1,'a2':.1,'c':30,'tan':0.1},use_weights=False)
            results[name]=result
        pickle.dump(results,open(f'C:\\Users\\Andrzej\\Desktop\\2send_1\\3_CONEshape\\{folder}_dump_no_weights',"wb"))

def calculate_offset():
    c = context.Context()
    gs = c.data_context.global_simulation
    points = [c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label(label)) for label in ['-1', '0', '1', '2', '3', '4', '5','6','7'] if c.data_context.get_point_from_label(label) is not None]

    angles = [translate_to_cone_coordinate(gs.cone_coeffs,gs.center,point)[1] for point in (points)]
    #print(angles)
    x = [(a+360-b)%360 for a,b in zip(angles[1:],angles)]
    #print(x)
    average_angle = sum(x)/len(x)
    #print(average_angle)
    offsets = [(360+angle-(i*average_angle)%360)%360 for i,angle in enumerate(angles)]
    #print(offsets)
    print(sum(offsets)/len(offsets))
    print(sum(offsets)/len(offsets))
    print(x)
    return(sum(offsets)/len(offsets),average_angle)

def get_initial_angle():
    c = context.Context()
    gs = c.data_context.global_simulation
    offset,filo_angle = calculate_offset()
    i=0
    # while True:
    #     try:
    #         new_point=c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label(str(i)))
    #     except:
    #         break
    #     point-new_point
    #     i-=1
    try:
        point = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label("-1"))
    except:
        point = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label("-2"))
    r,angle = translate_to_cone_coordinate(gs.cone_coeffs,gs.center,point)
    return round((angle-offset)/filo_angle)*filo_angle+offset

def form_cone(tan):
    #tangens of half aperture angle
    def ff(p):
        c = context.Context()
        center = c.data_context.get_point_from_label("c")
        center = Vector3(c.canvas._stack_coord_to_micron(center))
        point = Vector3(c.canvas._stack_coord_to_micron(p))
        l = (point-center).length/tan
        point = point-Vector3((0,l,0))
        point = c.canvas._micron_coord_to_stack(tuple(point))
        return point
    return ff

def move_points(f):
    c = context.Context()
    c.data_context.values["point"] = [f(p) for p in c.data_context.values["point"]]
    c.data_context.values["label"] = [f(p) for p in c.data_context.values["label"]]
    c.data_context.values["line"] = [(f(a),f(b)) for a,b in c.data_context.values["line"]]
    c.data_context.labels["point"] = {f(k):v for k,v in c.data_context.labels["point"].items()}
    c.data_context.labels["line"] = {(f(a),f(b)):v for (a,b),v in c.data_context.labels["line"].items()}
    c.data_context.labels["label"] = {f(k):v for k,v in c.data_context.labels["label"].items()}
    c.canvas.skeleton_graph = {f(k):{f(x) for x in v} for k,v in c.canvas.skeleton_graph.items()}
    c.canvas.reload_skeleton_graph(c.canvas.skeleton_graph)

def replace_dict(dictionary : Dict):
    def f(x):
        return dictionary.get(x,x)
    return f


def find_point_on_line_in_distance(_point,_center,distance):
    c : Context = context.Context()
    point = Vector3(c.canvas._stack_coord_to_micron(_point))
    center = Vector3(c.canvas._stack_coord_to_micron(_center))
    dir = point-center
    l = (dir.x**2+dir.z**2)**0.5
    new_point = center+dir*(distance/l)
    return c.canvas._micron_coord_to_stack(tuple(new_point))
