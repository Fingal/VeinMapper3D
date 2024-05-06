from itertools import product
from multiprocessing import connection
import pdb
from turtle import pd
from skeleton import *
import generate_line
from typing import List, Literal,Optional,Dict,Tuple,Set
import time
import random
from pyrr import vector3,Vector3
from math import floor
import math
import context


MATURING_TIME=0
MATURED_AGE = 1

CENTER = (527.0, 563.0, 100.0)
color_index=0
def get_color_index():
    global color_index
    color_index+=1
    return color_index
def normalize(vec):
    return vec/np.linalg.norm(vec)

def refine(path,indices,max_angle):
    result=[]
    for start,end in (zip(indices,indices[1:])):
        result.append(start)
        ps = np.array(path[start])
        pe = np.array(path[end])
        if end-start>1:
            best = max(((i,math.acos(np.dot(normalize(ps-np.array(p)),normalize(pe-np.array(p)))) ) 
                            for i,p in enumerate(path[start+1:end])), key=lambda x: x[1])
            if best[1]<max_angle:
                result.append(best[0]+1+start)

    result.append(end)
    return result


def simplified_path(path:Path,junctions:Set[Point]=set(),max_angle=3.1,refine_steps=2):
    last_point = np.array(path[1])
    vec = (np.array(path[0])-last_point)
    vec=normalize(vec)
    indices=[0,1]
    for i,point in enumerate(path[2:]):
        index=i+2
        other_vec = np.array(point)-last_point
        other_vec = normalize(other_vec)
        angle = math.acos(np.dot(vec,other_vec))
        if angle<max_angle or point in junctions:
            indices.append(index)
            vec = (np.array(last_point)-np.array(point))
            vec=normalize(vec)
            last_point = np.array(point)
    if indices[-1] != len(path)-1:
        indices.append(len(path)-1)
    for i in range(refine_steps):
        indices=refine(path,indices,max_angle)
    
    
    return [path[i] for i in indices]
    




def calculate_angles(center,a,b):
    
    center = (center)
    center = np.array((center[0],center[2]))

    points=[a,b]

    points = [point for point in points]
    points = [np.array((point[0],point[2])) for point in points]
    points = [(point-center)/np.linalg.norm(point-center) for point in points]

    return [math.acos(np.dot(a,b))/math.pi*180 for a,b in zip(points,points[1:])][0]   

def add_distance(center,distance,skeleton_graph):
    distanced={}
    for point in skeleton_graph.keys():
        _c=np.array(c.canvas._stack_coord_to_micron(center))
        p=np.array(c.canvas._stack_coord_to_micron(point))
        vec = p-_c
        dist=np.linalg.norm(vec)
        new_point=_c+vec*(dist+distance)/dist
        new_point=c.canvas._micron_coord_to_stack(tuple(new_point))
        distanced[point]=new_point
    graph={distanced[k]:{distanced[p] for p in v} for k,v in skeleton_graph.items()}
    return graph


def from_label(label):
    points = c.data_context.get_points_from_label(label)
    if points[1][-1]>points[0][-1]:
        points=points[::-1]
    return points


def _trim_skeleton(skeleton,trim_factor=8):
    def cos(a,b,c):
        return (a-c)@(b-c)/(np.linalg.norm(a-c)*np.linalg.norm(b-c))

    def trim_path(path,factor):
        result = []
        pos = 0
        while pos<len(path):
            step = 1
            l = 0
            while l<factor and pos+step<len(path):
                l+=sum((a-b)**2 for a,b in zip(path[pos+step-1],path[pos+step]))**0.5
                step+=1
            s = path[pos:pos+step]
            start = np.array(s[0])
            end = np.array(s[-1])
            result.append(s[0])
            if len(s)>1:
                if len(s)>2:
                    middle = max(s[1:-1],key=lambda x: cos(start,end,np.array(x)))
                    result.append(middle)

                result.append(s[-1])

            pos = pos+step
            if pos == len(path)-1:
                if path[-1] != result[-1]:
                    result.append(path[-1])
                pos+=1

        return result

    paths = get_all_paths(skeleton)
    trimmed_paths = [trim_path(path,trim_factor) for path in paths]
    result = {}
    for path in trimmed_paths:
        for a,b,c in zip(path,path[1:],path[2:]):
            if a in result:
                result[a].add(b)
            else:
                result[a]={b}
            if b in result:
                result[b].add(a)
                result[b].add(c)
            else:
                result[b]={a,c}
            if c in result:
                result[c].add(b)
            else:
                result[c]={b}
    return result


class GrowthCoeffDevelopment:
    def __init__(self,growth_coeffs_list:List[dict],distances_to_center: List[float],
                      coeffs_development_times : List[float],
                      n5_ages : List[float],n8_ages : List[float],
                      surface_frequency:List[float],growth_rate=None
                      ) -> None:
        self.growth_coeffs_list=growth_coeffs_list
        self.coeffs_development_times=coeffs_development_times
        self.distances_to_center=distances_to_center
        self.n5_ages=n5_ages
        self.n8_ages=n8_ages
        self.surface_frequency=surface_frequency
        self.growth_rate=growth_rate
        self.coeffs_list=None

    def calculate_coeffs(self,time: float):
        i=0
        while i<len(self.coeffs_list)-1:
            if self.coeffs_development_times[i]>time:
                c1=self.coeffs_list[i]
                c2=self.coeffs_list[i+1]
                ratio = time/self.coeffs_development_times[i]
                return {k:c1[k]*(1-ratio)+c2[k]*ratio for k in c1.keys()}
            else:
                time=time-self.coeffs_development_times[i]
            i+=1
        return self.coeffs_list[-1]



    def _calculate_approx_float(self,nX_ages,time: float):
        i=0
        steps_time = 0 
        while i<min(len(self.coeffs_development_times),len(nX_ages)-1):
            if self.coeffs_development_times[i]>time:
                c1=nX_ages[i]
                c2=nX_ages[i+1]
                ratio = time/self.coeffs_development_times[i]
                return c1*(1-ratio)+c2*ratio
            else:
                time=time-self.coeffs_development_times[i]
            i+=1
        return nX_ages[-1]
    
    def calculate_n5(self,time: float):
        return self._calculate_approx_float(self.n5_ages,time)

    def calculate_n8(self,time: float):
        return self._calculate_approx_float(self.n8_ages,time)

    def calculate_growth_coeffs(self,time: float):
        i=0
        steps_time = 0 
        while i<len(self.growth_coeffs_list)-1:
            if self.coeffs_development_times[i]>time:
                c1=self.growth_coeffs_list[i]
                c2=self.growth_coeffs_list[i+1]
                ratio = time/self.coeffs_development_times[i]
                return {k:c1[k]*(1-ratio)+c2[k]*ratio for k in c1.keys()}
            else:
                time=time-self.coeffs_development_times[i]
            i+=1
        return self.growth_coeffs_list[-1]
    
    def calculate_distance_to_center(self,time: float):
        i=0
        steps_time = 0 
        while i<len(self.distances_to_center)-1:
            if self.coeffs_development_times[i]>time:
                c1=self.distances_to_center[i]
                c2=self.distances_to_center[i+1]
                ratio = time/self.coeffs_development_times[i]
                return c1*(1-ratio)+c2*ratio
            else:
                time=time-self.coeffs_development_times[i]
            i+=1
        return self.distances_to_center[-1]

    def calculate_surface_frequency(self,time:float):
        return self._calculate_approx_float(self.surface_frequency,time)
        
    def calculate_growth_rate(self,time:float):
        if self.growth_rate:
            return self._calculate_approx_float(self.growth_rate,time)

class YoungLine:
    def __init__(self,age,initial_point,old_direction=None,color_index=-1,
                 maturing_time = 1,aprox_maturing_age=-2,is_left=False, 
                 global_simulation = None,repulsing_age=None,
                 starting_point=None):
        self.age=age
        if color_index == -1:
            self.color_index = get_color_index()
        else:
            self.color_index = color_index

        self.active=True
        self.adult=False
        self.maturing=False
        self.attracting=True
        self.maturing_time = maturing_time
        self.aprox_maturing_age = aprox_maturing_age
        self.old_direction=old_direction
        self.line: List[Tuple[float,float,float]]=[initial_point]
        if starting_point:
            self.line=[starting_point]+self.line
        self.connected_secondary : List[YoungLine] =[]
        self.connection : Optional[int] = None
        self.is_left=is_left
        self.maturing_adult_time=maturing_time
        self.repulsing_age=repulsing_age
        self.connection_point=None
        self.gs=context.Context().data_context.global_simulation
        self.primordium_label=None
        self.primary=None
        self.reach_ring_age=-10000.

        # if global_simulation is None:
        #     global_simulation = context.Context().data_context.global_simulation
        # self.global_simulation = global_simulation

    def start_maturing(self,maturing_adult_time=None):
        self.maturing=True
        self.active=False
        if maturing_adult_time is None:
            self.maturing_time=-1
            #self.adult=True
            if self.age < 0:
                self.maturing_adult_time = max(1,-self.age+self.gs.MATURED_AGE)
            else:
                #self.maturing_adult_time = min(3,max(4-self.age,2))
                self.maturing_adult_time = 2
        else:
            self.maturing_adult_time=maturing_adult_time
        # add maybe another value for adulting time or something
        #self.maturing_time = self.maturing_adult_time-self.gs.ADULTING_TIME
        self.maturing_time = self.maturing_adult_time
        # if self.age>self.aprox_maturing_age:
        #     self.maturing_time = (self.age-(self.aprox_maturing_age+self.maturing_time)+self.maturing_time)/2
    
    def mature(self,time_step):
        self.maturing_time -= time_step
        self.maturing_adult_time = max(0,self.maturing_adult_time-time_step)
        if self.maturing_time <=0:
            self.adult=True
            for s in self.connected_secondary:
                s.adult = True
                s.set_primordium_label(self.primordium_label)
        # if self.repulsing_age is not None:
        #     if self.maturing_time<self.repulsing_age:
        #         self.attracting=False

    def set_primordium_label(self,label):
        self.primordium_label=label
        for s in self.connected_secondary:
            s.set_primordium_label(label)

    def connect_secondary(self,other,connection_index: int):
        self.connected_secondary.append(other)
        other.color_index=other.color_index
        other.connection=connection_index
        other.primary = self
        other.set_primordium_label(self.primordium_label)
        if other.age>self.age and self.gs.context.data_context.settings.HIGHEST_AGE:
            self.age =other.age

    def get_label(self):
        if self.gs.context.data_context.settings.PRINT_MODE==context.PrintMode.FULL:
            if self.maturing:
                return f'{self.age:.2f} {MATURING_TIME-self.maturing_time:.2f} {self.primordium_label}'
            if self.primordium_label is not None:
                return f'{self.age:.2f} {self.primordium_label}'
            return f'{self.age:.2f}'
        if self.gs.context.data_context.settings.PRINT_MODE==context.PrintMode.INDEX_AGE:
            if self.primordium_label is not None:
                return f'{self.age:.2f} {self.primordium_label}'
            return f'{self.age:.2f}'
        if self.gs.context.data_context.settings.PRINT_MODE==context.PrintMode.INDEX_ONLY:
            if self.primordium_label is not None:
                return f'{self.primordium_label}'
            return f'{self.age:.2f}'
        

    def get_junction(self):
        result = []
        for line in self.connected_secondary:
            result.append(self.line[line.connection])

    def get_segments(self) -> List[Tuple[Tuple[float,float,float],Tuple[float,float,float]]]:
        x_coords=[]
        y_coords=[]
        xx=[x[0] for x in self.line]
        yy=[x[2] for x in self.line]
        x_coords.append(xx)
        y_coords.append(yy)
        for line in self.connected_secondary:
            points = line.line+[self.line[line.connection]]
            xx=[x[0] for x in points]
            yy=[x[2] for x in points]
            x_coords.append(xx)
            y_coords.append(yy)
        return (x_coords,y_coords)


    def init_direction(self,center):
        if len(self.line)==0:
            return
        if len(self.line)>1:
            a=self.line[-1]
            b=self.line[-2]
            direction = np.array(a)-np.array(b)
        if len(self.line)==1:
            global_simulation = context.Context().data_context.global_simulation
            prediction_lines = []
            for v in global_simulation.lines:
                prediction_lines.append((generate_line.graph_to_lines(v.skeleton,global_simulation.points),(v.age)))
            direction = generate_line.predict_step(self.line[-1],center,prediction_lines,
                        global_simulation.coeffs["pos_range"],
                        global_simulation.coeffs["neg_range"],
                        global_simulation.coeffs["inhibition_coef"],
                        global_simulation.coeffs["attraction_coef"],
                        global_simulation.coeffs["age_coef"],
                        global_simulation.coeffs["age_cut_off_coef"],
                        global_simulation.coeffs["straight_coef"],
                        global_simulation.coeffs["peak_coef"])
            a=center
            b=self.line[-1]
        self.old_direction=tuple(direction/np.linalg.norm(direction))

    

    def get_simplified_points_skeleton(self,max_angle=3.11,refine_steps=5):
        def _line_to_skeleton(line):
            skeleton = {}
            skeleton[line[0]]={line[1]}
            skeleton[line[-1]]={line[-2]}
            for a,b,c in zip(line,line[1:],line[2:]):
                skeleton[b]={a,c}
            return skeleton
        junctions=set()
        for secondary in self.connected_secondary:

            # for point in secondary.line:
            #     if point not in self.points:
            #         reversed_indices[point]=len(self.points)
            #         self.points.append(point)
            #     else:
            #         reversed_indices[point]=self.points.index(point)
            #refine connection
            def sorting(x):
                a = vector3.normalize(Vector3(secondary.old_direction))
                b = vector3.normalise(Vector3(x)-Vector3(secondary.line[-1]))
                result = 1-vector3.dot(a,b)
                if vector3.length(Vector3(x)-Vector3(secondary.line[-1]))>self.gs.connection_distance*3.5:
                    result +=10
                return result


            if secondary.connection != None:
                secondary.connection_point=self.line[secondary.connection]
            #refinement step(optional)
            REFINEMENT=False
            if REFINEMENT:
                best = sorted(self.line,key=sorting)[0]
                if best == self.line[-1] or best == self.line[-2]:
                    best = self.line[-3]
                secondary.connection_point=best
            print(secondary.connection_point)
            junctions.add(secondary.connection_point)
        junctions.add(self.line[-1])
        
        simplified=simplified_path(self.line,junctions,max_angle=max_angle,refine_steps=refine_steps)
        skeleton =_line_to_skeleton(simplified)
        for secondary in self.connected_secondary:
            sec_skeleton = secondary.get_simplified_points_skeleton(max_angle=max_angle,refine_steps=refine_steps)
            skeleton[secondary.connection_point].add(secondary.line[-1])
            for k,v in sec_skeleton.items():
                skeleton[k] = skeleton.get(k,set()) | v
        print(self.line[-1])
        if self.line[-1] not in skeleton:
            import pdb; pdb.set_trace()
        return skeleton

    
    def __str__(self) -> str:
        return self.get_label()
    def __repr__(self) -> str:
        return "Young age: "+self.get_label()





class AdultLine:
    def __init__(self,age,skeleton,label_pos,adulted = False,maturing_time=2,color_index=0,matured_age=None,primordium_label=None):
        self.maturing_time = maturing_time
        self.adulted = adulted
        if maturing_time<=0:
            self.adulted=True
        self.age=age
        self.label=None
        if adulted:
            self.maturing_time = -1
        self.label_pos = label_pos
        self.skeleton : Dict[int,Set[int]] = skeleton
        self.color_index = color_index
        self.context = context.Context()
        if matured_age == None:
            self.matured_age = self.context.data_context.global_simulation.MATURED_AGE
        else:
            self.matured_age=matured_age
        self.suffix_label = ""
        self.left_grown = False
        self.right_grown = False
        self.primordium_label = primordium_label

        self.line_n_5 : YoungLine=None
        self.line_n_8 : YoungLine=None

        self.position_n_5=None
        self.position_n_8=None

    def add_age(self,time,global_simulation):
        age_before=self.matured_age
        self.age+=time
        #add age young lines that are out of simulation for calculations
        if self.line_n_8 is not None and self.line_n_8 not in global_simulation.in_progres:
            self.line_n_8.age+=time

        #TODO possibly fix
        if not self.adulted:
            self.maturing_time-=time
            if self.maturing_time<=0:
                self.adulted = True
                self.matured_age = self.context.data_context.global_simulation.MATURED_AGE
        else:
            self.matured_age+=time

        return self.matured_age

    def __str__(self) -> str:
        return self.get_label()

    def __repr__(self) -> str:
        return "AdultLine age: "+self.get_label()

    def get_label(self) -> str:
        
        if self.context.data_context.settings.PRINT_MODE==context.PrintMode.FULL:
            if self.matured_age>=0:
                return f'{(self.age):.2f} {self.matured_age: .2f} {self.primordium_label}'
            else:
                return f'{(self.age):.2f} {self.primordium_label}'
        if self.context.data_context.settings.PRINT_MODE==context.PrintMode.INDEX_AGE:
            return f'{(self.age):.2f} {self.primordium_label}'
        if self.context.data_context.settings.PRINT_MODE==context.PrintMode.INDEX_ONLY:
            return f'{self.primordium_label}'
        if self.context.data_context.settings.PRINT_MODE==context.PrintMode.NOTHING:
            return ""

    def get_segments(self) -> List[Tuple[Tuple[float,float,float],Tuple[float,float,float]]]:
        paths = get_all_paths(self.skeleton)
        gs=self.context.data_context.global_simulation
        x_coords=[]
        y_coords=[]
        for path in paths:
            points = [gs.points[x] for x in path]
            xx=[x[0] for x in points]
            yy=[x[2] for x in points]
            x_coords.append(xx)
            y_coords.append(yy)
        return (x_coords,y_coords)
            


class GlobalSimulation:
    def __init__(self,_context):
        self.MATURED_AGE=MATURED_AGE
        self.is_young_left=False
        self.context : context.Context=_context
        self.points=[]
        self.indices={}
        self.simulation_time=0
        self.lines : List[AdultLine] = []
        self.in_progres: List[YoungLine]=[]
        self.labels={}
        self.center=None
        self.YOUNG_DISTANCE=27.73
        #self.OLD_DISTANCE=60
        #self.OLD_DISTANCE=29
        #self.YOUNG_DISTANCE=29
        self.cone_coeffs = {'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}
        self.growth_rate=36
        #self.connection_distance=4
        self.connection_distance=4
        self.adulting_age=-1
        self.meristem_grow_speed=1
        self.do_maximum_attraction=False
        self.primordium_label = -1
        self.primordium_connection_distance=0
        
        #self.young_attraction_distance=20
        self.young_attraction_distance=18
        self.young_attraction_strength=0.15

        self.grow_new_iv_ages : Tuple[int,int] = (2,4)
        self.initial_iv_ages : Tuple[int,int] = (0,0)

        self.max_adulted_age=25
        
        # self.growth_coeffs_young={'A': 0.14, 'B': 2.24, 'C': 54.3}
        #self.coeffs={'pos_range': 29.55286852843167, 'age_coef': 0.15000000000000005, 'age_cut_off_coef': 2.309319232725185, 'inhibition_coef': 4.2137849054572545, 'attraction_coef': 0.7403846635475495, 'neg_range': 37.903322449305996, 'straight_coef': 3.0297568280257003, 'inertia': 0.5, 'peak_coef': 0.798175481635862}
        # self.coeffs={'pos_range': 20.6139871413999,
        #             'age_coef': 0.0898168675836104,
        #             'inhibition_coef': 2.7873971411850995,
        #             'attraction_coef': 0.0,
        #             'neg_range': 69.22591211160096,
        #             'straight_coef': 3.0624121912173043,
        #             'inertia': 3.2745548038320416,
        #             'peak_coef': 0.99,
        #             'age_cut_off_coef': 2.8961433833351786}
        # self.coeffs={'pos_range': 20.6139871413999,
        #     'age_coef': 0.0898168675836104,
        #     'inhibition_coef': 1.9380879937972648,
        #     'attraction_coef': 0,
        #     'neg_range': 52.504265466109196,
        #     'straight_coef': 2.6968647751699355,
        #     'inertia': 0.968567709663606,
        #     'peak_coef': 0.7007414561054597,
        #     'age_cut_off_coef': 0.7488342990477376}
        self.coeffs={'pos_range': 20.6139871413999,
            'age_coef': 0.03963136053954481,
            'inhibition_coef': 2.7873971411850995,
            'attraction_coef': 0.0,
            'neg_range': 59.22591211160096,
            'straight_coef': 3.0624121912173043,
            'inertia': 0.5,
            #'inertia': 1.,
            'peak_coef': 0.99,
            'age_cut_off_coef': 0.8961433833351786}
        self.growth_coeff_development=GrowthCoeffDevelopment([dict(A=0.114, B=2.842, C=50.03),dict(A=0.2, B=6, C=62.6)],[50,62],[30],[13],[52],[12,10,8])
        #self.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
        #self.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
        self.simulation_time=0
        self.distance_treshold=(self.growth_coeff_development.calculate_distance_to_center(self.simulation_time),3)
        self.distance_treshold[0]+=self.primordium_connection_distance

    def point_position(self,index):
        return self.points[index]

    def coord_position(self,index):
        return self.context.canvas._micron_coord_to_stack(self.points[index])

    def init(self,skeleton : dict,labels : list,simulation_time=0,is_new_grown=None) -> None:
        import surface_points
        self.lines = []
        self.in_progres =[]
        self.labels={}
        self.indices={}
        self.simulation_time=simulation_time
        self.center = self.context.data_context.values['center']
        self.points=list(skeleton.keys())
        if self.center in self.points:
            self.points.remove(self.center)
        #self.indices = {i:i for i,v in enumerate(self.points)}
        reversed_indices = {v:i for i,v in enumerate(self.points)}
        def label_to_age(label):
            try:
                return float(label)
            except:
                print
                return float(label[:-1])
        self.primordium_label = int(round(min(map(label_to_age,labels))))-1
        for i,label in enumerate(labels):
            label_pos=self.context.data_context.get_point_from_label(label)
            if label_pos is None:
                continue
            try:
                age=float(label)*self.growth_coeff_development.calculate_surface_frequency(simulation_time)/12
            except Exception as e:
                print(e)
                age=float(label[:-1])*self.growth_coeff_development.calculate_surface_frequency(simulation_time)/12
            print(label)
            _skeleton={reversed_indices[k]:{reversed_indices[x] for x in v} for k,v in find_compact_component(skeleton,label_pos).items()}
            cc = find_compact_component(skeleton,label_pos)
            branching_point=None
            for k,v in cc.items():
                if len(k)>2:
                    branching_point=reversed_indices[k]
            adulted=age>=self.MATURED_AGE
            maturing_time=max(0,-age+self.MATURED_AGE)
            #self.lines.append(AdultLine(age,_skeleton,reversed_indices[label_pos],adulted=adulted,maturing_time=maturing_time,color_index=get_color_index()))
            self.lines.append(AdultLine(age,_skeleton,reversed_indices[label_pos],adulted=adulted,maturing_time=maturing_time,color_index=get_color_index(),matured_age=age,primordium_label=int(label_to_age(label))))
            if is_new_grown is not None:

                l=self.lines[-1]
                l.right_grown=is_new_grown[i][0]
                l.left_grown=is_new_grown[i][-1]
        # _in_progres=self.context.data_context.get_points_from_label('p')
        # for point in _in_progres:
        #     self.in_progres[reversed_indices[point]]={reversed_indices[k]:{reversed_indices[x] for x in v} for k,v in find_compact_component(skeleton,point).items()}
        self.lines.sort(key=lambda x: -x.age)
        print()
        self.growth_coeffs=self.growth_coeff_development.calculate_growth_coeffs(self.simulation_time)
        self.distance_treshold=(self.growth_coeff_development.calculate_distance_to_center(self.simulation_time),3)
        print(self.simulation_time)
        print(self.growth_coeffs)
        print(self.distance_treshold)


        
        self.center=self.context.canvas._stack_coord_to_micron(self.center)
        self.points = [self.context.canvas._stack_coord_to_micron(p) for p in self.points]
        self.surface_points=surface_points.SurfacePoints(self,skip_primordiums=1)

    # def get_graph(self) -> dict:
    #     skeleton={}
    #     for _,line in self.lines.items():
    #         skeleton.update({self.point_position(k):{self.points[self.indices[x]] for x in v} for k,v in line.items()})

    #     for _,line in self.in_progres.items():
    #         skeleton.update({self.point_position(k):{self.points[self.indices[x]] for x in v} for k,v in line.items()})
    #     return skeleton

    def update_age(self,time_step : float):
        self.surface_points.step(time_step)

        #Young lines pass
        to_remove : Set[YoungLine] = set()
        for line in self.in_progres:
            if line.maturing:
                line.mature(time_step)
            line.age+=time_step
            MAX_AGE = 15
            if line.active and line.age>MAX_AGE:
                    to_remove.add(line)
            
        self.simulation_time+=time_step
            
        for line in to_remove:
            self.in_progres.remove(line)
        

        #setting labels increasing primordium stage
        # for i,line in enumerate(sorted(filter(lambda x: x.adulted,self.lines),key=lambda x: x.matured_age)):
        #     line.suffix_label=str(i+1)
            #print("age:",line.matured_age,i+1)
        

        #Adult lines pass
        for line  in self.lines:
            result = line.add_age(time_step,self)
            if self.growth_coeff_development.calculate_n8(self.simulation_time)<result and not line.left_grown:
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!",result[0],self.growth_coeff_development.calculate_n8(self.simulation_time),result[1])
                self.create_new_line(line,self.YOUNG_DISTANCE,self.initial_iv_ages[1],not self.is_young_left)
                line.left_grown = True

            if self.growth_coeff_development.calculate_n5(self.simulation_time)<result and not line.right_grown:
                self.create_new_line(line,self.YOUNG_DISTANCE,self.initial_iv_ages[0],self.is_young_left)
                line.right_grown = True


        #TODO update distance treshold
        #distance_treshold_original = self.distance_treshold[0]/(self.growth_coeffs['C']/self.growth_coeffs_young['C'])
        #self.growth_coeffs={k:(min(30,self.simulation_time*self.meristem_grow_speed)*self.growth_coeffs_old[k]+max(0,30-self.simulation_time*self.meristem_grow_speed)*v)/30 for k,v in self.growth_coeffs_young.items()}
        self.growth_coeffs=self.growth_coeff_development.calculate_growth_coeffs(self.simulation_time)
        
        self.distance_treshold=(self.growth_coeff_development.calculate_distance_to_center(self.simulation_time),3)

        growth_rate = self.growth_coeff_development.calculate_growth_rate(self.simulation_time)
        if growth_rate:
            self.growth_rate = growth_rate*12

        self.trim_old_paths()


    def make_lines_adult(self):
        #adding adult nodes
        def _line_to_skeleton(skeleton,line):
            for b,point,n in zip(line.line[:-2],line.line[1:-1],line.line[2:]):
                skeleton[point]={b,n}
        to_remove: Set[YoungLine]=set()
        for line in self.in_progres:
            if line.adult==True and line.connection is None:
                # if line.primordium_label==-31:
                #     import pdb;pdb.set_trace()
                to_remove.add(line)
                r = [l for l in line.connected_secondary]
                while r:
                    secondary = r.pop()
                    r.extend(secondary.connected_secondary)
                    to_remove.add(secondary)
                reversed_indices = {}
                # for point in line.line:
                #     # indices[len(self.points)]=point
                #     # reversed_indices[point]=len(self.points)
                #     #apply modified option
                #     if point not in self.points:
                #         reversed_indices[point]=len(self.points)
                #         self.points.append(point)
                #     else:
                #         reversed_indices[point]=self.points.index(point)
                skeleton=line.get_simplified_points_skeleton()

                # skeleton[line.line[0]]={line.line[1]}
                # skeleton[line.line[-1]]={line.line[-2]}
                # _line_to_skeleton(skeleton,line)


                # connections = []

                # for secondary in line.connected_secondary:

                #     skeleton[secondary.line[0]]={secondary.line[1]}
                #     connection_point=secondary.connection_point
                #     skeleton[secondary.line[-1]]={secondary.line[-2],connection_point}
                #     skeleton[connection_point].add(secondary.line[-1])
                #     _line_to_skeleton(skeleton,secondary)
                    
                # if connections:
                #     branching_point = min(connections,key=lambda x: -line.line.index(x))
                # else:
                #     branching_point = None

                print('skeleton size',len(skeleton.keys()))
                # _skeleton=None
                # if len(skeleton.keys())>70:
                #     _skeleton=skeleton
                #     skeleton = _trim_skeleton(skeleton)
                #     print('skeleton trimmed size',len(skeleton.keys()))
                #     if line.line[-1] not in skeleton:
                #         if len([k for k,v in skeleton.items() if len(v)>2])!=len([k for k,v in _skeleton.items() if len(v)>2]):
                #             skeleton=_skeleton
                for point in skeleton.keys():
                    try:
                        reversed_indices[point]=self.points.index(point)
                    except:
                        reversed_indices[point]=len(self.points)
                        self.points.append(point)
                try:
                    label_pos=reversed_indices[line.line[-1]]
                except:
                    import pdb; pdb.set_trace()
                skeleton = {reversed_indices[k]:{reversed_indices[v] for v in vv} for k,vv in skeleton.items()}

                if line.age < 0:
                    maturing_time = min(3,-line.age+self.MATURED_AGE)
                else:
                    maturing_time = min(3,max(8-line.age,2))

                maturing_time = line.maturing_adult_time
                #import pdb; pdb.set_trace()
                self.lines.append(AdultLine(line.age,skeleton,label_pos,adulted=maturing_time<=0,color_index=line.color_index,maturing_time=maturing_time,primordium_label=line.primordium_label))
        for line in to_remove:
            self.in_progres.remove(line)

    def trim_old_paths(self,keep_age=30):
        #removes old paths
        t = time.time()
        self.lines=sorted(self.lines,key=lambda x: -x.matured_age)
        keep_amount=len(list(filter(lambda x: x.matured_age<=20,self.lines)))
        keep = self.lines[-keep_amount:]
        points_to_keep = set()
        line : AdultLine
        for line in keep:
            for k,v in line.skeleton.items():
                points_to_keep.add(k)
                points_to_keep.union(v)

        new_points = []
        indices = {}
        for point in points_to_keep:
            indices[point]=len(new_points)
            new_points.append(self.points[point])

        for line in keep:
            line.skeleton={indices[k]:{indices[vv] for vv in v} for k,v in line.skeleton.items()}
            line.label_pos=indices[line.label_pos]

        self.lines=keep

        self.points=new_points
        #indices = {i:self.points[i] for i in points_to_keep}


    def simulation(self,steps : int,time_step: float,coeffs : Optional[Dict[str,float]]=None):
        print("start simulation")
        t=time.time()
        if coeffs is None:
            # coeffs={'pos_range': 107.89361255813142, 'age_coef': 0.14, 'inhibition_coef': 6.355953160070807, 'attraction_coef': 2.293621646025508, 'neg_range': 32.98578691659749, 'straight_coef': 2.3624899123526997, 'inertia': 2.7, 'peak_coef': 0.4152491906221574,'age_cut_off_coef':4}
            if self.context.data_context.settings.UPDATE_COEFFS_WITH_TIME:
                self.coeffs=self.growth_coeff_development.calculate_coeffs(self.simulation_time)
            coeffs=self.coeffs

        print([len(p.line) for p in self.in_progres])
        if self.simulation_time>5 and self.context.data_context.settings.HIGHEST_AGE:
            self.in_progres = sorted(self.in_progres,key=lambda x:-x.age)
        if self.simulation_time>5 and self.context.data_context.settings.FIRST_REACHED:
            self.in_progres = sorted(self.in_progres,key=lambda x:abs(x.reach_ring_age))
        generate_line.global_simulation_step(steps,self,self.cone_coeffs,**coeffs,growth_rate=max(12,random.normalvariate(self.growth_rate,0)),
                                             distance_treshold=self.distance_treshold,connection_distance=self.connection_distance,
                                             time=time_step,young_attraction_distance=self.young_attraction_distance,
                                             young_attraction_strength=self.young_attraction_strength,do_maximum_attraction=self.do_maximum_attraction,max_adulted_age=self.max_adulted_age)
        print([len(p.line) for p in self.in_progres])

        #update primirdial age
        # for line in self.in_progres:
        #     line.age+=1

        # self.lines = {str(int(k)+1):v for k,v in self.lines.items()}
        # self.labels={k:str(int(v)+1) for k,v in self.labels.items()}


        print(f'points number: {len(self.points)}')
        print('simulation step time:',time.time()-t)
        print(f'simulation age: {self.simulation_time}')

    def positions_to_simulation_vectors(self,points,coeffs=None):
        if coeffs==None:
            if self.context.data_context.settings.UPDATE_COEFFS_WITH_TIME:
                self.coeffs=self.growth_coeff_development.calculate_coeffs(self.simulation_time)
            coeffs=self.coeffs
        prediction_lines=generate_line.gen_prediction_lines(self)

        result = [
            generate_line.simulation_vector_in_pos(prediction_lines,point,self.cone_coeffs,self.center,**coeffs)
            for point in points
        ]
        return result
        

        #self.make_lines_adult()
    def get_closest_end_to_point(self, skeleton : Dict[int,Set[int]],point: Tuple[float,float,float]):
        return min([k for k,v in skeleton.items() if len(v)<=1],key= lambda x: sum((a-b)**2 for a,b in zip(self.points[x],point)))

    def get_closest_to_center(self, skeleton : Dict[int,Set[int]]):
            return min([k for k,v in skeleton.items() if len(v)<=1],key= lambda x: sum((a-b)**2 for a,b in zip(self.points[x],self.center)))


    def get_meristem_size(self):
        y = self.points[self.get_closest_to_center(min(self.lines,key=lambda x: x.matured_age).skeleton)]
        d2 = sum((a-b)**2 for a,b in zip(y,self.center))**0.5
        try:
            x=min([x for x in self.in_progres if x.maturing==True],key=lambda x: x.maturing_time).line[-1]
            d1=sum((a-b)**2 for a,b in zip(x,self.center))**0.5
        except:
            d1=d2
        return (d1+d2)/2


    def distance_points(self,skeleton, a,b):
        position_skeleton = {self.points[k]:set(self.points[p] for p in v) for k,v in skeleton.items()}
        path = find_primary(position_skeleton,self.points[a],self.points[b])
        return sum(sum((w-u)**2 for w,u in zip(a,b))**0.5 for (a,b) in path)
        
    def neighbour_angle(self,skeleton, index):
        if len(skeleton[index])!=2:
            return 3.14
        else:
            a,b = skeleton[index]
            a=np.array(self.points[a])-np.array(self.points[index])
            b=np.array(self.points[b])-np.array(self.points[index])
            a=a/np.linalg.norm(a)
            b=b/np.linalg.norm(b)
            return np.arccos(np.dot(a,b))


    def young_line_start(self,_line:AdultLine,distance,is_left,start=None):
        skeleton=_line.skeleton
        def s(a,b):
            v1 = np.array(a) - np.array(b)
            v1=v1/np.linalg.norm(v1)
            v1=v1+np.array(b)
            v2 = np.array(b) - self.center
            v1=v1/np.linalg.norm(v1)
            v2=v2/np.linalg.norm(v2)
            return np.cross(v1,v2)[1]

        def get_point_distance(start,line,distance,is_left,closest_limit = 20):
            curr_distnace = 0
            point_before=None
            branched = True
            last_point=self.points[start]
            point=self.points[list(line[start])[0]]
            while curr_distnace<distance:
                new_curr_distnace=curr_distnace+np.linalg.norm(np.array(point)-np.array(last_point))
                if new_curr_distnace>distance:
                    t = (distance-curr_distnace)/np.linalg.norm(np.array(point)-np.array(last_point))
                    result = tuple(np.array(last_point)*(1-t)+t*np.array(point))
                    return result,last_point,branched
                else:
                    curr_distnace=new_curr_distnace

                point_before=last_point
                last_point=point

                if closest_limit>0:
                    a=np.array(last_point)
                    b=np.array(point_before)
                    c=b-a
                    c=c/np.linalg.norm(c)
                    if is_left:
                        c=np.array((c[2],0,-c[0]))
                    else:
                        c=np.array((-c[2],0,c[0]))
                    point = (a+c*8)
                    for l in self.lines:
                        if l.get_label()!=_line.get_label():
                            for i in l.skeleton.keys():
                                if i<len(self.points) and np.linalg.norm(np.array(self.points[i])-point)<closest_limit:
                                    return last_point,point_before,branched

                points=[self.points[p] for p in line[self.points.index(last_point)] if self.points[p]!=point_before]
                if (len(points)==1):
                    point=points[0]
                elif (len(points)>1):
                    #s2=lambda x: s(x,points[0])
                    s2=lambda x: s(x,last_point)
                    s_points = sorted(points,key=s2)
                    branched=True
                    if is_left:
                        point=s_points[-1]
                    else:
                        point=s_points[0]
                elif (len(points)==0):
                    return last_point,point_before,branched

            
            return last_point,point_before,branched

        def check_angle(_a,_b,_center):
            a=np.array((_a[0],0,_a[2]))
            b=np.array((_b[0],0,_b[2]))
            center=np.array((_center[0],0,_center[2]))
            v=a-b
            v=v/np.linalg.norm(v)
            vc=center-b
            vc=vc/np.linalg.norm(vc)
            d = sum((a*b) for a, b in zip(v, vc))
            return abs(d)<0.4

        DEFAULT_SIZE = 40
        MIN_SIZE = 30

        if start is None:
            start=self.get_closest_to_center(skeleton)
        line=skeleton
        branching_points = [self.distance_points(skeleton,start,x) for x,v in skeleton.items() if len(v)==3]
        knees = sorted([(k,self.neighbour_angle(skeleton,k),self.distance_points(skeleton,start,k)) for k in skeleton.keys() if self.neighbour_angle(skeleton,k)<2.2],key=lambda x: x[1])
        if branching_points:
            #branching_point_distance = max(MIN_SIZE,min(branching_points))
            branching_point_distance = min(branching_points)
        else:
            knees = sorted([(k,self.neighbour_angle(skeleton,k),self.distance_points(skeleton,start,k)) for k in skeleton.keys() if self.neighbour_angle(skeleton,k)<2.7],key=lambda x: x[1])
            if len(knees)==1:
                branching_point_distance = max(30,knees[0][2])
            elif len(knees)>0:
                closest = min(knees, key=lambda x: x[2])
                if MIN_SIZE<closest[2]<80:
                    branching_point_distance = closest[2]
                elif MIN_SIZE<knees[0][2]<80:
                    branching_point_distance = knees[0][2]
                elif closest[2]<80:
                    branching_point_distance = closest[2]
                else:
                    branching_point_distance=DEFAULT_SIZE
            else:
                branching_point_distance=DEFAULT_SIZE
        print(f"\nKnees amount = {len(knees)}\n")
        if knees:
            a,b,branched=get_point_distance(start,line,distance+branching_point_distance,is_left)
        else: 
            a,b,branched=get_point_distance(start,line,distance+branching_point_distance,is_left,10)

        def draw(*arself):
            self.context.graphic_context.mark_temporary_points_add([self.context.canvas._micron_coord_to_stack(a) for a in arself])
        def draw_path(*arself):
            self.context.graphic_context.mark_temporary_paths_add([[self.context.canvas._micron_coord_to_stack(a) for a in arself]])
        #self.context.graphic_context.mark_temporary_points([self.context.canvas._micron_coord_to_stack(a),self.context.canvas._micron_coord_to_stack(b)])
        # if branched:
        #     if check_angle(a,b,self.center):
        #         a=np.array(a)
        #         center=np.array((self.center[0],0,self.center[2]))
        #         vc=center-a
        #         return tuple(a+vc/np.linalg.norm(vc)*5)
        a=np.array(a)
        b=np.array(b)
        c=b-a
        c=c/np.linalg.norm(c)
        if is_left:
            c=np.array((c[2],0,-c[0]))
        else:
            c=np.array((-c[2],0,c[0]))
        point = tuple(a+c*5)
        p=[self.context.canvas._micron_coord_to_stack(x) for x in [tuple(a),tuple(b),point]]
        # print("points,branching",[self.context.canvas._micron_coord_to_stack(x) for x in [tuple(a),tuple(b),point]])
        # print(np.array(p[1])-np.array(p[0]))
        # print(np.array(p[2])-np.array(p[0]))
        # print("is left",is_left)
        # draw(tuple(a))
        # draw_path(tuple(a+0.1),tuple(b+0.1))
        return point,tuple(a)

    def create_new_line(self,line : AdultLine,distance,age,is_left):
        skeleton = line.skeleton
        #distance = np.random.normal(loc=distance,scale=2.)
        distance = distance
        points=self.young_line_start(line,distance,is_left)

        #project to cone
        point = generate_line.project_to_cone(points[0],self.center,**self.cone_coeffs)
        point2 = generate_line.project_to_cone(points[1],self.center,**self.cone_coeffs)
        result = YoungLine(age,point,color_index=get_color_index(),maturing_time=2+self.adulting_age,is_left=is_left,starting_point=point2)
        result.init_direction(self.center)
        if is_left:
            line.line_n_8=result
        else:
            line.line_n_5=result
        self.in_progres.append(result)

    def mark_index_points(self,points):
        c=self.context
        c.graphic_context.mark_temporary_points([c.canvas._micron_coord_to_stack(self.points[p]) for p in points])

    def calculate_new_lines_positions(self):
        for line in self.lines:
            line.position_n_5 = self.young_line_start(line,self.YOUNG_DISTANCE,False)
            line.position_n_8 = self.young_line_start(line,self.YOUNG_DISTANCE,True)

    def _recalculate_labels(self):
        for i,line in enumerate(sorted(self.lines,key=lambda x: x.age)):
            line.label=str(i-1)

    def get_graph(self) -> dict:

        skeleton={}
        for _skeleton in (line.skeleton for line in self.lines):
            skeleton.update({self.coord_position(k):{self.coord_position(x) for x in v} for k,v in _skeleton.items()})

        young_skeleton={}
        #growing lines
        for line in self.in_progres:
            if len(line.line)>1:
                young_skeleton[line.line[0]]={line.line[1]}
                young_skeleton[line.line[-1]]={line.line[-2]}
                for b,point,n in zip(line.line[:-2],line.line[1:-1],line.line[2:]):
                    young_skeleton[point]={b,n}

        for line in self.in_progres:
            for c in line.connected_secondary:
                young_skeleton[c.line[-1]].add(line.line[c.connection])
                young_skeleton[line.line[c.connection]].add(c.line[-1])

        skeleton.update({self.context.canvas._micron_coord_to_stack(k):{self.context.canvas._micron_coord_to_stack(x) for x in v} for k,v in young_skeleton.items()})
        return skeleton

    def grow(self,time):
        self.points = generate_line.grow_points(self.points,self.center,time)


    def get_data(self):
        def _pos_to_index(x):
            result = 0
            xx=self.context.canvas._micron_coord_to_stack(x)
            for v in xx:
                result=result*10000+round(v,1)
            return result


        self._recalculate_labels()
        skeleton=self.get_graph()

        values = {}
        values['point']=list(skeleton.keys())
        values['line']=graph_to_lines(skeleton)
        values['label']=[self.coord_position(line.label_pos) for line in self.lines]
        values['center']=tuple(round(x*100)/100 for x in self.context.canvas._micron_coord_to_stack(self.center))
        values['point'].append(tuple(round(x*100)/100 for x in self.context.canvas._micron_coord_to_stack(self.center)))
        values['label'].append(tuple(round(x*100)/100 for x in self.context.canvas._micron_coord_to_stack(self.center)))


        labels = {}
        labels['point']={k:'' for k in skeleton.keys()}
        labels['line']={k:'' for k in values['line']}
        labels['label']={self.coord_position(line.label_pos):line.get_label() for line in self.lines}
        color_map={_pos_to_index(self.points[self.get_closest_to_center(line.skeleton)]):line.color_index for line in self.lines}

        for k,v in labels['label'].items():
            labels['point'][k]=v
        for line in self.in_progres:
            if line.get_label() not in color_map:
                color_map[_pos_to_index(line.line[-1])]=line.color_index
            if line.connection is None:
                color_map[_pos_to_index(line.line[-1])]=line.color_index
            labels['point'][self.context.canvas._micron_coord_to_stack(line.line[-1])]=line.get_label()
            labels['label'][self.context.canvas._micron_coord_to_stack(line.line[-1])]=line.get_label()

            values['label'].append(self.context.canvas._micron_coord_to_stack(line.line[-1]))
            
        #labels['label'][values['center']]='center'
        labels['point'][values['center']]='center'

        if self.context.data_context.settings.SHOW_SURFACE_POINTS:
            for point in self.surface_points.existing_points:
                values["point"].append(self.context.canvas._micron_coord_to_stack(point.position))
                values["label"].append(self.context.canvas._micron_coord_to_stack(point.position))
                labels['label'][self.context.canvas._micron_coord_to_stack(point.position)]=str(point)

        # JET distance_cololors_calculation
        distances = {}
        for line in self.lines:
            _skeleton = line.skeleton
            start=self.get_closest_to_center(_skeleton)
            branching_points = [self.distance_points(_skeleton,start,x) for x,v in _skeleton.items() if len(v)==3]

            if branching_points:
                distances[_pos_to_index(self.points[start])]=min(branching_points)
            else:
                distances[_pos_to_index(self.points[start])] = 0

        #import pdb; pdb.set_trace()

        for line in self.in_progres:
            if not line.connected_secondary:
                if _pos_to_index(line.line[-1]) not in distances:
                    distances[_pos_to_index(line.line[-1])]=0
            else:
                connection=max(secondary.connection for secondary in line.connected_secondary)
                distance = sum(sum((w-u)**2 for w,u in zip(a,b))**0.5 for (a,b) in zip(line.line[connection:-1],line.line[connection+1:-1]))
                start = line.line[-1]
                distances[_pos_to_index(start)]=distance
                for secondary in line.connected_secondary:
                    print("reassigning", secondary.get_label(),secondary.color_index,distance)
                    distances[_pos_to_index(secondary.line[-1])]=distance
                #print(str(line),line.color_index,distances[line.color_index])
        highest_distance = max([v for k,v in distances.items()])+0.000001
        if highest_distance>0:
            distances = {k:0.999991-min(0.99999,(v/150)**0.8) for k,v in distances.items()}
        self.context.simulation_controller.color_range = distances


        return (skeleton,values,labels,color_map)

    def calculate_consecutive_angles(self):
        points=[]
        labels=[]
        for line in sorted(self.lines,key=lambda x: x.matured_age):
            points.append(self.points[self.get_closest_to_center(line.skeleton)])
            labels.append(line.get_label())
        angles = np.zeros((len(points),len(points)))
        for ((i,a),(j,b)) in product(enumerate(points),repeat=2):
            if i>=j:
                continue
            angles[i,j] = calculate_angles(self.center,a,b)
            angles[j,i] = calculate_angles(self.center,a,b)
        return labels,angles

    def predict_point(self,pos):
        prediction_lines = []
        for v in self.lines:
            prediction_lines.append((generate_line.graph_to_lines(v.skeleton,self.points),(v.matured_age)))
        coeffs = dict(**self.coeffs)
        del coeffs["inertia"]
        return generate_line.predict_step_global_simulation(pos,self.center,prediction_lines,**coeffs)



    def project_to_cone(self,point):
        return generate_line.project_to_cone(point,self.center,self.cone_coeffs['a1'],self.cone_coeffs['a2'],self.cone_coeffs['c'],self.cone_coeffs['tan'])



def _test():
    c.data_context.global_simulation.init(c.canvas.skeleton_graph,[str(i) for i in range(-4,12)])
    c.data_context.global_simulation.move()
    s,v,l=c.data_context.global_simulation.get_data()
    c.data_context.values=v
    c.data_context.labels=l
    c.canvas.load_skeleton_graph(s)












