from skeleton import *
import generate_line
from typing import List, Literal,Optional,Dict,Tuple,Set
import time
import random
from pyrr import vector3,Vector3
from math import floor


CENTER = (527.0, 563.0, 100.0)
color_index=0
def get_color_index():
    global color_index
    color_index+=1
    return color_index

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
    def __init__(self,coeffs_list:List[dict],distances_to_center: List[float],coeffs_development_times : List[float]) -> None:
        self.coeffs_list=coeffs_list
        self.coeffs_development_times=coeffs_development_times
        self.distances_to_center=distances_to_center

    def calculate_growth_coeffs(self,time: float):
        i=0
        steps_time = 0 
        while i<len(self.coeffs_development_times):
            if self.coeffs_development_times[i]>time:
                c1=self.coeffs_list[i]
                c2=self.coeffs_list[i+1]
                ratio = time/self.coeffs_development_times[i]
                return {k:c1[k]*(1-ratio)+c2[k]*ratio for k in c1.keys()}
            else:
                time=time-self.coeffs_development_times[i]
        return self.coeffs_list[-1]
    
    def calculate_distance_to_center(self,time: float):
        i=0
        steps_time = 0 
        while i<len(self.coeffs_development_times):
            if self.coeffs_development_times[i]>time:
                c1=self.distances_to_center[i]
                c2=self.distances_to_center[i+1]
                ratio = time/self.coeffs_development_times[i]
                return c1*(1-ratio)+c2*ratio
            else:
                time=time-self.coeffs_development_times[i]
        return self.distances_to_center[-1]

class YoungLine:
    def __init__(self,age,initial_point,old_direction=None,color_index=-1,maturing_time = 2,aprox_maturing_age=-2):
        self.age=age
        if color_index == -1:
            self.color_index = get_color_index()
        else:
            self.color_index = color_index

        self.active=True
        self.adult=False
        self.maturing=False
        self.maturing_time = maturing_time
        self.aprox_maturing_age = aprox_maturing_age
        self.old_direction=old_direction
        self.line: List[Tuple[float,float,float]]=[initial_point]
        self.connected_secondary : List[YoungLine] =[]
        self.connection : Optional[int] = None

    def start_maturing(self):
        self.maturing=True
        if self.age>self.aprox_maturing_age:
            self.maturing_time = (self.age-(self.aprox_maturing_age+self.maturing_time)+self.maturing_time)/2

    def get_label(self):
        return f'{self.age:.2f}'


    def init_direction(self,center):
        if len(self.line)==0:
            return
        if len(self.line)>1:
            a=self.line[-1]
            b=self.line[-2]
        if len(self.line)==1:
            a=center
            b=self.line[-1]
        direction = np.array(a)-np.array(b)
        self.old_direction=tuple(direction/np.linalg.norm(direction))



class AdultLine:
    def __init__(self,age,skeleton,label_pos,adulted = False,maturing_time=2,color_index=0,matured_age=-1):
        self.maturing_time = maturing_time
        self.age=age
        self.label=None
        if adulted:
            self.maturing_time = -1
        self.adulted = adulted
        self.label_pos = label_pos
        self.skeleton : Dict[int,Set[int]] = skeleton
        self.color_index = color_index
        self.matured_age = matured_age

    def add_age(self,time):
        self.age+=time
        #TODO possibly fix
        if not self.adulted:
            self.maturing_time-=time
            if self.maturing_time<=0:
                self.adulted = True
                self.matured_age = 0
        else:
            self.matured_age+=time

        if self.matured_age-time<floor(self.matured_age):
            return int(floor(self.matured_age))
        else:
            return None

    def get_label(self):
        if self.matured_age>=0:
            return f'{(self.age):.2f} {self.matured_age: .2f}'
        else:
            return f'{(self.age):.2f}'



class GlobalSimulation:
    def __init__(self,context):
        self.is_young_left=False
        self.context=context
        self.points=[]
        self.indices={}
        self.simulation_age=0
        self.lines : List[AdultLine] = []
        self.in_progres: List[YoungLine]=[]
        self.labels={}
        self.center=None
        self.OLD_DISTANCE=27.73
        self.YOUNG_DISTANCE=27.73
        #self.OLD_DISTANCE=60
        #self.OLD_DISTANCE=29
        #self.YOUNG_DISTANCE=29
        self.cone_coeffs = {'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}
        self.growth_rate=36
        self.connection_distance=5
        self.adulting_age=-1
        self.meristem_grow_speed=1
        self.young_attraction_distance=6

        self.grow_new_iv_ages : Tuple[int,int] = (2,4)
        self.initial_iv_ages : Tuple[int,int] = (0,0)
        
        # self.growth_coeffs_young={'A': 0.14, 'B': 2.24, 'C': 54.3}
        self.coeffs={'pos_range': 29.55286852843167, 'age_coef': 0.15000000000000005, 'age_cut_off_coef': 2.309319232725185, 'inhibition_coef': 4.2137849054572545, 'attraction_coef': 0.7403846635475495, 'neg_range': 37.903322449305996, 'straight_coef': 3.0297568280257003, 'inertia': 0.5, 'peak_coef': 0.798175481635862}
        
        self.growth_coeff_development=GrowthCoeffDevelopment([dict(A=0.114, B=2.842, C=50.03),dict(A=0.2, B=6, C=62.6)],[50,62],[30])
        #self.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
        #self.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
        self.simulation_time=0
        self.distance_treshold=(self.growth_coeff_development.calculate_distance_to_center(self.simulation_time),3)

    def point_position(self,index):
        return self.points[index]

    def coord_position(self,index):
        return self.context.canvas._micron_coord_to_stack(self.points[index])

    def init(self,skeleton : dict,labels : list) -> None:
        self.lines = []
        self.in_progres =[]
        self.labels={}
        self.indices={}
        self.simulation_age=0
        self.center = self.context.data_context.values['center']
        self.points=list(skeleton.keys())
        if self.center in self.points:
            self.points.remove(self.center)
        #self.indices = {i:i for i,v in enumerate(self.points)}
        reversed_indices = {v:i for i,v in enumerate(self.points)}
        for label in labels:
            label_pos=self.context.data_context.get_point_from_label(label)
            if label_pos is None:
                continue
            age=float(label)
            print(label)
            _skeleton={reversed_indices[k]:{reversed_indices[x] for x in v} for k,v in find_compact_component(skeleton,label_pos).items()}
            cc = find_compact_component(skeleton,label_pos)
            branching_point=None
            for k,v in cc.items():
                if len(k)>2:
                    branching_point=reversed_indices[k]
            adulted=age>=0
            maturing_time=max(0,-age)
            #self.lines.append(AdultLine(age,_skeleton,reversed_indices[label_pos],adulted=adulted,maturing_time=maturing_time,color_index=get_color_index()))
            self.lines.append(AdultLine(age,_skeleton,reversed_indices[label_pos],adulted=adulted,maturing_time=maturing_time,color_index=get_color_index(),matured_age=age))
        # _in_progres=self.context.data_context.get_points_from_label('p')
        # for point in _in_progres:
        #     self.in_progres[reversed_indices[point]]={reversed_indices[k]:{reversed_indices[x] for x in v} for k,v in find_compact_component(skeleton,point).items()}

        
        self.growth_coeffs=self.growth_coeff_development.calculate_growth_coeffs(0)


        
        self.center=self.context.canvas._stack_coord_to_micron(self.center)
        self.points = [self.context.canvas._stack_coord_to_micron(p) for p in self.points]

    # def get_graph(self) -> dict:
    #     skeleton={}
    #     for _,line in self.lines.items():
    #         skeleton.update({self.point_position(k):{self.points[self.indices[x]] for x in v} for k,v in line.items()})

    #     for _,line in self.in_progres.items():
    #         skeleton.update({self.point_position(k):{self.points[self.indices[x]] for x in v} for k,v in line.items()})
    #     return skeleton

    def update_age(self,time_step : float):
        to_remove : Set[YoungLine] = set()
        for line in self.in_progres:
            if line.maturing:
                line.maturing_time-=time_step
            if line.maturing and line.maturing_time<0:
                line.adult=True
            line.age+=time_step
            if line.active and line.age>20:
                    to_remove.add(line)
            
        self.simulation_time+=time_step
            
        for line in to_remove:
            self.in_progres.remove(line)
            


        for line  in self.lines:
            result = line.add_age(time_step)
            if result == self.grow_new_iv_ages[1]:
                self.create_new_line(line,self.OLD_DISTANCE,self.initial_iv_ages[1],not self.is_young_left)

            if result == self.grow_new_iv_ages[0]:
                self.create_new_line(line,self.YOUNG_DISTANCE,self.initial_iv_ages[0],self.is_young_left)

        self.simulation_age+=time_step

        #TODO update distance treshold
        #distance_treshold_original = self.distance_treshold[0]/(self.growth_coeffs['C']/self.growth_coeffs_young['C'])
        #self.growth_coeffs={k:(min(30,self.simulation_age*self.meristem_grow_speed)*self.growth_coeffs_old[k]+max(0,30-self.simulation_age*self.meristem_grow_speed)*v)/30 for k,v in self.growth_coeffs_young.items()}
        self.growth_coeffs=self.growth_coeff_development.calculate_growth_coeffs(0)
        
        self.distance_treshold=(self.growth_coeff_development.calculate_distance_to_center(self.simulation_time),3)
        self.trim_old_paths()


    def make_lines_adult(self):
        #adding adult nodes
        def _line_to_skeleton(skeleton,line):
            for b,point,n in zip(line.line[:-2],line.line[1:-1],line.line[2:]):
                skeleton[point]={b,n}
        to_remove: Set[YoungLine]=set()
        for line in self.in_progres:
            if line.adult==True:
                to_remove.add(line)
                reversed_indices = {}
                skeleton={}
                # for point in line.line:
                #     # indices[len(self.points)]=point
                #     # reversed_indices[point]=len(self.points)
                #     #apply modified option
                #     if point not in self.points:
                #         reversed_indices[point]=len(self.points)
                #         self.points.append(point)
                #     else:
                #         reversed_indices[point]=self.points.index(point)

                skeleton[line.line[0]]={line.line[1]}
                skeleton[line.line[-1]]={line.line[-2]}
                _line_to_skeleton(skeleton,line)


                connections = []
                for secondary in line.connected_secondary:
                    to_remove.add(secondary)

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
                        if vector3.length(Vector3(x)-Vector3(secondary.line[-1]))>self.connection_distance*3.5:
                            result +=10
                        return result

                    best = sorted(line.line,key=sorting)[0]
                    if best == line.line[-1] or best == line.line[-2]:
                        best = line.line[-3]
                    secondary.connection=line.line.index(best)
                    connections.append(best)

                    skeleton[secondary.line[0]]={secondary.line[1]}
                    connection_point=line.line[secondary.connection]
                    skeleton[secondary.line[-1]]={secondary.line[-2],connection_point}
                    skeleton[connection_point].add(secondary.line[-1])
                    _line_to_skeleton(skeleton,secondary)
                    
                if connections:
                    branching_point = min(connections,key=lambda x: -line.line.index(x))
                else:
                    branching_point = None

                print('skeleton size',len(skeleton.keys()))
                _skeleton=None
                if len(skeleton.keys())>70:
                    _skeleton=skeleton
                    skeleton = _trim_skeleton(skeleton)
                    print('skeleton trimmed size',len(skeleton.keys()))
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
                if branching_point:
                    branching_point=reversed_indices[branching_point]

                self.lines.append(AdultLine(line.age,skeleton,label_pos,adulted=True,color_index=line.color_index))
        for line in to_remove:
            self.in_progres.remove(line)

    def trim_old_paths(self,keep_age=30):
        t = time.time()
        self.lines=sorted(self.lines,key=lambda x: -x.age)
        keep_amount=len(list(filter(lambda x: x.age<=20,self.lines)))
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
            coeffs=self.coeffs

        print([len(p.line) for p in self.in_progres])
        generate_line.global_simulation_step(steps,self,self.cone_coeffs,**coeffs,growth_rate=max(12,random.normalvariate(self.growth_rate,12)),distance_treshold=self.distance_treshold,connection_distance=self.connection_distance,time=time_step,young_attraction_distance=self.young_attraction_distance)
        print([len(p.line) for p in self.in_progres])

        #update primirdial age
        # for line in self.in_progres:
        #     line.age+=1

        # self.lines = {str(int(k)+1):v for k,v in self.lines.items()}
        # self.labels={k:str(int(v)+1) for k,v in self.labels.items()}


        print(f'points number: {len(self.points)}')
        print('simulation step time:',time.time()-t)
        print(f'simulation age: {self.simulation_age}')

        #self.make_lines_adult()
    def get_closest_to_center(self, skeleton : Dict[int,Set[int]]):
            return min([k for k,v in skeleton.items() if len(v)<=1],key= lambda x: sum((a-b)**2 for a,b in zip(self.points[x],self.center)))


    def distance_points(self,skeleton, a,b):
        position_skeleton = {self.points[k]:set(self.points[p] for p in v) for k,v in skeleton.items()}
        path = find_primary(position_skeleton,self.points[a],self.points[b])
        return sum(sum((w-u)**2 for w,u in zip(a,b))**0.5 for (a,b) in path)
        

    def young_line_start(self,skeleton,distance,is_left,start=None):

        def s(a,b):
            v1 = np.array(a) - self.center
            v2 = np.array(b) - self.center
            return np.cross(v1,v2)[1]

        def get_point_distance(start,line,distance,is_left):
            curr_distnace = 0
            point_before=None
            last_point=self.points[start]
            point=self.points[list(line[start])[0]]
            while curr_distnace<distance:
                new_curr_distnace=curr_distnace+np.linalg.norm(np.array(point)-np.array(last_point))
                if new_curr_distnace>distance:
                    t = (distance-curr_distnace)/np.linalg.norm(np.array(point)-np.array(last_point))
                    result = tuple(np.array(last_point)*(1-t)+t*np.array(point))
                    return result,point
                else:
                    curr_distnace=new_curr_distnace


                point_before=last_point
                last_point=point
                points=[self.points[p] for p in line[self.points.index(last_point)] if self.points[p]!=point_before]
                if (len(points)==1):
                    point=points[0]
                elif (len(points)>1):
                    s2=lambda x: s(x,points[0])
                    s_points = sorted(points,key=s2)
                    if is_left:
                        point=s_points[-1]
                    else:
                        point=s_points[0]
                elif (len(points)==0):
                    return last_point,point_before

            return last_point,point_before

        if start is None:
            start=self.get_closest_to_center(skeleton)
        line=skeleton
        branching_points = [self.distance_points(skeleton,start,x) for x,v in skeleton.items() if len(v)==3]
        if branching_points:
            branching_point_distance = min(branching_points)
        else:
            branching_point_distance=30

        a,b=get_point_distance(start,line,distance+branching_point_distance,is_left)
        a=np.array(a)
        b=np.array(b)
        c=b-a
        c=c/np.linalg.norm(c)
        if is_left:
            c=np.array((-c[2],0,c[0]))
        else:
            c=np.array((c[2],0,-c[0]))
        point = tuple(a+c*4)

        return point

    def create_new_line(self,line : AdultLine,distance,age,is_left):
        skeleton = line.skeleton
        distance = np.random.normal(loc=distance,scale=2.)
        point=self.young_line_start(skeleton,distance,is_left)

        #project to cone
        point = generate_line.project_to_cone(point,self.center,**self.cone_coeffs)
        result = YoungLine(age,point,color_index=get_color_index(),maturing_time=2+self.adulting_age)
        result.init_direction(self.center)
        self.in_progres.append(result)

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
        self._recalculate_labels()
        skeleton=self.get_graph()

        values = {}
        values['point']=list(skeleton.keys())
        values['line']=graph_to_lines(skeleton)
        values['label']=[self.coord_position(line.label_pos) for line in self.lines]
        values['center']=self.context.canvas._micron_coord_to_stack(self.center)
        values['point'].append(self.context.canvas._micron_coord_to_stack(self.center))
        values['label'].append(self.context.canvas._micron_coord_to_stack(self.center))


        labels = {}
        labels['point']={k:'' for k in skeleton.keys()}
        labels['line']={k:'' for k in values['line']}
        labels['label']={self.coord_position(line.label_pos):line.get_label() for line in self.lines}
        color_map={line.get_label():line.color_index for line in self.lines}

        for k,v in labels['label'].items():
            labels['point'][k]=v
        for line in self.in_progres:
            color_map[line.get_label()]=line.color_index
            labels['point'][self.context.canvas._micron_coord_to_stack(line.line[-1])]=line.get_label()
            labels['label'][self.context.canvas._micron_coord_to_stack(line.line[-1])]=line.get_label()

            values['label'].append(self.context.canvas._micron_coord_to_stack(line.line[-1]))
            
        labels['label'][(self.context.canvas._micron_coord_to_stack(self.center))]='center'

        return (skeleton,values,labels,color_map)


def _test():
    c.data_context.global_simulation.init(c.canvas.skeleton_graph,[str(i) for i in range(-4,12)])
    c.data_context.global_simulation.move()
    s,v,l=c.data_context.global_simulation.get_data()
    c.data_context.values=v
    c.data_context.labels=l
    c.canvas.load_skeleton_graph(s)









