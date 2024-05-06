
import re
from math import ceil
import pickle
import copy
import time
from itertools import product
from random import sample
import glob


from PIL import Image
import pickle
import os
import numpy as np

#from skeleton import *
#from stack_io import *

#from global_simulation import GlobalSimulation
#import generate_line

def _gen_line(arg):
    return generate_line.generate_line(*arg)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Context(Borg):
    def __init__(self):
        Borg.__init__(self)


class DataContext():
    def __init__(self):
        self.values={'label':[],'point':[],'line':[],'center':None}
        self.labels={'label':{},'point':{},'line':{}}
        self.hidden=set()
        self.context=Context()
        self.dr_stack : np.array=None
        self.pi_stack : np.array=None
        self.heightmap=None
        self.dr_stack_shape=None
        #self.global_simulation : GlobalSimulation= GlobalSimulation(self.context)
        #self.test_coefs={'pos_range': 107.28579816664298, 'age_coef': 0.2798657353773545, 'inhibition_coef': 0.02620979385574644, 'attraction_coef': 2.7563011926944743, 'neg_range': 11.711794061841385, 'straight_coef': 1.6000000000000019, 'inertia': 1.3848891360943227, 'peak_coef': 0.48510826869913243}

    def add_junction(self,line,junction):
        return self.context.canvas.add_junction(line,junction)



    def pos_to_name(self,pos):
        pass

    def name_to_pos(self,name):
        pass

    def hide_data(self,datas : set):
        self.hidden=self.hidden | datas
        self.context.skeleton_distances.reload_grid()

    def show_data(self,datas : set):
        self.hidden=self.hidden - datas
        self.context.skeleton_distances.reload_grid()

    
    def hide_all(self):
        self.hidden=self.hidden | set(self.values['point']) | set(self.values['line'])
        self.context.skeleton_distances.reload_grid()

    def refresh_hidden(self):
        self.hidden=set(filter(lambda x: x in self.values['point'] or x in self.values['line'] or x == self.value['center']),self.hidden)



    def reload_all(self):
        pass


    def reload_from_skeleton(self):
        assert False,"not finished"
        for point in self.context.canvas.skeleton_graph.keys():
            if point not in self.values['point']:
                self.values['point'].append(point)
                self.labels['point'][point]=""
                self.hidden.add(point)
        self.reload_all()


    def reload_points(self,values,hidden):
        self.values=values
        self.hidden=hidden
        loaded_points=[point for point in self.values['point']]
        #TODO modify this function to load all skeleton keys
        self.hide_data(set(self.context.canvas.skeleton_graph.keys())-set(loaded_points))
        self.reload_all()

    def sort(self):
        self.values['point']=sorted(self.values['point'],key=self._key('point'))
        self.values['line']=sorted(self.values['line'],key=self._key('line'))
        self.reload_all()

    def get_labels(self):
        return [(label,point) for (point, label) in self.labels['label'].items()]


    def get_point_from_label(self,label):
        try:
            return [(point) for (point, _label) in self.labels['point'].items() if label.strip() == _label.strip()][0]  
        except:
            try:
                return [(point) for (point, _label) in self.labels['label'].items() if label.strip() == _label.strip()][0]
            except:
                pass
                #print('ERROR', label)  

    def get_points_from_label(self,label):
        try:
            return [(point) for (point, _label) in self.labels['point'].items() if label.strip() == _label.strip()]
        except:
            pass
            #print('ERROR', label)    

    def get_distinct_points(self,label):
        end=self.get_point_from_label(label)
        graph=find_compact_component(self.context.canvas.skeleton_graph,end)
        endpoints = [x for x,y in graph.items() if (len(y)==1 or len(y)>2) and x!=end]
        return (end,endpoints)

    def get_closest_label(self,point,max_tries=1000):
        points = [point]
        visited = set(points)
        i=1
        while points:
            if i>1000:
                break
            i+=1
            point = points.pop(0)
            label = self.labels['point'][point].strip()
            if label:
                return label
            else:
                for _point in self.context.canvas.skeleton_graph[point]:
                    if _point not in visited:
                        visited.add(_point)
                        points.append(_point)
        return ' '

    def _flatten_data(self):
        self.context.canvas._flatten_skeleton()
        new_hidden=set()
        for obj in self.hidden:
            if len(obj)==2:
                ((a,b,c),(d,e,f))=obj
                new_hidden.add(((a,b,100),(d,e,100)))
            else:
                (a,b,c)=obj
                new_hidden.add((a,b,100))
        self.hidden=new_hidden
        new_labels={}
        new_labels['point']={(a,b,100):v for (a,b,c),v in self.labels['point'].items()}
        new_labels['line']={((a,b,100),(d,e,100)):v for ((a,b,c),(d,e,f)),v in self.labels['line'].items()}
        new_labels['label']={(a,b,100):v for (a,b,c),v in self.labels['label'].items()}
        self.labels=new_labels
        self.values['point']=[(a,b,100) for (a,b,c) in self.values['point']]
        self.values['label']=[(a,b,100) for (a,b,c) in self.values['label']]
        self.values['line']=[((a,b,100),(d,e,100)) for ((a,b,c),(d,e,f)) in self.values['line']]
        self.reload_all()

        

    def _sample_distance_by_label(self,first_label,second_label,function=None,func_kwargs={}):
        _first = self.get_distinct_points(first_label)
        _second = self.get_distinct_points(second_label)
        ends=self.get_points_from_label('x')
        
        first_end=None
        labeled=self.get_points_from_label('x'+second_label)
        for point in _first[1]:
            if point in labeled:
                first_end=point
                break
        if first_end==None:   
            for point in _first[1]:
                if point in ends:
                    first_end=point
                    break
        if first_end==None:
            first_end=sorted(_first[1],key=lambda x: self.context.canvas.point_line_distance(x,[_second[0],_second[1][0]]))[0]
        first = (first_end,_first[0])
        
        second_end=None
        labeled=self.get_points_from_label('x'+first_label)
        for point in _second[1]:
            if point in labeled:
                second_end=point
                break
        if second_end==None:
            for point in _second[1]:
                if point in ends:
                    second_end=point
                    break
        if second_end==None:   
            second_end=sorted(_second[1],key=lambda x: self.context.canvas.point_line_distance(x,[first[0],first[1]]))[0]
        second = (second_end,_second[0])
        self.context.graphic_context.mark_temporary_points([*first,*second])
        if function is not None:
            return first,second,function(first,second,**func_kwargs)
        else:
            return first,second,self.sample_distance(first,second,**func_kwargs)

    def _skeleton_values(self,labels):
        result={}
        self.context.graphic_context.mark_temporary_points([])
        points_to_mark=[]
        for first,*b in labels:
            result[first]={}
            for second in b:
                points_first,points_second,result[first][second]=self._sample_distance_by_label(first,second)
                points_to_mark=points_to_mark+[*points_first,*points_second]
        self.context.graphic_context.mark_temporary_points(points_to_mark)
        return result

    def sample_distance(self,first, second,distamce=1):
        points=self.context.canvas.get_equal_points(*first,1)
        straight=[]
        for i in range(len(points)+1):
            t=i/len(points)
            straight.append(self.context.canvas.point_line_distance(tuple(a*(1-t)+b*t for a,b in zip(*first)),second))
        return ([self.context.canvas.point_line_distance(point,second) for point in points],straight)

    def sample_random_distance(self,first, second,distance=1,percent=90,sigma=30,sample_size=1000):
        _t=time.time()
        points=self.context.canvas.get_equal_points(*first,distance)
        
        random=[]
        scale=self.context.canvas.unit_to_microns_coef
        _first=self.context.canvas._stack_coord_to_opengl(first[0])
        _first=(_first[0]*scale,_first[1]*scale,_first[2]*scale)
        _second=self.context.canvas._stack_coord_to_opengl(first[1])
        _second=(_second[0]*scale,_second[1]*scale,_second[2]*scale)
        line=self.context.canvas._lines_coord_to_microns(find_primary(self.context.canvas.skeleton_graph, *second))
        temporary_paths=[]
        for i in range(sample_size):
            l=generate_line.generate_line(_first,_second,sigma,1)
            random.append(generate_line.calculate_distance(l,line))
            if i%100==0:
                print(f'{i/10}%')
                temporary_paths.append(l)
        print(time.time()-_t)
        _points=[self.context.canvas._stack_coord_to_micron(x) for x in points]
        self.context.graphic_context.mark_temporary_paths(temporary_paths)
        straight=[]
        for i in range(len(points)+1):
            t=i/len(points)
            straight.append(self.context.canvas.point_line_distance(tuple(a*(1-t)+b*t for a,b in zip(*first)),second))

        adjusted = np.array([np.interp(np.arange(0, len(r), len(r)/len(straight)), np.arange(0, len(r)), r)[:len(straight)] for r in random]).T

        print(adjusted.dtype)
        print(adjusted.shape,adjusted[0].shape)
        result=[]
        #print(len(straight),[r.shape for r in adjusted])
        for i,value in enumerate(straight):
            err=1
            done = False
            while not done:
                if np.sum(np.logical_and(adjusted[i]<value+err,adjusted[i]>value-err))>(percent/100)*adjusted[i].shape[0]:
                    done=True
                    result.append(err)
                else:
                    err+=1
            print(i,err,np.max(np.abs(adjusted[i]-value)))
        # return result

        print(time.time()-_t)
        return ([self.context.canvas.point_line_distance(point,second) for point in points],straight,result,adjusted[:3])
        #return ([self.context.canvas.point_line_distance(point,second) for point in points],straight)

    def _generate_lines(self,_labels,inertia=2,pos_range=120,neg_range=50,inhibition_coef=1,attraction_coef=1,age_coef=0.1,age_cut_off_coef=4,straight_coef=10,peak_coef=0.5,draw=False):
        result=[]
        other_pos=[]
        difference=[]
        if self.context.data_context.values['center'] is not None:
            center=self.context.canvas._stack_coord_to_micron(self.context.data_context.values['center'])
        else:
            center = (0,0,0)
            size = 0
            for i in range(-2,4):
                point=self.context.data_context.get_point_from_label(str(i))
                if point is not None:
                    center = tuple(a+b for a,b in zip(center,point))
                    size+=1
            center = tuple(a/size for a in center)
            center=self.context.canvas._stack_coord_to_micron(center)
        for labels in _labels:
            lines=[]
            try:
                for l in labels[1:]:
                    point=self.get_points_from_label(l)[0]
                    graph=find_compact_component(self.context.canvas.skeleton_graph,point)

                    lines.append((self.context.canvas._lines_coord_to_microns(graph_to_lines(graph)),int(l)))

                    #_start=self.get_points_from_label(f'START {labels[0]}')[0]
            except IndexError:
                continue
            def _calculation(sufix):
                _start=self.get_points_from_label(f'{labels[0]}{sufix}s') 
                if not _start:
                    return
                _start=_start[0]
                start=self.context.canvas._stack_coord_to_micron(_start)
                #_target = self.get_points_from_label(f'END {labels[0]}')
                _target=[]
                for label in [f'{labels[0]}e',f'{labels[0]}{sufix}e',f'{labels[0]}']:
                    _target = self.get_points_from_label(label)
                    if _target:
                        _target=_target[0]
                        break
                if not _target:
                    return
                #print(_target)
                target=self.context.canvas._stack_coord_to_micron(_target)
                #print(target,start)
                t=time.time()
                original_line=[]
                graph=find_compact_component(self.context.canvas.skeleton_graph,_target)
                original_line=(self.context.canvas._lines_coord_to_microns(graph_to_lines(graph)))

                _result,_other_pos=generate_line.generate_model_line(start,target,original_line,lines,inertia,pos_range,neg_range,inhibition_coef,attraction_coef,age_coef,age_cut_off_coef,straight_coef,peak_coef,center)
                #print(f'time taken {time.time()-t}')
                t=time.time()
                cutoff=-1
                if len(_result)>=1000:
                    for i in range(100,970):
                        distance=lambda start,end:sum(((x - y)) ** 2 for x, y in zip(start, end)) ** 0.5
                        try:
                            if distance(_result[i],result[i+20])<5:
                                cutoff=i
                                break
                        except IndexError:
                            pass

                if cutoff>-1:
                    _result=_result[:cutoff]
                    print(f'cutting in {cutoff}')
                
                
                original=self.context.canvas._lines_coord_to_microns(find_primary(self.context.canvas.skeleton_graph,_start,_target))
                #print(original)
                #print(original)
                end_distance=sum(((x - y)) ** 2 for x, y in zip(_result[-1], target))
                distances=generate_line.calculate_distance(_result,original)
                max_d=max([d**2 for d in distances])
                try:
                    avg_d=sum([d**4 for d in distances])/len(distances)
                except:
                    avg_d=sum([d**4 for d in distances])
                #difference.append(sum(distance**2 for distance in distances)/len(distances)**2+end_distance**3/27)
                #difference.append(max_d+end_distance)

                #normalize by line length
                l=sum(((x - y)) ** 2 for x, y in zip(start, target))*0.5

                difference.append(avg_d**0.5/l+end_distance)
                #print('result:',[i for i in _result if len(i)!=3 or type(i)!=tuple],'\n\n\n')
                if draw:
                    _result=[self.context.canvas._micron_coord_to_stack(i) for i in _result]
                    _other_pos=[self.context.canvas._micron_coord_to_stack(i) for i in _other_pos]
                    result.append(_result)
                    other_pos.extend(_other_pos)
                    self.context.graphic_context.mark_temporary_paths(result)
            
            _calculation('')
            _calculation('r')
            _calculation('l')

            
            #print(f'time taken 2 {time.time()-t}')
            #print('other')
            #print([(round(x,2),round(y,2),round(z,2)) for x,y,z in other_pos])
        if draw:
            self.context.graphic_context.mark_temporary_paths(result)
            self.context.graphic_context.mark_temporary_points(other_pos)
    
        # d_value=sum([d**2 for d in difference])/len(difference)
        # d_value=d_value**0.5
        d_value=max(difference)
        #print(d_value)
        return result,d_value



    def _test(self,steps=1000,values={'strongest':5,'attraction_coef':3,'neg_range':30,'straight_coef':5,'inertia':30}):
        #for 90 labels = [(str(i),str(i+8),str(i+13)) for i in [-4,-3,-2,-1]] 
        #for 87 labels = [(str(i),str(i+8),str(i+13)) for i in [-4,-3,-2,-1]]
        files=[]
        file_numbers=[]
        labels_list=[]
        from calculation_config import only90
        for x in only90:
            _file_numbers=x.file_numbers
            file_numbers.extend(_file_numbers)
            number=x.number
            for path in [
                f'C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{number}_dr{n}flat3.ske' for n in _file_numbers]:
                with open(path,'rb') as file:
                    files.append(pickle.load(file))
            #labels_list = [[(str(i),str(i+8),str(i+13)) for i in l] for l in [[-3,-2,-1],[-3,-2],[-5,-4,-3,-2,-1]]]
            #labels_list.append([('-4', '4', '9'), ('-3', '5', '10'), ('-1', '4', '7')])
            labels_list.extend(x.labels_list)
        def perform_calc(labels,v):
            result=0
            for number,file,labels in zip(file_numbers,files,labels_list):
                #print(number)
                if number==50 or number==48 or number==31 or number==43:
                    continue
                self._load_data_no_graphics(file)
                _,_d=self._generate_lines(labels,**v)
                result+=_d
            return result/len(files)
        #x,d_value=self._generate_lines(labels,strongest=5,attraction_coef=3,neg_range=30,straight_coef=5,inertia=30)
        d_value=perform_calc(labels_list,values)
        
        #values={'strongest':5,'attraction_coef':3,'neg_range':30,'straight_coef':5,'inertia':30}
        #values={'strongest':5,'attraction_coef':3,'neg_range':30,'straight_coef':5}
        min_step={'pos_range': 2, 'strongest': 0.1, 'age_coef': 0.01, 'inhibition_coef': 0.1, 'attraction_coef': 0.1, 'neg_range': 1, 'straight_coef': 0.1, 'inertia': 0.2,'peak_coef':0.02,'age_cut_off_coef':0.05}
        min_value={'pos_range': 2, 'strongest': 0.1, 'age_coef': 0.01, 'inhibition_coef': 0.01, 'attraction_coef': 0.01, 'neg_range': 1, 'straight_coef': 0.3, 'inertia': 0.5,'peak_coef':0.01,'age_cut_off_coef':-10}
        #min_value={'pos_range': 2, 'strongest': 0.1, 'age_coef': 0.00, 'inhibition_coef': 0.01, 'attraction_coef': 0.01, 'neg_range': 1, 'straight_coef': 0.3, 'inertia': 0.5,'peak_coef':0.01}
        max_value={'pos_range': 300, 'age_coef': 10.00, 'inhibition_coef': 20., 'attraction_coef': 20., 'neg_range': 100, 'straight_coef': 20, 'inertia': 500,'peak_coef':0.99,'age_cut_off_coef':20}

        def next_steps(values,min_step=min_step,pick=5):
            other=sample(range(len(values.keys())),len(values.keys())-pick)
            result= [{key:max(min_value[key],min(max_value[key],val+x[i]*max(val*0.05,min_step.get(key,0)))) for i,(key,val) in enumerate(values.items())} 
                            for x in product([-1,0,1],repeat=len(values.keys())) if len([v for v in other if x[v]!=0])==0]
            
            duplicates=[]
            for i,item in enumerate(result):
                for j,second in enumerate(result[i+1:]):
                    same=True
                    for (k,v) in item.items():
                        if second[k]!=v:
                            same=False
                            break
                    if same:
                        duplicates.append(j)
            result = [i for j, i in enumerate(result) if j not in duplicates]
            return result
        possible_values=next_steps(values)
        jumps=0
        file=open('results.txt',mode='a')
        file.write(f'mew test\n')
        file.close()
        visited=[]
        t=time.time()
        for j in range(10):
            tries=0
            for _i in range(steps):
                print('time',time.time()-t)
                temp_d=[]
                print(len(possible_values))
                for _v in possible_values:
                    _d=perform_calc(labels_list,_v)
                    temp_d.append(_d)
                best_d=min(temp_d)
                best_index=temp_d.index(best_d)
                best=possible_values[best_index]
                if (best_d/d_value>0.9995):
                    tries+=1
                    print(f'nothing better found for {tries} time(s)')
                else:
                    print(f'step {_i} best so far {min(temp_d)}')
                    tries=0
                    jumps=0
                    file=open('results.txt',mode='a')
                    file.write(f'{best}\n')
                    d_value=best_d
                    possible_values=next_steps(best)
                    file.close()
                if tries>5 or jumps>0:
                    jumps+=1
                    break
            if tries>5 or jumps>0:
                print(f'jump, try number {j}')
                if jumps==1:
                    file=open('results.txt',mode='a')
                    file.write(f'best so far {best_d}\n')
                    file.close()
                    possible_values=[]
                    for key in (best.keys()):
                        for x in [-1,1]:
                            result=dict(best)
                            val=result[key]
                            result[key]=max(min_value[key],min(max_value[key],val+x*max(val*0.05,min_step.get(key,0))))
                            possible_values.append(result)
                    
                if jumps>1:
                    possible_values=next_steps(best,min_step={k:v*(1+(jumps-1)*2) for k,v in min_step.items()},pick=len(best))
                    possible_values=sample(possible_values,1200)
                    possible_values=list(filter(lambda x: x not in visited, possible_values))
                    visited.extend(possible_values)
                    _possible_values=[]
                    _min_step={k:v*(10+(jumps)*4) for k,v in min_step.items()}
                    for key in (best.keys()):
                        for x in [-1,1]:
                            result=dict(best)
                            val=result[key]
                            result[key]=max(min_value[key],min(max_value[key],val+x*max(val*0.05,_min_step.get(key,0))))
                            _possible_values.append(result)
                    _possible_values=list(filter(lambda x: x not in visited, _possible_values))
                    possible_values.extend(_possible_values)
                        
                best_d=best_d*1.1
        self._generate_lines(labels_list[-1],**best,draw=True)
        self.context.canvas.redraw_graph()
        self.reload_all()
        return best

        

    

    def _key(self,typ):
        def f(value):
            label=self.labels[typ][value]
            if len(label)==0:
                return 'zzzzzzz'
            else:
                return label
        return f

    def set_label(self,type,pos,label):
        self.labels[type][pos]=label



    def add_label(self,point,name=""):
        if point not in self.values['label']:
            self.values['label'].append(point)
        self.labels['label'][point]=self.labels['point'][point]
        self.reload_all()

    def add_lines(self,lines,label=""):
        for start,end in lines:
            self.add_points((start,end))
            if (start,end) in self.hidden or (end,start) in self.hidden:
                try:
                    self.hidden.remove((start,end))
                except KeyError:
                    self.hidden.remove((end,start))
            elif (end,start) not in self.values["line"] and (start,end) not in self.values["line"]:
                self.context.canvas.connect_points(start,end)
                self.values["line"].append((start, end))
                self.labels["line"][(start,end)]=label
        self.reload_all()

    def add_points(self,points,label=""):
        for point in points:
            if point in self.hidden:
                self.hidden.remove(point)
            elif point not in self.values['point']:
                self.values["point"].append(point)
                self.labels["point"][point]=label
            self.context.canvas.add_point(point)
        self.reload_all()

    def set_center(self,point):
        if point in self.values['point'] and not self.context.canvas.skeleton_graph[point]:
            self.values['point'].remove(point)
            del self.labels['point'][point]
            del self.context.canvas.skeleton_graph[point]
        self.values['center']=point
        self.reload_all()

    def save_data(self,file):
        save={'graph':self.context.canvas.skeleton_graph,'labels':self.labels,'values':self.values,'coord_data':{'scaling_ratio':self.context.canvas.scaling_ratio,'shape':self.dr_stack_shape,'unit_to_microns_coef':self.context.canvas.unit_to_microns_coef,'permuation':(1,0,2)},'hidden':self.hidden}
        pickle.dump(save,file)

    def _load_data_no_graphics(self,save):
        values=save['values']
        hidden=set()
        if save.get('coord_data',None)!=None and (self.dr_stack_shape == None or type(self.dr_stack)==type(None)):
            self.dr_stack_shape=save['coord_data']['shape']
        if self.context.canvas.scaling_ratio==None or type(self.dr_stack)==type(None):
            data = save.get('coord_data',None)
            if data:
                self.context.canvas.scaling_ratio=data.get('scaling_ratio',None)
                self.context.canvas.unit_to_microns_coef=data.get('unit_to_microns_coef',None)

        if 'label' not in self.values:
            values['label']=[]
        if 'hiden' in save:
            hidden=save['hidden']
        if 'labels' in save:
            self.labels=save['labels']
            self.values=values
        else:
            for typ,l in values.items():
                print(typ)
                if typ != 'center':
                    self.values[typ]=[]
                    for (label,value) in l:
                        self.values[typ].append(value)
                        self.labels[typ][value]=label
                else:
                    self.values['center']=l
        self.hidden=hidden
        self.context.canvas.skeleton_graph=save['graph']


    def load_data(self,save):
        values=save['values']
        hidden=set()
        if save.get('coord_data',None)!=None and (self.dr_stack_shape == None or type(self.dr_stack)==type(None)):
            self.dr_stack_shape=save['coord_data']['shape']
        if self.context.canvas.scaling_ratio==None or type(self.dr_stack)==type(None):
            data = save.get('coord_data',None)
            if data:
                self.context.canvas.scaling_ratio=data.get('scaling_ratio',None)
                self.context.canvas.unit_to_microns_coef=data.get('unit_to_microns_coef',None)

        if 'label' not in self.values:
            values['label']=[]
        if 'hiden' in save:
            hidden=save['hidden']
        if 'labels' in save:
            self.labels=save['labels']
            self.values=values
        else:
            for typ,l in values.items():
                print(typ)
                if typ != 'center':
                    self.values[typ]=[]
                    for (label,value) in l:
                        self.values[typ].append(value)
                        self.labels[typ][value]=label
                else:
                    self.values['center']=l
        self.context.canvas.reload_skeleton_graph(save['graph'])
        self.hidden=hidden
        self.reload_all()


    def remove_line(self,start,end):
        self.values['line'].remove((start,end))
        self.context.canvas.remove_line(start,end)
        if (start,end) in self.hidden:
            self.hidden.remove((start,end))
        self.reload_all()


    def remove(self,values,typ):
        #print('VALUES', values)
        for p in values:
            while p in self.values[typ]:
                self.values[typ].remove(p)
            if p in self.labels[typ]:
                del self.labels[typ][p]
            if p in self.hidden:
                self.hidden.remove(p)
            if typ=='point':
                self.context.canvas.remove_point(p)
                for start, end in copy.copy(self.values['line']):
                    if start == p or end == p:
                        self.values['line'].remove((start,end))
                        if (start,end) in self.hidden:
                            self.hidden.remove((start,end))
        self.reload_all()
            
    def create_skeleton_stack(self):
        stack = np.zeros(self.dr_stack_shape,dtype=np.uint16)
        r=2
        lines = graph_to_lines(self.context.canvas.skeleton_graph)
        for pos in self.values['point']:
            stack[int(pos[0])-r:int(pos[0])+r+1,int(pos[1])-r:int(pos[1])+r+1,int(pos[2])-r:int(pos[2])+r+1]=2000
        for start,end in lines:
            l=int(ceil(5*max(abs(a-b) for a,b in zip(start,end))))
            a = np.array(start)
            b = np.array(end)
            for i in range(l+1):
                pos=a*(1-i/l)+(i/l)*b
                try:
                    stack[int(pos[0])-r:int(pos[0])+r+1,int(pos[1])-r:int(pos[1])+r+1,int(pos[2])-r:int(pos[2])+r+1]=4000
                except Exception as e:
                    print(e)
                    print(a,b)
                    print(pos)
        return stack
        

    
    def check_heightmap_cache(self,point):
        if type(self.heightmap) == type(None):
            self.heightmap = np.zeros((self.pi_stack.shape[0],self.pi_stack.shape[1]))
        else:
            return self.heightmap[point[0],point[1]]>0 


    def compute_height_distance(self,point):
        coords = (int(point[0]),int(point[1]),int(point[2]))
        # if notself.check_heightmap_cache(point):
        #     return self.heightmap[coords[0],coords[1]]
        line = self.pi_stack[coords[0],coords[1],:]
        z = line.shape[0]-1
        max_value = np.amax(line)
        start = -1
        end = -1
        for i in range(z,0,-1):
            if start<0 and line[i]>max_value*0.8:
                start = i
            if start>0 and line[i]<max_value*0.5:
                end = i
                break
        #self.heightmap[point[0],point[1]] = (start*0.7+end*0.3)
        return (start*0.7+end*0.3)
        #self.compute_heightmap()
        #return self.heightmap[point[0],point[1]]

    def concentration_on_surface(self,point,r=10,depth=10):
        result = np.zeros((2*r+1,2*r+1,depth))
        for i,j in product(range(-r,r+1),repeat=2):
            height=int(self.compute_height_distance(point))
            result[i+r,j+r,:]=self.dr_stack[point[0]+i,point[1]+j,height-depth:height]
        return result

    def calculate_approx_survace(self,point,r=3,depth=10):
        c = self.concentration_on_surface(point,r,depth)
        return np.sum(np.max(c,axis=2))/(c.shape[0]*c.shape[1])

    def calculate_approx_survace_with_average(self,point,r=3,depth=20):
        c = self.concentration_on_surface(point,r,depth)
        return (np.sum(np.max(c,axis=2))/(c.shape[0]*c.shape[1]),np.sum(c)/(c.shape[0]*c.shape[1]*c.shape[2]))

    def calculate_concentration_height(self,point,depth=10,value=2000):
        height=int(self.compute_height_distance(point))
        values=self.dr_stack[point[0],point[1],height-depth:height]
        #print(f"values {np.max(values)>value}")
        for i in reversed(range(0,depth)):
            if values[i]>value:
                print(f"picked {values[i]} in {depth-i}")
                return depth-i
        return depth+100
    
    def calculate_min_concentration_height(self,point,r=3,depth=20,value=2000):
        #print("start")
        heights = list(self.calculate_concentration_height((point[0]+i,point[1]+j,point[2]),depth,value) for i,j in product(range(-r,r+1),repeat=2))
        filtered = list(filter(lambda x:x<100,heights))
        if filtered:
            return min(heights),sum(filtered)/len(filtered)
        else:
            return min(heights),120

    def load_from_global_simulation(self):
        skeleton,values,labels,color_map = self.global_simulation.get_data()
        self.values=values
        self.labels=labels
        self.context.canvas.reload_skeleton_graph(skeleton)
        self.context.simulation_controller.set_color_map(color_map)
        self.hidden=set()

    def init_global_simulation(self,labels=[str(i) for i in range(-4,15)]):
        self.global_simulation.init(self.context.canvas.skeleton_graph,labels)



class GraphicContext():
    def __init__(self):
        self.skeleton_lines = ([],[])
        self.marked_selection_points = []
        self.marked_selection_lines = []
        self.marked_temporary_points = []
        self.marked_temporary_line = []
        self.context=Context()
        self.curve_tension=0.0

    def refresh(self):
        self.context.canvas.Refresh()

    def set_graph_lines(self,lines,tangents):
        self.skeleton_lines=(lines,tangents)
        lines,tangents=self.temporary_lines()
        self.context.canvas.skeleton_bundle.set_lines(self.skeleton_lines[0]+lines,self.skeleton_lines[1]+tangents)
        self.context.canvas.skeleton_bundle.mark_points(self.marked_selection_points+self.marked_temporary_points)
        self.context.canvas.skeleton_bundle.mark_lines(self.marked_selection_lines+lines)
        self.refresh()

    def redraw_graph(self):
        self.context.canvas.redraw_graph()
        self.refresh()

    def mark_selection_lines(self,lines):
        self.marked_selection_lines=self.context.canvas.get_selected_lines(lines)
        lines,tangents=self.temporary_lines()
        self.context.canvas.skeleton_bundle.mark_lines(self.marked_selection_lines+lines)
        self.refresh()
    
    def set_lines_color(self,lines,color):
        lines=self._lines_to_opengl_coord(lines)
        self.context.canvas.skeleton_bundle.set_lines_color(lines,color)
        self.refresh()

    def mark_selection_points(self,points):
        self.marked_selection_points=self._points_to_opengl_coord(points)
        self.context.canvas.skeleton_bundle.mark_points(self.marked_selection_points+self.marked_temporary_points)
        self.refresh()

    def _points_to_opengl_coord(self,points):
        return list(map(self.context.canvas._stack_coord_to_opengl,points))

    def _lines_to_opengl_coord(self,lines):
        return self.context.canvas._lines_coord_to_opengl(lines)
    
    def mark_temporary_paths(self,paths):
        self.marked_temporary_line=paths
        points=[]
        for path in paths:
            points.extend(path)
        if hasattr(self.context, 'image_slicer'):
            self.mark_slides_temporary_points(points)
        self.redraw_graph()
        
    def mark_slides_temporary_points(self,points):
        self.context.image_slicer.temporary_points=points

    def mark_slice_points(self,points):
        self.context.image_slicer.temporary_points=points

    def mark_temporary_points(self,points):
        self.marked_temporary_points=self._points_to_opengl_coord(points)
        #print(points)
        self.context.canvas.skeleton_bundle.mark_points(self.marked_selection_points+self.marked_temporary_points)
        self.refresh()

    def calculate_path_tangent(self,path):
        def p_fun(t,a,b,m,i):
            p = (2*t**3 - 3*t**2+1) *a + (t**3 - 2*t**2+t)*m[i]+((-2)*t**3 + 3*t**2)*b+(t**3 - t**2)*m[i+1]
            return p
        tangents=[]
        m=[]
        points = [path[0]]+path+[path[-1]]
        for a,b in zip(map(np.array,points[:-1]),map(np.array,points[2:])):
            m.append(tuple((self.context.graphic_context.curve_tension)*(b-a)*(2)))
        for (m1,m2) in zip(m[:-1], m[1:]):
            tangents.append((m1,m2))
        return tangents

    def clear(self):
        self.marked_selection_points = []
        self.marked_selection_lines = []
        self.marked_temporary_points = []
        self.marked_temporary_line = []
        self.redraw_graph()

    def temporary_lines(self):
        lines = []
        tangents = []
        for path in self.marked_temporary_line:
            if len(path)>=2:
                p=list(map(self.context.canvas._stack_coord_to_opengl,path))
                t=self.calculate_path_tangent(p)
                lines.extend(self._path_to_lines(p))
                tangents.extend(t)
        return (lines,tangents)
    

    def _path_to_lines(self,path):
        return list(zip(path,path[1:]))

    def set_zoom_scale_field(self,scale):
        self.context.side_bar.scale_label.SetLabel(f'{scale:.2f}')



        


context=Context()
context.data_context=DataContext()
context.graphic_context=GraphicContext()

def image_to_array(img: Image) -> np.array:
    dtype = np.uint16
    w, h = img.size
    array = np.zeros((h, w, img.n_frames), dtype=dtype)
    for i in range(img.n_frames):
        img.seek(i)
        array[:, :, i] = np.array(img)
    return array



class CanvasMockup():
    pass
    def __init__(self) -> None:
        self.scaling_ratio = None
        self.skeleton_graph = None

    def add_point(self,point):
        if point not in self.skeleton_graph:
            self.skeleton_graph[point]=set()

def add_surface_points(path):
    data_context = DataContext()
    data_context.context.canvas = CanvasMockup()
    with open(path+".ske","rb") as file:
        data_context._load_data_no_graphics(pickle.load(file))
    img = Image.open(path+".tif")
    data_context.pi_stack=image_to_array(img)


    for old_point,label in list(data_context.labels['point'].items()):
        if label:
            x=data_context.compute_height_distance(old_point)
            new_point= (old_point[0],old_point[1],x)
            name=label+"-s"
            data_context.add_points([new_point],label=name)
            data_context.add_label(new_point,name)  

        
    with open(path+"_surface.ske","wb") as file:
        data_context.save_data(file)

files =[filename for filename in glob.iglob('./**/**', recursive=True)]

for _path in filter(lambda x: "tif" in x[-4:], files):
    if ".tiff" == _path[-5:]:
        path = _path[:-5]
    if ".tif" == _path[-4:]:
        path = _path[:-4]
    else:
        continue
    if path+".ske" in files:
        add_surface_points(path)