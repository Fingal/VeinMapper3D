import subprocess
import copy
from typing import Optional



def run_space_exploration_negative(names,steps,suffix='max'):
    _min_value={'pos_range': 20, 'strongest': 0.1, 'age_coef': 0.01, 'inhibition_coef': 0.01, 'attraction_coef': 0.01, 'neg_range': -0.01, 'straight_coef': 0.3, 'inertia': 0.5,'peak_coef':0.01,'age_cut_off_coef':-10}

    _max_value={'pos_range': 140, 'age_coef': 10.00, 'inhibition_coef': 0.01, 'attraction_coef': 5., 'neg_range': 2, 'straight_coef': 20, 'inertia': 500,'peak_coef':0.99,'age_cut_off_coef':20}

    processes=[]
    for name in names:
        for i in range(steps):
            #nums=[10,20,20,20],out_filename='exploration_results\\dump_old_4',min_value=None,max_value=None
            _min_value['pos_range']=20+(120)/steps*i
            _max_value['pos_range']=20+(120)/steps*(i+1)
            processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py", f'[{int(20/steps)},20,20,1]',f'exploration_results\\dump_{name}_{suffix}_neg_{i}',name,str(_min_value),str(_max_value)]))
        print(f'{name} started')
        for p in processes:
            p.wait()
def run_space_exploration(names,steps,suffix='max'):
    _min_value={'pos_range': 20, 'strongest': 0.1, 'age_coef': 0.01, 'inhibition_coef': 0.01, 'attraction_coef': 0.01, 'neg_range': 0.01, 'straight_coef': 0.3, 'inertia': 0.5,'peak_coef':0.01,'age_cut_off_coef':-10}

    _max_value={'pos_range': 140, 'age_coef': 10.00, 'inhibition_coef': 5, 'attraction_coef': 5., 'neg_range': 100, 'straight_coef': 20, 'inertia': 500,'peak_coef':0.99,'age_cut_off_coef':20}

    processes=[]
    for name in names:
        for i in range(steps):
            #nums=[10,20,20,20],out_filename='exploration_results\\dump_old_4',min_value=None,max_value=None
            _min_value['pos_range']=25+(120)/steps*i
            _max_value['pos_range']=25+(120)/steps*(i+1)
            processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py", f'[{int(20/steps)},10,10,10,5,5]',f'exploration_results\\dump_{name}_{suffix}_{i}',name,str(_min_value),str(_max_value)]))
        print(f'{name} started')
        for p in processes:
            p.wait()
processes=[]
def run_optimalization(config : str, in_file : str,start : int,end : int,out_file: Optional[str]=None,processes=[]):
    if out_file is None:
        out_file=in_file+'result'
    processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py",config, in_file,out_file,str(start),str(end)]))

def run_optimalization_on_given(config : str, start_value : dict,out_file: Optional[str]=None,processes=[]):
    if out_file is None:
        out_file=in_file+'result'
    processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py",config, str(start_value),out_file]))

def run_from_file(config : str, in_file : str,start : int,end : int,out_file: Optional[str]=None,processes=[]):
    if out_file is None:
        out_file=in_file+'result'
    processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py",config, in_file,out_file,str(start),str(end)]))
def optimalization_main():
    steps=10
    for name in ['dr7','dr8','dr9']:
        processes=[]
        for j in range(4):
            start = 5*j
            end = 5*(j+1)
            print(start,end)
            for i in range(start,end):
                run_optimalization(name,f'exploration_results\\dump_{name}_max_all_n',10*i,10*(i+1),f'exploration_results\\optimalization\\dump_{name}_max_{i}',processes=processes)
            for p in processes:
                p.wait()
            print(f"{name} finished!")
            print(f"{name} finished!")
            print(f"{name} finished!")
            print(f"{name} finished!")

def optimalization_main_2():
    steps=10
    for names in [['dr7','dr8','dr9','dr34'],['dr35','dr37','dr79','dr86','dr87']]:
        processes=[]
        for name in names:
            run_optimalization(name,f'exploration_results\\dump_{name}_max_all_n',0,40,f'exploration_results\\optimalization\\dump_{name}_avg',processes=processes)
        for p in processes:
            p.wait()

def optimalization_main_3():
    steps=10
    for names in [['old','older','young']]:
        processes=[]
        for name in names:
            run_optimalization(name,f'exploration_results\\dump_{name}_max_all_n',0,40,f'exploration_results\\optimalization\\dump_{name}_max',processes=processes)
        for p in processes:
            p.wait()


def explore_generated():
    processes=[]
    batch_size=320000//5
    for i in range(5):
        run_from_file("old",f'exploration_results\\to_test',batch_size*i,batch_size*(i+1),f'exploration_results\\PCA\\dump_old_{i}',processes=processes)
    for p in processes:
        p.wait()

def optimize_PCA():
    import pickle
    result = []
    for i in range(4):
        with open(f"exploration_results\\PCA\\dump_old_{i}","rb") as file:
            result.extend(pickle.load(file))
    result = list(filter(lambda x: x[1]<300,result))
    result = list(map(lambda x:x[0],sorted(result, key=lambda x: x[1])))[:100]
    print(len(result))
    name = "old"
    print(len(result))
    for step in range(len(result)//5):
        print(f"\n\nNow processing {step}\n")
        processes=[]
        for i in range(5):
            run_optimalization_on_given(name,result[step*5+i],f'exploration_results\\PCA\\optimalization\\dump_{name}_max_{step*5+i}',processes=processes)
        for p in processes:
            p.wait()

def explore(sample_name,path,filename,threads,samples):
    if path[-1]!="\\" or path[-1]!="/":
        path=path+"\\"
    processes=[]
    batch_size=samples//threads+1
    for i in range(threads):
        run_from_file(sample_name,f'{path}{filename}',batch_size*i,batch_size*(i+1),f'{path}{filename}_{i}',processes=processes)
    for p in processes:
        p.wait()


def explore_mediumSAM():
    
    path="C:\\Users\\Andrzej\\Desktop\\2send_1\\Space testing\\mediumSAM\\"
    processes=[]
    batch_size=1711//5+1
    for i in range(5):
        run_from_file("mediumSAM",f'{path}new_explore2',batch_size*i,batch_size*(i+1),f'{path}new_explore2_{i}',processes=processes)
    for p in processes:
        p.wait()

def optimize_mediumSAM():
    processes=[]
    batch_size=10201//5+1
    path="C:\\Users\\Andrzej\\Desktop\\2send_1\\Space testing\\"
    for i in range(5):
        processes.append(subprocess.Popen(["python", "C:\\Users\\Andrzej\\Desktop\\volumetric-2\\CIM.py","mediumSAM",f'{path}to_optimize',f'{path}optimized_new_{i}',str(0+i),str(1000),str(5)]))
    for p in processes:
        p.wait()

def optimize(name,path,filename,items,threads):
    processes=[]
    batch_size=items//threads+1
    for i in range(threads):
        processes.append(subprocess.Popen(["python", ".\\CIM.py",name,f'{path}{filename}',f'{path}{filename}_optimized_{i}',str(0+i),str(items),str(threads)]))
    for p in processes:
        p.wait()

import time

t=time.time()
#explore("mediumSAM","C:\\Users\\Andrzej\\Desktop\\2send_1\\speed test","new_explore2",12,1711)
optimize("bigSAM","C:\\Users\\Andrzej\\Desktop\\2send_1\\bigSAM\\space_exploration\\","new_explore2",1711,5)
print("\n"*2,"FINISHED", time.time()-t)
#optimize_mediumSAM()
#run_space_exploration(['dr34','dr35','dr37','dr79','dr86','dr87'],5,suffix='max')
#run_space_exploration_negative(['dr34','dr35','dr37','dr79','dr86','dr87',],5,suffix='max')
#explore_generated()
#optimize_PCA()
#explore_mediumSAM_PCA()

#run_space_exploration(['mediumSAM'],5,suffix='max')
#run_space_exploration(['singleMediumSAM'],5,suffix='max')

#optimize_mediumSAM()