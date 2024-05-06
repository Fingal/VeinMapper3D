
from random import choices
from typing import *
from context import *
import math
import matplotlib.pyplot as plt
import calculation_config
from pathlib import Path
from skeleton import find_compact_component
import random

def calculate_consecutive_angles(center,labels:List[str]):
    context=Context()
    data_context : DataContext = context.data_context
    
    center = context.canvas._stack_coord_to_opengl(center)
    center = np.array((center[0],center[2]))

    points = [context.canvas._stack_coord_to_opengl(data_context.get_point_from_label(label)) for label in labels]
    points = [np.array((point[0],point[2])) for point in points]
    points = [(point-center)/np.linalg.norm(point-center) for point in points]

    return [math.acos(np.dot(a,b))/math.pi*180 for a,b in zip(points,points[1:])]



    #return [math.acos(np.dot(points[i-1],points[i]))/math.pi*180 for i in range(len(points))]

# removes age from simulation result
def strip_age():
    data_context : DataContext = Context().data_context
    labels = {}
    for k,v in data_context.labels["point"].items():
        x=v.split()
        if len(x)>1:
            labels[k]=x[-1]
        else:
            labels[k]=""
    data_context.labels["point"]=labels

    labels = {}
    for k,v in data_context.labels["label"].items():
        x=v.split()
        if len(x):
            labels[k]=x[-1]
        else:
            labels[k]=""
    data_context.labels["label"]=labels

#c.data_context.values["center"]=c.data_context.get_point_from_label("c")


# converts description from descriptions_skeleton_models.xlsx

def convert_data_description_skeletons_model(text):
    lines = text.strip().split("\n")
    lines2= [line.strip().split("\t") for line in lines]

    line3=[]
    for line in lines2:
        new_line = (int(line[0]),set())
        for p in line[1:]:
            if not p:
                continue
            a,b = p.split("; ")
            new_line[1].add(int(a))
            new_line[1].add(int(b))
        line3.append(new_line)
    return line3


def convert_labels(iv_id: int,other_ids,suffix = ""):
    context = Context()
    start_label = str(iv_id)+"st"+suffix
    start = context.data_context.get_point_from_label(start_label)
    if start is None and suffix=="":
        for x in convert_labels(iv_id,other_ids,suffix="R"):
            yield x
        for x in convert_labels(iv_id,other_ids,suffix="L"):
            yield x
    if start is not None:
        end_label = str(iv_id) + "e"+suffix
        end = context.data_context.get_point_from_label(end_label)
        if end is None:
            end = context.data_context.get_point_from_label(str(iv_id) + "e")
        first = (start,end)

        for other_id in other_ids:
            end_second =  context.data_context.get_point_from_label(str(other_id)+"e"+suffix)
            if end_second is None:
                end_second = context.data_context.get_point_from_label(str(other_id)+"e")
            if end_second is None:
                end_second = context.data_context.get_point_from_label(str(other_id))
            if end_second is not None:
                start_second = context.data_context.get_point_from_label(str(other_id)+"st"+suffix)
                if start_second is None:
                    start_second = context.data_context.get_point_from_label(str(other_id)+"st")
                if start_second is None:
                    points= find_compact_component(context.canvas.skeleton_graph,end_second)
                    start_second = min(points,key=lambda x: (x[0]-start[0])**2+(x[1]-start[1])**2+(x[2]-start[2])**2)
                second = (start_second,end_second)
                yield (first,second,iv_id,other_id,suffix)
        


def calculate_all():
    context=Context()
    #config = calculation_config.mediumSAM()
    config = calculation_config.FigRandomWalk()
    results={"vein end":{},"center":{}}
    for name,labels in zip(config.filenames,config.labels_list):
        for l in labels:
            l = [int(i) for i in l]
            print(name,l)
            context.data_context.load_data(pickle.load(open(f"{config.path}{name}","rb")))
            print(l)
            for first,second,iv_id,other_id,suffix in convert_labels(l[0],l[1:]):
                print("calculations for", first,second,iv_id,other_id,suffix)


                result = context.data_context.sample_random_distance(first,second)



                results["vein end"][name[:-4],iv_id,other_id,suffix]=(*result[:-1],random.choices(result[-1],k=10))

                center = context.data_context.get_point_from_label("c")

                plt.axis([0, 200, 0, 150])
                for random_result in result[-1]:
                    plt.plot(np.linspace(0,len(result[0]),len(random_result)),random_result,c="red",alpha=0.01)

                plt.plot(np.linspace(0,len(result[0]),len(result[-4])),result[-4],c=(0.8,0,0,0.7))

                plt.plot(result[0],c="blue")
                Path(f"{config.path}\\plots\\{name[:-4]}\\vein_end\\{other_id-iv_id}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{config.path}\\plots\\{name[:-4]}\\vein_end\\{other_id-iv_id}\\{name[:-4]}{suffix}_{iv_id} {other_id}",dpi=200)
                plt.clf()
                print(first,",",second,",other_end=",(center[0],center[1],first[1][2]),",center=",center)

                result = context.data_context.sample_random_distance(first,second,other_end=(center[0],center[1],first[1][2]),center=center)


                
                plt.axis([0, 200, 0, 150])
                for random_result in result[-1]:
                    plt.plot(np.linspace(0,len(result[0]),len(random_result)),random_result,c="red",alpha=0.01)

                plt.plot(np.linspace(0,len(result[0]),len(result[-4])),result[-4],c=(0.8,0,0,0.7))

                plt.plot(result[0],c="blue")
                Path(f"{config.path}\\plots\\{name[:-4]}\\center\\{other_id-iv_id}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{config.path}\\plots\\{name[:-4]}\\center\\{other_id-iv_id}\\{name[:-5]}{suffix}_{iv_id} {other_id}",dpi=200)
                plt.clf()

                results["center"][name[:-4],iv_id,other_id,suffix]=(*result[:-1],random.choices(result[-1],k=10))
                results["center"][name[:-4],iv_id,other_id,suffix,"max_distance"]=(max(result[0]),list(map(lambda x: max(x),result[-1])))
                results["center"][name[:-4],iv_id,other_id,suffix,"min_distance"]=(min(result[0]),list(map(lambda x: min(x),result[-1])))
    pickle.dump(results,open(f"{config.path}\\dump","wb"))

def calculate_all_no_plot():
    context=Context()
    config = calculation_config.mediumSAM()
    results={"vein end":{},"center":{}}
    for name,labels in zip(config.filenames,config.labels_list):
        for l in labels:
            l = [int(i) for i in l]
            print(name,l)
            context.data_context.load_data(pickle.load(open(f"{config.path}{name}","rb")))
            print(l)
            for first,second,iv_id,other_id,suffix in convert_labels(l[0],l[1:]):
                print(first,second,iv_id,other_id,suffix)


                result = context.data_context.sample_random_distance(first,second)



                results["vein end"][name[:-4],iv_id,other_id,suffix]=(*result[:-1],random.choices(result[-1],k=10))

                center = context.data_context.get_point_from_label("c")

                result = context.data_context.sample_random_distance(first,second,other_end=(center[0],center[1],first[1][2]),center=center)

                results["center"][name[:-4],iv_id,other_id,suffix]=(*result[:-1],random.choices(result[-1],k=10))
                results["center"][name[:-4],iv_id,other_id,suffix,"max_distance"]=(max(result[0]),list(map(lambda x: max(x),result[-1])))
                results["center"][name[:-4],iv_id,other_id,suffix,"min_distance"]=(min(result[0]),list(map(lambda x: min(x),result[-1])))
    pickle.dump(results,open(f"{config.path}\\dump","wb"))

#comparison between model and original
def compare_model_original():
    context=Context()
    config = calculation_config.mediumSAM()
    results={"vein end":{},"center":{}}
    for name,labels in zip(config.filenames,config.labels_list):
        for l in labels:
            l = [int(i) for i in l]
            print(name,l)
            context.data_context.load_data(pickle.load(open(f"{config.path}\\plots\\{name[:-4]}_e.ske","rb")))
            print(l)
            for first,second,iv_id,other_id,suffix in convert_labels(l[0],l[1:2]):
                print(first,second,iv_id,other_id,suffix)


                # result = context.data_context.compare_random_distance(first,second)



                # results["vein end"][name[:-4],iv_id,other_id,suffix]=(*result[:-1],random.choices(result[-1],k=10))


                # plt.axis([0, 200, 0, 150])
                # for random_result in result[-1]:
                #     plt.plot(np.linspace(0,len(result[0]),len(random_result)),random_result,c="red",alpha=0.01)

                # plt.plot(result[0],c="blue")
                # Path(f"{config.path}\\plots\\{name[:-4]}\\vein_end\\{other_id-iv_id}").mkdir(parents=True, exist_ok=True)
                # plt.savefig(f"{config.path}\\plots\\{name[:-4]}\\vein_end\\{other_id-iv_id}\\{name[:-4]}{suffix}_{iv_id} {other_id}",dpi=200)
                # plt.clf()

                center = context.data_context.get_point_from_label("c")
                result = context.data_context.compare_random_distance(first,other_end=(center[0],center[1],first[1][2]),center=center)


                
                # plt.axis([0, 200, 0, 150])
                # for random_result in result[-1]:
                #     plt.plot(np.linspace(0,len(result[0]),len(random_result)),random_result,c="red",alpha=0.01)

                # plt.plot(result[0],c="blue")
                # Path(f"{config.path}\\plots_model_compare").mkdir(parents=True, exist_ok=True)
                # plt.savefig(f"{config.path}\\plots_model_compare\\{name[:-4]}-{iv_id}",dpi=200)
                # plt.clf()

                results["center"][name[:-4],iv_id]=(*result[:-1],random.choices(result[-1],k=10))
                #return
    pickle.dump(results,open(f"{config.path}\\dump_model_compare","wb"))
                
        
#(((285, 584, 217), (422, 550, 282)), ((293, 595, 223), (357, 633, 375)), -4, 4, '')
#other_end=(437.83000000000004, 586.973, 244.95)

#result = c.data_context.sample_random_distance(((404, 1004, 92), (487, 705, 210)),((144, 886, 132), (464, 658, 348)))
# plt.plot(np.linspace(0,170,len(x)),x)


def _test():
    context=Context()
    result = context.data_context.sample_random_distance(((285, 584, 217), (422, 550, 282)), ((394, 458, 242), (428, 487, 398)),other_end=(437.83000000000004, 586.973, 244.95))
    #result = context.data_context.sample_random_distance(((285, 584, 217), (422, 550, 282)), ((293, 595, 223), (341, 614, 260)),other_end=(437.83000000000004, 586.973, 244.95))
    for random_result in result[-1]:
        plt.plot(np.linspace(0,len(result[0]),len(random_result)),random_result,c="red",alpha=0.01)

    plt.plot(result[0],c="blue")
    plt.show()


def x():
    for xx in result[-1]:
        plt.plot(np.linspace(0,len(result[0]),len(xx)),xx,c="red",alpha=0.05)

P=[[1,1,0], [2,1,0], [2,2,0]]
Q=[[2,2,0], [0,1,1], [2,4,-1]]




lines=["""-6	-1; 2	2; 7
-5	0; 3	3; 8
-4	1; 4	4; 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-4	1; 4	4; 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-5	0; 3	3; 8
-4	1; 4	4; 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-5	0; 3	
-4	1; 4	4; 9
-3	2; 5	
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-4	1; 4		
-3	2; 5	5; 10	2; 10
-2	3; 6		
-1	4; 7		
""",
"""-4	1; 4	
-3	2; 5	
-2	3; 6	
-1	4; 7	7; 12
0	8; 5	
""",
"""-4	1; 4	4; 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	
0	8; 5	
""",
"""-4	1; 4	4, 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-5		3; 8
-4	1; 4	4; 9
-3	2; 5	5; 10
-2	3; 6	6; 11
-1	4; 7	7; 12
0	5; 8	8; 13
""",
"""-5	0; 3	3; 8
-4	1; 4	4; 9
-2	3; 6	6; 11
-1	4; 7	7; 12
"""]