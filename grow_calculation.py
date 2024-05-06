import numpy as np
import math
from itertools import product

def to_array(text:str):
    result = np.array([list(map(float,i.split())) for i in text.split('\n')])
    result = result - (result==0)*1
    return result.T

def calculate_distance(arr):
    distances = []
    for x  in arr:
        result=[]
        for a,b in zip(x,x[1:]):
            if a==-1 or b==-1:
                result.append(-1)
            else:
                result.append(b-a)
        distances.append(result)
    return distances

coeffs={'a':41.5,'pow':0.11,'transposition':1.0}
square_coeffs = {'':0.3358,'b':5.5424,'c':38}
square_step = {'A':0.003,'B':0.05,'C':0.1}
line_coeffs = {'a':10,'b':26}
line_step = {'a':0.01,'b':0.2}
def exp2(i,a,pow,transposition):
    return a*math.e**(pow*(i+transposition))

def exp(i,a,pow,transposition):
    return a*math.e**(pow*(i))

def square(i,A,B,C):
    return A*i**2+B*i+C

def line(i,a,b):
    return a*i+b


def age_mapping(i):
    return (i-1)*2+1


def calculate_error(arr: np.array,function,coeffs,transpositions=None,age_mapping=None):
    if transpositions is None:
        transpositions=[0]*arr.shape[0]
    error=0
    for j,line in enumerate(arr):
        for i,x in enumerate(line):
            if age_mapping:
                age = age_mapping(i+transpositions[j])
            else:
                age = i-1+transpositions[j]
            if x!=-1:
                error+=(function(age,**coeffs)-x)**2
    return error

def calculate_transition(arr,function,coeffs,r=1,age_mapping=None):
    transpositions=[]
    for line in arr:
        transposition=0
        error=1000000
        for new_transposition in np.linspace(0,r,100):
            new_error=0
            for i,x in enumerate(line):
                if x!=-1:
                    if age_mapping:
                        age = age_mapping(i+new_transposition)
                    else:
                        age = i-1+new_transposition
                    new_error+=(function(age,**coeffs)-x)**2
            print(new_error)
            if new_error<error:
                error = new_error
                transposition = new_transposition
                
        transpositions.append(transposition)
    return transpositions
            

def get_distances_from_labels(context,center,labels):
    data_context : DataContext = context.data_context
    
    center = context.canvas._stack_coord_to_micron(center)
    center = np.array((center[0],center[2]))

    points = [context.canvas._stack_coord_to_micron(data_context.get_point_from_label(label)) for label in labels]
    points = [np.array((point[0],point[2])) for point in points]
    points = [np.linalg.norm(point-center) for point in points]
    return points

def find_coeff(arr,function,coeffs,min_step=square_step,steps=1000,transpositions=None,age_mapping=None):
    error=calculate_error(arr,function,coeffs,transpositions)
    for i in range(steps):
        tries=0
        while tries<100:
            tries+=1
            new_errors=[]
            for direction in product((-1,0,1), repeat=len(coeffs)):
                new_coeffs={}
                for (k,v),d in zip(min_step.items(),direction):
                    new_coeffs[k]=v*d*1/tries+coeffs[k]
                new_errors.append((calculate_error(arr,function,new_coeffs,transpositions,age_mapping),new_coeffs))
            
            test = min(new_errors,key=lambda x: x[0])
            if test[0]<error:
                error=test[0]
                coeffs=test[1]
                break
        print('error:',error)
        if tries==1000:
            #print(new_errors)
            break
    print(transpositions)
    return coeffs

def labels_to_arr(c,labels,center):
    distances = []
    for label in labels:
        point = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label(label))
        d = ((point[0]-center[0])**2+(point[2]-center[2])**2)**0.5
        distances.append(d)
    return np.array([distances])


#0.03500000014901161 0.03999999910593033 22.799999237060547 2.4200000762939453 0.14000000059604645 2.240000009536743 54.29999923706055
#0.03500000014901161 0.03999999910593033 22.799999237060547 2.4200000762939453 0.11400000005960464 2.8420000076293945 50.029998779296875
# square_coeffs=find_coeff(arr,square,square_coeffs,min_step={'a':0.003,'b':0.05,'c':0.05},transpositions=transpositions,age_mapping=age_mapping)
# transpositions = calculate_transition(arr,square,square_coeffs,age_mapping=age_mapping)
# 34
# 2679.5992036162725 {'a': 45.50000000000006, 'pow': 0.111}
# 2767.3703409199993 {'a': 0.5218000000000002, 'b': 5.292400000000001, 'c': 50.20000000000017}

# 30
# 3637.6051527497602 {'a': 51.90000000000015, 'pow': 0.106, 'transposition': -0.04000000000000076}
# 3820.9486974 {'a': 0.5278000000000002, 'b': 4.792400000000002, 'c': 51.50000000000019}

# 37
# 9149.114965384842 {'a': 47.70000000000009, 'pow': 0.09699999999999999, 'transposition': 1.5900000000000005}
# 8491.477699480003 {'a': 0.40480000000000005, 'b': 5.3424000000000005, 'c': 54.70000000000024}

#40
# 13737.694275267666 {'a': 55.00000000000019, 'pow': 0.07499999999999997, 'transposition': 2.3499999999999934}
# 11368.158649879999 {'a': 0.20379999999999987, 'b': 5.992399999999999, 'c': 62.60000000000035} {'a':0.2,'b':6.3,'c':69}
# no p-1 11313.06847236  {'a': 0.20979999999999988, 'b': 6.342399999999998, 'c': 68.8000000000001}

# plt.plot(np.linspace(0,15,100),[square(i,**{'a': 0.2098, 'b': 6.3424, 'c': 68.8}) for i in np.linspace(0,10,100)],
#          np.linspace(0,15,100),[line(i,**{'a': 9.29, 'b': 63.60}) for i in np.linspace(0,15,100)],
#          np.linspace(0,15,100),[exp2(i,**{'a': 58.00, 'pow': 0.075, 'transposition': 2.65}) for i in np.linspace(0,15,100)])



a=[-5, -3, -1, 1, 3, 5, 7]

def compute(step_size=0.1,grow_amp=0.1,steps=1000):
    added=[]
    small_size=[]
    avg_distances=[]
    b=list(reversed(a))
    for i in range(steps):
        #print([f'{i:.2f}' for i in b])
        size = len(b)
        b=grow(b,step_size,grow_amp+i/1000)
        if len(b)>size:
            avg_distance=sum(x-y for x,y in zip(b,b[1:]) if y>=0)/len([i for i in zip(b[1:],b) if i[1]>=0])
            avg_distances.append(avg_distance)
            b=[i for i in b if i<=8]
            small_size.append(len(b))
            added.append(i*step_size)
    return b,added,small_size,avg_distances

def grow(x,step_size=0.1,grow_amp=0.1):
    result=[]
    add_new=False
    for i in (x):
        if i<0:
            result.append(i+step_size*(1+grow_amp))
        else:
            result.append(i+step_size)
        if i<3 and i+step_size>3:
            add_new=True

    if add_new:
        result.append(-5)
    return result

b,aa,sizes,avg_distances=compute(step_size=0.1,grow_amp=0.1,steps=1000)

b=grow(b,grow_amp=1)
print([f'{i:.2f}' for i in b if i<8])
[i-j for i,j in zip(aa[1:],aa)]