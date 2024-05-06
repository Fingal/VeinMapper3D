from time import time
from generate_line import *
x=generate_line((0,0,0),(100,100,100),3,1)
y=generate_line((0,0,0),(20,100,100),1,1)
yy=list(zip(y,y[1:]))
print(len(x),len(yy))
print(yy[:3])
t=time()
z=calculate_distance(x,yy)
print(time()-t)
