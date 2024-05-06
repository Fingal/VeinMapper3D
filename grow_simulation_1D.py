import random
from math import ceil
positions = []
time = 0
#start_position = 45.3
dt=0.001
coeffs={'A': 0.11900277137012134, 'B': 5.910352308336133, 'C': 50.38910353160223}
coeffs={'A': 0.11900277137012134, 'B': 5.910352308336133, 'C': 66.38910353160223}

def d_position(time):
    return time*coeffs['A']+coeffs['B']

def new_primordium(start_position : float):
    positions.append((start_position,0))

def time_step():
    global time
    if time%1>(time+dt)%1:
        new_primordium(random.normalvariate(coeffs['C'],3))
    global positions
    positions = [(d_position(age)*dt+distance,age+dt) for distance,age in positions]
    time+=dt


def do_steps(time_steps,steps):
    for _ in range(steps):
        for _ in range(time_steps):
            time_step()
        print(positions)

#new_primordium(random.normalvariate(coeffs['C'],3))
do_steps(ceil(1/dt),10)