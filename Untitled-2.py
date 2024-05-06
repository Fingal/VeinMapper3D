def get_closest(b):
    a=min(gs.points,key=lambda x: (Vector3(x)-Vector3(c.canvas._stack_coord_to_micron(b))).length)
    return gs.points.index(a)
def display(*x):
    if len(x)>1:
        c.graphic_context.mark_temporary_paths([[c.canvas._micron_coord_to_stack(xx) for xx in x]])
    c.graphic_context.mark_selection_points([c.canvas._micron_coord_to_stack(xx) for xx in x])


def display_star(*x):
    c.graphic_context.mark_temporary_paths([[c.canvas._micron_coord_to_stack(x[0]),c.canvas._micron_coord_to_stack(xx)] for xx in x[1:]])

def make_gs_copy():
    global gs_copy 
    gs.context=None
    gs_copy=copy.deepcopy(gs)
    gs.context=c

def recover_gs():
    global gs
    gs=copy.deepcopy(gs_copy)
    global c
    c.data_context.global_simulation=gs
    gs.context=c 

calculate_n_8_adult(c.simulation_controller)
