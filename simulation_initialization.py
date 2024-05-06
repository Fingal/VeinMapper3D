from context import *
from global_simulation import *
import generate_line
from shape_calculation import cone_calculation
from surface_points import DoubleSurfacePoint
import copy

# grow_coeffs=[{'A': 0.2614290317628093, 'B': 7.062656265881261, 'C': 35.85128301663676},
# {'A': 0.17592879728901245, 'B': 6.563675767507394, 'C': 56.3787741621418},
# {'A': 0.11900277137012134, 'B': 5.910352308336133, 'C': 66.38910353160223}]
grow_coeffs=[{'A': 0.2614290317628093, 'B': 7.062656265881261, 'C': 35.85128301663676},
{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
#{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
{'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
grow_times = [20,24]
#meristem_distances=[52.94,60.66,64.63]
meristem_distances=[49.94,57.66,61.63]
meristem_distances=[48.0,52.66,60.63]
#meristem_distances=[51.94,59.66,63.63]
n5_ages=[1,10/3,4]
#n5_ages=[1,50/12,4]
n5_ages=[1,45/13,40/13]
#n5_ages=[1*13/12,45/13,40/13]
#n8_ages=[4,2.5,2]
n8_ages=[3,20/13,16/13]
#n8_ages=[3*13/12,20/13,16/13]
surface_frequency = [13,10,8]
#growth_rate = [3.35,3.52,4.2]
growth_rate = [3.5,4,5]
#growth_rate = [3.3, 3.8, 4.8]


def get_start_end(label):
    c = Context()
    start=c.data_context.get_point_from_label(label)
    cc = find_compact_component(c.canvas.skeleton_graph,start)
    points = [k for k,v in cc.items() if len(v)==1]
    return start,*[p for p in points if p!=start]

def growth_coeffs_full():
    return GrowthCoeffDevelopment(grow_coeffs,meristem_distances,grow_times,n5_ages,n8_ages,surface_frequency,growth_rate)

def growth_coeffs_no_change(i):
    return GrowthCoeffDevelopment(grow_coeffs[i-1:i],meristem_distances[i-1:i],grow_times,n5_ages[i-1:i],n8_ages[i-1:i],surface_frequency[i-1:i],growth_rate[i-1:i])

def growth_coeffs_start_at(i):
    return GrowthCoeffDevelopment(grow_coeffs[i-1:],meristem_distances[i-1:],grow_times[i-1:],n5_ages[i-1:],n8_ages[i-1:],surface_frequency[i-1:],growth_rate[i-1:])



def _project_to_cone(point):
    c=Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    return generate_line.project_to_cone(point, gs.center, gs.cone_coeffs["a1"], gs.cone_coeffs["a2"],gs.cone_coeffs["c"],gs.cone_coeffs["tan"])

def get_line_between_points(c,start,end,index):
    gs : GlobalSimulation = c.data_context.global_simulation 

    result = YoungLine(index,(0,0,0))
    result.line = list(map(c.canvas._stack_coord_to_micron,find_path_between_nodes(c.canvas.skeleton_graph,start,end,find_only_primary=True)))
    result.line = [generate_line.project_to_cone(point, gs.center, gs.cone_coeffs["a1"], gs.cone_coeffs["a2"],gs.cone_coeffs["c"],gs.cone_coeffs["tan"]) for point in result.line]
    result.init_direction(None)
    return result

def init_xp41_dr40():
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    gs.cone_coeffs={'a1': 0.0750684470752261, 'a2': 0.06062554570662967, 'c': 44.77150721867286, 'tan': 2.844490099567475}
    gs.growth_coeffs_young = gs.growth_coeffs_old = gs.growth_coeffs = {'A': 1.1542373900443812, 'B': 5.10437764088337, 'C': 36.245233618273865}
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7', '8'])
    gs.grow_new_iv_ages = (1,4)
    in_progres=[]
    in_progres.append(get_line_between_points(c, (626, 654, 97), (461, 586, 136), -2))
    in_progres.append(get_line_between_points(c, (535, 319, 44), (448, 409, 115), -3))
    in_progres.append(get_line_between_points(c, (513, 778, 73), (443, 671, 106), -4))
    in_progres.append(get_line_between_points(c, (529, 359, 65), (468, 430, 115), -5))
    gs.in_progres=in_progres

    
def init_exp16_4iaa_c():
    c = Context()
    gs : GobalSimulation = c.data_context.global_simulation 
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    gs.growth_coeffs_young = gs.growth_coeffs_old = {'A': 0.5931006790621957, 'B': 14.828647340331328, 'C': 64.04020191538628}
    gs.cone_coeffs={'a1': 0.0750684470752261, 'a2': 0.06062554570662967, 'c': 44.77150721867286, 'tan': 2.844490099567475}
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7'])
    gs.grow_new_iv_ages = (1,4)
    in_progres=[]
    in_progres.append(get_line_between_points(c, (137, 99, 60), (160, 213, 161), -2))
    in_progres.append(get_line_between_points(c, (361, 326, 96), (300, 273, 159), -3))
    in_progres.append(get_line_between_points(c, (110, 392, 108), (180, 301, 199), -4))
    gs.in_progres=in_progres

def init_xp41_dr35():
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\smallSAM\\checked\\xp41_dr35_model2.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\init_xp41_dr35"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': 0.03888888888888889, 'a2': 0.022367205215419503, 'c': 46.22342687074827, 'tan': 3.8809837018142304}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8'],
                            is_new_grown=[(False,),(False,),(True,False,),(True,False,),(True,True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (472, 94, 35),(346, 244, 94), -2))
    in_progres.append(get_line_between_points(c, (594, 283, 23),(447, 221, 73), -2))
    in_progres[0].connected_secondary.append(in_progres[1])
    in_progres[1].active=False
    in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((427, 210, 75))))

    
    in_progres.append(get_line_between_points(c, (62, 48, 30), (170, 209, 87), -3))
    in_progres.append(get_line_between_points(c, (47, 262, 40),(118.285, 212.47000000000003, 63.0), -3))

    
    in_progres[2].connected_secondary.append(in_progres[3])
    in_progres[3].active=False
    in_progres[3].connection=in_progres[2].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((127, 205, 63))))

    #in_progres.append(get_line_between_points(c, (283, 509, 47),(286, 451, 85), -4))
    
    for X in [[(419.02, 436.83, 63.85), (376.9, 422.28, 73.78)],[(289.81, 471.41, 69.7), (306.37, 433.3, 78.89)]]:
        p=[c.canvas._stack_coord_to_micron(x) for x in X]

        line = YoungLine(1,p[0])
        line.line=p
        line.init_direction(gs.center)

        in_progres.append(line)
    
    # line = YoungLine(1,(306.37, 433.3, 78.89))
    # line.line=[(306.37, 433.3, 78.89), (289.81, 471.41, 69.7)]
    # line.init_direction(gs.center)

    # in_progres.append(line)
    

    gs.in_progres=in_progres

    line_1 = [x for x in gs.lines if x.age==0][0]
    gs.create_new_line(line_1,gs.YOUNG_DISTANCE,gs.initial_iv_ages[0],False)
    line_4 = [x for x in gs.lines if x.age==3][0]
    gs.create_new_line(line_4,gs.OLD_DISTANCE,gs.initial_iv_ages[1],True)

#in progress
#!!!!!!
def init_xp41_dr34():

    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\smallSAM\\global_simulation\\xp41_dr34_model.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\init_xp41_dr34"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': 0.062055555555555565, 'a2': 0.010908203003788667, 'c': 25.19928254273292, 'tan': 2.4202071879245963}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8',"9"],
                            is_new_grown=[(False,),(False,),(False,False,),(True,False,),(True,False,),(True,False),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)  
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (284, 356, 82),(355, 445, 156), -2))
    in_progres.append(get_line_between_points(c, (256, 484, 103),(308, 438, 127), -2))
    in_progres[0].connected_secondary.append(in_progres[1])
    in_progres[1].active=False
    in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((324, 428, 126))))

    
    in_progres.append(get_line_between_points(c, (621, 728, 88), (535, 650, 135), -3))
    in_progres.append(get_line_between_points(c, (467, 825, 78),(498, 662, 143), -3))

    
    gs.in_progres=in_progres


def init_xp41_dr37():

    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\smallSAM\\global_simulation\\xp41_dr37_model.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\init_xp41_dr37"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': 0.08659661022508933,
                    'a2': 0.08071768707482996,
                    'c': 37.01646538922777,
                    'tan': 2.7866034067095278}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7',"9"],
                            is_new_grown=[(False,),(False,),(False,),(False,),(False,True,),(False,True,),(False,True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)  
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (601, 218, 31),(596, 397, 127), -2))
    in_progres.append(get_line_between_points(c, (798, 390, 42),(632, 388, 91), -2))
    

    in_progres.append(get_line_between_points(c, (253, 355, 0), (301, 415, 80), -3))
    in_progres.append(get_line_between_points(c, (273, 503, 54),(307, 489, 74), -3))

    
    in_progres.append(get_line_between_points(c, (616, 704, 91), (585, 626, 113), -4))
    in_progres.append(get_line_between_points(c, (763, 685, 85),(651, 563, 104), -4))

    

    gs.in_progres=in_progres

def init_maximum_attraction(init_function):
    c = Context()
    init_function()
    c.data_context.global_simulation.do_maximum_attraction=True
    c.data_context.image_path += "_attraction_2"
    c.data_context.global_simulation.surface_points.side_influence=0.05
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass
    #c.data_context.global_simulation.surface_points.angle_error=15
def init_xp42_dr48():

    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\smallSAM\\global_simulation\\xp42_dr48_model -1 connected.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\init_xp42_dr48"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': -0.1620280756670409,
      'a2': -0.09734989921894684,
      'c': 42.35151322529684,
      'tan': 3.51050998209393}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8',"9"],
                            is_new_grown=[(False,),(False,),(False,False,),(True,False,),(True,False,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)  
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (636, 296, 93),(509, 481, 170), -2))
    in_progres.append(get_line_between_points(c, (458, 325, 146),(493.464, 428.196, 159.95600000000002), -2))
    in_progres[0].connected_secondary.append(in_progres[1])
    in_progres[1].active=False
    in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((513, 447, 161))))

    
    in_progres.append(get_line_between_points(c, (166, 562, 131), (266, 579, 149), -3))
    in_progres.append(get_line_between_points(c, (234, 699, 108),(285, 659, 125), -3))

    
    in_progres.append(get_line_between_points(c, (611, 781, 3),(582, 742, 41), -4))

    
    gs.in_progres=in_progres

def init_xp42_dr51():

    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\smallSAM\\global_simulation\\xp42_dr51_model.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\init_xp42_dr51"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': -0.1620280756670409,
      'a2': -0.09734989921894684,
      'c': 42.35151322529684,
      'tan': 3.51050998209393}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8'],
                            is_new_grown=[(False,),(False,),(False,False,),(True,False,),(True,False,),(True,False),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)  
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (433, 285, 144),(474, 404, 204), -3))
    in_progres.append(get_line_between_points(c, (605, 255, 147),(502, 375, 185), -3))
    in_progres[0].connected_secondary.append(in_progres[1])
    in_progres[1].active=False
    in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((480, 383, 195))))

    
    in_progres.append(get_line_between_points(c, (569, 740, 180), (476, 643, 202), -2))
    in_progres.append(get_line_between_points(c, (508, 911, 89),(438, 668, 180), -2))

    

    
    gs.in_progres=in_progres

def init_streamplot_gs(medium_n5_time=None,suffix="",coeffs={'pos_range': 20.6139871413999,
        'age_coef': 0.03963136053954481,
        'inhibition_coef': 2.7873971411850995,
        'attraction_coef': 0.0,
        'neg_range': 59.22591211160096,
        'straight_coef': 3.0624121912173043,
        'inertia': 0.5,
        #'inertia': 1.,
        'peak_coef': 0.99,
        'age_cut_off_coef': 0.8961433833351786}):
    c = Context()
    c.data_context.global_simulation=GlobalSimulation(c)
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\stream_plot\\exp9_Dr5v2_1_stream plots prunedzzz2.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = f""
    
    gs.coeffs=copy.deepcopy(coeffs)

    gs.cone_coeffs={'a1': 0.08454166666666667, 'a2': -0.0762300171753204, 'c': -48.23415972102842, 'tan': 3.521112408134457}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['1,','2','7','15'],
                            is_new_grown=[(False,),(False,),(False,),(False,False,)])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    
    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-6s"),c.data_context.get_point_from_label("-6"), -6))
    gs.in_progres=in_progres

def init_small_idealized(medium_n5_time=None,suffix="",coeffs={'pos_range': 20.6139871413999,
        'age_coef': 0.03963136053954481,
        'inhibition_coef': 2.7873971411850995,
        'attraction_coef': 0.0,
        'neg_range': 59.22591211160096,
        'straight_coef': 3.0624121912173043,
        'inertia': 0.5,
        #'inertia': 1.,
        'peak_coef': 0.99,
        'age_cut_off_coef': 0.8961433833351786}):
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\global_simulation\\small_ideal_round.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = f".\\images\\ideal_small_full_{suffix}"

    
    if medium_n5_time:
        c.data_context.image_path = f".\\images\\ideal_small_full_{medium_n5_time}_40{'_' if suffix else ''}{suffix}"
        global n5_ages
        n5_ages=[1,medium_n5_time/12,40/12]
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    
    gs.coeffs=copy.deepcopy(coeffs)

    gs.cone_coeffs={'a1': 0.0, 'a2': 0.0, 'c': 0.0, 'tan': 3.11}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    gs.growth_coeff_development = growth_coeffs_full()
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8','9','10'],
                            is_new_grown=[(False,),(False,),(False,False,),(True,False,),(True,False,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    
    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-2s"),c.data_context.get_point_from_label("-2"), -2))
    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-2bs"), c.data_context.get_point_from_label("-2be"), -2))
    in_progres[0].connected_secondary.append(in_progres[1])
    in_progres[1].active=False
    in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((491.0, 296.0, 1.591))))

    
    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-3s"), c.data_context.get_point_from_label("-3"), -3))
    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-3bs"), c.data_context.get_point_from_label("-3be"), -3))
    in_progres[2].connected_secondary.append(in_progres[3])
    in_progres[3].active=False
    in_progres[3].connection=1

    in_progres.append(get_line_between_points(c, c.data_context.get_point_from_label("-4s"), c.data_context.get_point_from_label("-4"), -4))
    gs.in_progres=in_progres

    
#test for only two elements
def init_single_test(medium_n5_time=None,suffix="",coeffs={'pos_range': 20.6139871413999,
        'age_coef': 0.03963136053954481,
        'inhibition_coef': 2.7873971411850995,
        'attraction_coef': 0.0,
        'neg_range': 59.22591211160096,
        'straight_coef': 3.0624121912173043,
        'inertia': 0.5,
        #'inertia': 1.,
        'peak_coef': 0.99,
        'age_cut_off_coef': 0.8961433833351786}):
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\global_simulation\\small_ideal_round.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = f".\\images\\ideal_test_{suffix}"
    
    if medium_n5_time:
        c.data_context.image_path = f".\\images\\ideal_test_{medium_n5_time}_40"
        global n5_ages
        n5_ages=[1,medium_n5_time/12,40/12]
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    
    gs.coeffs=copy.deepcopy(coeffs)

    gs.cone_coeffs={'a1': 0.0, 'a2': 0.0, 'c': 0.0, 'tan': 3.11}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    gs.growth_coeff_development = growth_coeffs_full()
    for p in [c.data_context.get_point_from_label(x) for x in ['-1','0']]:
        center=c.data_context.get_point_from_label("c")
        t01=10
        t02=1
        t1=t01/(t01+t02)
        t2=t02/(t01+t02)
        c.data_context.move_point(p,tuple((np.array(p)+t2*(np.array(center)-np.array(p)))))
    
    c.data_context.init_global_simulation(labels=['-1','0'],
                            is_new_grown=[(False,),(False,),(False,False,),(True,False,),(True,False,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    #gs.is_young_left=True
    gs.surface_points.skip_primordiums=1.8
    c.data_context.global_simulation.growth_coeff_development = growth_coeffs_no_change(1)
    gs.growth_coeff_development.growth_rate=[9.5,8.5]
    gs.growth_coeff_development.growth_coeffs_list=[{'A': 0.7614290317628093, 'B': 10.062656265881261, 'C': 30.85128301663676}]
    c.data_context.global_simulation.growth_coeff_development.n5_ages=[3,3]
    c.data_context.global_simulation.growth_coeff_development.n8_ages=[1]
    c.data_context.global_simulation.growth_coeff_development.surface_frequency=[16,12]
    c.data_context.global_simulation.growth_coeff_development.distances_to_center=[40.0]
    gs.surface_points._extra_offset_step=0
    gs.surface_points.extra_offset=-0.8




def init_small_only_growth():
    c = Context()
    init_small_idealized()
    c.data_context.image_path = f".\\images\\ideal_small__higher_growth_extension_time"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass
    gc = growth_coeffs_no_change(1)
    gc.coeffs_development_times=grow_times
    import copy
    gc.growth_coeffs_list=copy.deepcopy(grow_coeffs)
    # for g  in gc.coeffs_list:
    #     g["b"]+=1
    # gc.n5_ages=[2]
    # gc.n8_ages=[2]

    gc.growth_rate=[3]
    c.data_context.global_simulation.growth_coeff_development=gc

    #jeszcze może sprawdzic dla niskich wartości extension rate


def init_small_growth_frequency():
    c = Context()
    init_small_idealized()
    c.data_context.image_path = f".\\images\\ideal_small_higher_growth_frequency_extension_time"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass
    gc = growth_coeffs_no_change(1)
    gc.coeffs_development_times=grow_times
    gc.growth_coeffs_list=grow_coeffs
    gc.growth_coeffs_list=copy.deepcopy(grow_coeffs)
    # for g  in gc.coeffs_list:
    #     g["b"]+=1
    # gc.n5_ages=[2]
    # gc.n8_ages=[2]

    gc.growth_rate=[3]
    gc.surface_frequency=surface_frequency
    c.data_context.global_simulation.growth_coeff_development=gc


def init_small_no_growth(**kwargs):
    print("XXXXXX")
    c = Context()
    init_small_idealized(**kwargs)
    print("XXXXXX")
    suffix=kwargs.get("suffix","")
    #c.data_context.image_path = f".\\images\\test_5\\no_growth_higher_growth_rate"
    c.data_context.image_path = f".\\images\\no_growth{suffix}"
    import os
    try:
        print("1XXXXXX",c.data_context.image_path)
        os.mkdir(c.data_context.image_path)
    except:
        pass
    # for g  in gc.coeffs_list:
    #     g["b"]+=1
    # gc.n5_ages=[2]
    # gc.n8_ages=[2]
    

    c.data_context.global_simulation.growth_coeff_development = growth_coeffs_no_change(1)
    # c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=[grow_coeffs[0]]
    # c.data_context.global_simulation.growth_coeff_development.surface_frequency=[surface_frequency[0]]
    # c.data_context.global_simulation.growth_coeff_development.growth_rate=[growth_rate[0]]

def init_medium_no_growth(**kwargs):
    c = Context()
    #init_medium_idealized(**kwargs)
    init_medium_idealized()
    suffix=kwargs.get("suffix","")
    #c.data_context.image_path = f".\\images\\test_5\\no_growth_higher_growth_rate"
    c.data_context.image_path = f".\\images\\no_growth_medium{suffix}"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass
    # for g  in gc.coeffs_list:
    #     g["b"]+=1
    # gc.n5_ages=[2]
    # gc.n8_ages=[2]
    

    c.data_context.global_simulation.growth_coeff_development = growth_coeffs_no_change(2)
    # c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=[grow_coeffs[0]]
    # c.data_context.global_simulation.growth_coeff_development.surface_frequency=[surface_frequency[0]]
    # c.data_context.global_simulation.growth_coeff_development.growth_rate=[growth_rate[0]]
    
def init_small_no_growth_frequency():
    c = Context()
    init_small_idealized()
    #c.data_context.image_path = f".\\images\\test_5\\no_growth_higher_growth_rate_frequency"
    c.data_context.image_path = f".\\images\\test_6\\no_growth_change_higher_growth_rate_frequency"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass
    # for g  in gc.coeffs_list:
    #     g["b"]+=1
    # gc.n5_ages=[2]
    # gc.n8_ages=[2]

    c.data_context.global_simulation.growth_coeff_development = growth_coeffs_full()
    c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=[grow_coeffs[0]]
    c.data_context.global_simulation.growth_coeff_development.growth_rate=[5]



def init_lucas(medium_n5_time=None):
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\lucas\\lucas_skeleton.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\lucas reconstruction original 2"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.coeffs={'pos_range': 20.6139871413999,
        'age_coef': 0.03963136053954481,
        'inhibition_coef': 3.7873971411850995,
        'attraction_coef': 0.0,
        'neg_range': 69.22591211160096,
        'straight_coef': 3.0624121912173043,
        'inertia': 0.5,
        'peak_coef': 0.99,
        'age_cut_off_coef': 0.8961433833351786}
    gs.cone_coeffs={'a1': 0.0, 'a2': 0.0, 'c': 0.0, 'tan': 3.11}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    gs.growth_coeff_development = growth_coeffs_no_change(1)
    gs.growth_coeff_development.distances_to_center=[55]
    c.data_context.init_global_simulation(labels=['0', '1', '2', '3', '4', '5', '6', '7','8','9','10'],
                            is_new_grown=[(False,),(False,),(True,False,),(True,False,),(True,True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),])
    gs.grow_new_iv_ages = (0,3)
    #???

    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  True
    #gs.max_adulted_age=7.0
    gs.max_adulted_age=7.0

    in_progres: List[YoungLine]=[]

    in_progres.append(get_line_between_points(c, (401.0, 1018.0, 1.1754),(435.0, 766.0, 1.4204), -1))
    in_progres.append(get_line_between_points(c, (261.0, 802.0, 1.2537), (346.0, 815.0, 1.3212),-1))

    in_progres.append(get_line_between_points(c, (784.0, 758.0, 1.1976),(606.0, 764.0, 1.3672), -2))
    in_progres.append(get_line_between_points(c, (639.0, 914.0, 1.2416),(630.0, 866.0, 1.2848), -2))

    in_progres.append(get_line_between_points(c, (485.0, 436.0, 1.2292),(538.0, 563.0, 1.3477), -3))

    gs.in_progres=in_progres

    gs.surface_points.angle=gs.surface_points.angle
    gs.surface_points._extra_offset_max=0
    gs.surface_points._extra_offset=0
    gs.surface_points.time_passed=1
    gs.surface_points.skip_primordiums=1
    gs.surface_points.current_angle=(gs.surface_points.current_angle+gs.surface_points.angle+360)%360
    #print(gs.surface_points.angle,gs.surface_points.offset)
def init_medium_idealized():
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\global_simulation\\medium_ideal\\2.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = ".\\images\\ideal_medium2"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': 0.0, 'a2': 0.0, 'c': 0.0, 'tan': 3.47}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    #gs.growth_coeff_development = growth_coeffs_no_change(2)
    gs.growth_coeff_development = growth_coeffs_full()
    #gs.growth_coeff_development = growth_coeffs_start_at(2)
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8','9','10','11','12','13'],
                            is_new_grown=[(False,),(False,),(False,),(False,),(False,True,),(False,True,),(False,True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),(True,),],
                            simulation_time=20)
    gs.grow_new_iv_ages = (4,2)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  True

    in_progres: List[YoungLine]=[]

    for i in range(-7,-1):
        if i == -6:
            continue
        a,b = get_start_end(str(i))
        print(i,a,b)
        in_progres.append(get_line_between_points(c, b,a, i))

    # in_progres.append(get_line_between_points(c, (220.0, 502.0, 1.28),(292.0, 410.0, 1.38), -3))
    # in_progres.append(get_line_between_points(c, (121.0, 333.0, 1.2), (240.41, 410.26, 1.33), -3))
    # in_progres[0].connected_secondary.append(in_progres[1])
    # in_progres[1].active=False
    # in_progres[1].connection=in_progres[0].line.index(_project_to_cone(c.canvas._stack_coord_to_micron((257.0, 421.0, 1.35))))

    
    # in_progres.append(get_line_between_points(c, (488.0, 198.0, 1.28), (474.0, 327.0, 1.4), -2))
    # in_progres.append(get_line_between_points(c, (670.0, 257.0, 1.2),(504.42, 293.08, 1.36), -2))
    # in_progres[2].connected_secondary.append(in_progres[3])
    # in_progres[3].active=False
    # in_progres[3].connection=1


    # in_progres.append(get_line_between_points(c, (568.0, 610.0, 1.24),(534.0, 532.0, 1.32), -4))

    gs.in_progres=in_progres
    #gs.coeffs['inhibition_coef']+=1
    #gs.coeffs['neg_range']+=3


def init_bijugate_idealized(suffix="",coeffs={'pos_range': 20.6139871413999,
        'age_coef': 0.03963136053954481,
        'inhibition_coef': 3.2873971411850995,
        'attraction_coef': 0.0,
        'neg_range': 79.22591211160096,
        'straight_coef': 3.0624121912173043,
        'inertia': 0.5,
        #'inertia': 1.,
        'peak_coef': 0.99,
        'age_cut_off_coef': 0.8961433833351786},
        angle=69.01):
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\bijugate\\skeleton labeled with extending updated labels.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")
    c.data_context.image_path = f".\\images\\ideal_bijugate{suffix}"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': 0.0, 'a2': 0.0, 'c': 0.0, 'tan': 3.47}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    time_mul = 2
    #gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs[:1],meristem_distances[:1],[],n5_ages[:1],n8_ages[:1],surface_frequency[:1])
    gs.growth_coeff_development = growth_coeffs_no_change(2)
    gs.growth_coeff_development.n8_ages=[1.75*time_mul]
    gs.growth_coeff_development.n5_ages=[1*time_mul]
    gs.growth_coeff_development.growth_rate=[6]
    gs.growth_coeff_development.surface_frequency=[12*time_mul]
    c.data_context.init_global_simulation(labels=['0', '1', '2', '3', '4', '5', '6',
                                                  '0b', '1b', '2b', '3b', '4b', '5b', '6b',],
                            is_new_grown=[(False,),(False,),(True,True,),(True,True,),(True,True,),(True,),(True,),
                                          (False,),(False,),(True,True,),(True,True,),(True,True,),(True,),(True,),])
    
    for line in gs.lines:
        line.matured_age=(line.matured_age*1)
    gs.coeffs=copy.copy(coeffs)
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False
    gs.max_adulted_age=4*time_mul

    in_progres: List[YoungLine]=[]
    extra_offset=-1.
    gs.surface_points=DoubleSurfacePoint(gs,1.0+extra_offset,-1,-angle,349.1722-angle,extra_offset=extra_offset)
    gs.surface_points.extra_offset_max=extra_offset
    gs.surface_points._extra_offset_step=0
    #gs.surface_points=DoubleSurfacePoint(gs,0,0,-75,349.1722-75,-3)
    in_progres.append(get_line_between_points(c, (780.0, 687.0, 1.2993),(708.0, 567.0, 1.3901), -1))
    in_progres.append(get_line_between_points(c, (340.0, 390.0, 1.2946), (411.0, 505.0, 1.3797), -1))
    #gs.coeffs['inhibition_coef']+=1
    #gs.coeffs['neg_range']+=3




    #in_progres.append(get_line_between_points(c, (283, 509, 47),(286, 451, 85), -4))
    
    # for X in [[(419.02, 436.83, 63.85), (376.9, 422.28, 73.78)],[(289.81, 471.41, 69.7), (306.37, 433.3, 78.89)]]:
    #     p=[c.canvas._stack_coord_to_micron(x) for x in X]

    #     line = YoungLine(1,p[0])
    #     line.line=p
    #     line.init_direction(gs.center)

    #     in_progres.append(line)
    
    # line = YoungLine(1,(306.37, 433.3, 78.89))
    # line.line=[(306.37, 433.3, 78.89), (289.81, 471.41, 69.7)]
    # line.init_direction(gs.center)

    # in_progres.append(line)
    

    gs.in_progres=in_progres

    # line_1 = [x for x in gs.lines if x.age==0][0]
    # gs.create_new_line(line_1,gs.YOUNG_DISTANCE,gs.initial_iv_ages[0],False)
    # line_4 = [x for x in gs.lines if x.age==3][0]
    # gs.create_new_line(line_4,gs.OLD_DISTANCE,gs.initial_iv_ages[1],True)



def init_xp5_dr5():
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open(".\\skeletons\\global_simulation\\xp5_dr5.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")

    c.data_context.image_path = ".\\images\\big_delay_surface"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': -0.11474333333333332, 'a2': -0.1509559861271626, 'c': -1.7163421439242152, 'tan': 3.6591668218746016}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs,meristem_distances,grow_times,n5_ages,n8_ages,surface_frequency)
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8','9','10','11','12'],
                                            simulation_time=20,
                                            is_new_grown = [(False,),(False,), (False,), (False,False), (False,False), (True,False), (True,True), (False,True), (True,),(True,),(True,),(True,),(True,),(True,)])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    in_progres.append(get_line_between_points(c, (912, 588, 84), (787, 606, 138), -8*10/13))
    in_progres.append(get_line_between_points(c, (830, 728, 74), (765, 669, 122), -8*10/13))

    in_progres.append(get_line_between_points(c, (175, 256, 224), (338, 398, 260), -7*10/12))
    in_progres.append(get_line_between_points(c, (418, 289, 212), (414, 392, 216), -7*10/12))

    in_progres.append(get_line_between_points(c, (404, 1004, 92), (487, 705, 210), -6*10/12))
    in_progres.append(get_line_between_points(c, (515, 804, 151), (505, 798, 162), -6*10/12))

    in_progres.append(get_line_between_points(c, (947, 329, 71),(691, 484, 191), -5*10/12))
    in_progres.append(get_line_between_points(c, (802, 183, 115), (679, 437, 187), -5*10/12))

    in_progres.append(get_line_between_points(c, (282, 587, 215),(422, 550, 282), -4*10/12))
    in_progres.append(get_line_between_points(c, (746, 867, 59), (615, 674, 227), -3*10/12))


    in_progres.append(get_line_between_points(c, (595, 117, 163), (549, 443, 240), -2*10/12))


    #in_progres.append(get_line_between_points(c, (263, 629, 217), (288, 621, 225), -9*10/12))

    gs.in_progres=in_progres
    
# def init_xp14_21():
#     c = Context()
#     gs : GlobalSimulation = c.data_context.global_simulation 
#     with open(".\\skeletons\\global_simulation\\xp14_21.ske","rb") as file:
#         c.data_context.load_data(pickle.load(file))
#     c.data_context.values['center'] = c.data_context.get_point_from_label("c")
#     gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
#     gs.cone_coeffs={'a1': -0.025881440770232956, 'a2': 0.11337176409037654, 'c': 54.200859758355904, 'tan': 4.617482857751077}
#     c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7','8','9','10','11','12','13'],
#                                             simulation_time=20,
#                                             is_new_grown = [(False,),(True,)])
def init_stream_plot():
    c = Context()
    gs : GlobalSimulation = c.data_context.global_simulation 
    with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\PISANIE\\ilustracje\\stream plot\\xp48_dr90_model_for -3 and -4  2.ske","rb") as file:
        c.data_context.load_data(pickle.load(file))
    c.data_context.values['center'] = c.data_context.get_point_from_label("c")

    c.data_context.image_path = ".\\images\\big_delay_surface"
    import os
    try:
        os.mkdir(c.data_context.image_path)
    except:
        pass

    gs.cone_coeffs={'a1': -0.11474333333333332, 'a2': -0.1509559861271626, 'c': -1.7163421439242152, 'tan': 3.6591668218746016}
    gs.center=c.canvas._stack_coord_to_micron(c.data_context.values['center'])
    gs.growth_coeff_development = GrowthCoeffDevelopment(grow_coeffs,meristem_distances,grow_times,n5_ages,n8_ages,surface_frequency)
    c.data_context.init_global_simulation(labels=['-4','-3','-2','-1','0', '1', '2', '3', '4', '5', '6', '7','8','9','10','11','12'],
                                            simulation_time=20,
                                            is_new_grown = [(False,),(False,), (False,), (False,),(False,), (False,), (False,False), (False,False), (True,False), (True,True), (False,True), (True,),(True,),(True,),(True,),(True,),(True,)])
    gs.grow_new_iv_ages = (0,3)
    #???
    gs.initial_iv_ages = (-4,-4)
    gs.initial_iv_ages = (0,0)
    gs.is_young_left =  False

    in_progres: List[YoungLine]=[]
    # in_progres.append(get_line_between_points(c, (912, 588, 84), (787, 606, 138), -8*10/13))
    # in_progres.append(get_line_between_points(c, (830, 728, 74), (765, 669, 122), -8*10/13))

    # in_progres.append(get_line_between_points(c, (175, 256, 224), (338, 398, 260), -7*10/12))
    # in_progres.append(get_line_between_points(c, (418, 289, 212), (414, 392, 216), -7*10/12))

    # in_progres.append(get_line_between_points(c, (404, 1004, 92), (487, 705, 210), -6*10/12))
    # in_progres.append(get_line_between_points(c, (515, 804, 151), (505, 798, 162), -6*10/12))

    # in_progres.append(get_line_between_points(c, (947, 329, 71),(691, 484, 191), -5*10/12))
    # in_progres.append(get_line_between_points(c, (802, 183, 115), (679, 437, 187), -5*10/12))

    # in_progres.append(get_line_between_points(c, (282, 587, 215),(422, 550, 282), -4*10/12))
    # in_progres.append(get_line_between_points(c, (746, 867, 59), (615, 674, 227), -3*10/12))


    # in_progres.append(get_line_between_points(c, (595, 117, 163), (549, 443, 240), -2*10/12))


    #in_progres.append(get_line_between_points(c, (263, 629, 217), (288, 621, 225), -9*10/12))

    gs.in_progres=in_progres