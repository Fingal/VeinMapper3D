from itertools import product
import matplotlib.pyplot as plt 
from types import MethodType
import numpy as np
import generate_line
from simulation_controller import SimulationController
from skeleton import *
from global_simulation import YoungLine
from skeleton import *
from random import random
from calculation_skeleton_features import *

c : Context=locals()['CONTEXT']

#labels=[('-3','10')]
labels = [(str(i),str(i+8),str(i+13)) for i in [-5,-4,-3,-2,-1]]
[('-5', '3', '8'), ('-4', '4', '9'), ('-3', '5', '10'), ('-2', '6', '11'), ('-1', '7', '4')]

#c.skeleton_distances.load_skeleton('C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\xp48_dr5v2_90_ske_global_simulation3.ske')


#x,d_value=self._generate_lines(labels)

# best for 3d {'pos_range': 177.29465325468755, 'strongest': 1.4329423777785555, 'age_coef': 0.14071004226562506, 'inhibition_coef': 0.5302494097246091, 'attraction_coef': 3.209628037917293, 'neg_range': 9.70087200000543, 'straight_coef': 14.589737840609605, 'inertia': 17.738701653625107}
# {'pos_range': 177.29465325468755, 'strongest': 1.4329423777785555, 'age_coef': 0.14071004226562506, 'inhibition_coef': 0.5302494097246091, 'attraction_coef': 3.209628037917293, 'neg_range': 9.70087200000543, 'straight_coef': 14.589737840609605, 'inertia': 17.738701653625107}
# {'pos_range': 188.9555488152633, 'strongest': 0.30000000000000004, 'age_coef': 0.12950000000000028, 'inhibition_coef': 0.10000000000000003, 'attraction_coef': 0.9999999999999999, 'neg_range': 10, 'straight_coef': 1.0, 'inertia': 18.19125}
# {'pos_range': 182.90436758160953, 'strongest': 1.0999999999999994, 'age_coef': 0.1, 'inhibition_coef': 0.10000000000000003, 'attraction_coef': 3.3205217667119324, 'neg_range': 12.11073967000479, 'straight_coef': 9.427572831796875, 'inertia': 21.379241976453912}
#c.data_context._generate_lines(labels,**{'pos_range': 177.29465325468755, 'strongest': 1.4329423777785555, 'age_coef': 0.14071004226562506, 'inhibition_coef': 0.5302494097246091, 'attraction_coef': 3.209628037917293, 'neg_range': 9.70087200000543, 'straight_coef': 14.589737840609605, 'inertia': 17.738701653625107})
#{'pos_range': 102.73541336939884, 'strongest': 0.4, 'age_coef': 0.10000000000000003, 'inhibition_coef': 0.10000000000000003, 'attraction_coef': 0.68503746875, 'neg_range': 16.03796282082979, 'straight_coef': 0.39999999999999997, 'inertia': 15.3964125}


# check {'pos_range': 132.01320020389173, 'strongest': 0.6, 'age_coef': 0.07, 'inhibition_coef': 0.010000000000000002, 'attraction_coef': 0.3850374687500001, 'neg_range': 5.03796282082979, 'straight_coef': 1.6, 'inertia': 27.64974478617119}
# {'pos_range': 125.41254019369714, 'strongest': 0.4, 'age_coef': 0.05, 'inhibition_coef': 0.010000000000000002, 'attraction_coef': 0.98503746875, 'neg_range': 5.03796282082979, 'straight_coef': 1.3, 'inertia': 38.90596757495893}
center = (550.2857142857143, 575.7142857142857, 266.57142857142856)

def calculate(values,key,strength):
    _v=dict(values)
    _v[key]=(_v[key]*(1+strength/100))
    _,d=c.data_context._generate_lines(labels,**_v)
    return (d,key,strength)
    
#v={'pos_range': 137.33109139515335, 'strongest': 0.7999999999999999, 'age_coef': 0.08, 'inhibition_coef': 0.010000000000000002, 'attraction_coef': 0.78503746875, 'neg_range': 15.987867913777716, 'straight_coef': 1.0, 'inertia': 23.884889136094323}
#v= {'pos_range': 132.01320020389173, 'strongest': 0.6, 'age_coef': 0.07, 'inhibition_coef': 0.010000000000000002, 'attraction_coef': 0.3850374687500001, 'neg_range': 5.03796282082979, 'straight_coef': 1.6, 'inertia': 27.64974478617119}
# for 84 {'pos_range': 78.11393745799018, 'age_coef': 0.07, 'inhibition_coef': 0.019999999999999993, 'attraction_coef': 2.1, 'neg_range': 20.10891679794909, 'straight_coef': 1.3, 'inertia': 0.8848891360943227, 'peak_coef': 0.448596741796875}
# best for all {'pos_range': 60.47000856815009, 'age_coef': 0.19548149999999997, 'inhibition_coef': 0.14, 'attraction_coef': 6.051750762134889, 'neg_range': 18.008072999954862, 'straight_coef': 3.1000000000000014, 'inertia': 8.884889136094323, 'peak_coef': 0.7899999999999998}
# best for no Y {'pos_range': 63.17643828600603, 'age_coef': 0.17548149999999996, 'inhibition_coef': 0.019999999999999993, 'attraction_coef': 5.4344306731692456, 'neg_range': 19.007285882459264, 'straight_coef': 2.8000000000000016, 'inertia': 3.3848891360943227, 'peak_coef': 0.8399999999999999}
# best for old no Y {'pos_range': 48.603998494681036, 'age_coef': 0.2983057350553191, 'inhibition_coef': 1.8974826990902753, 'attraction_coef': 1.7619492609087894, 'neg_range': 20.749256907201456, 'straight_coef': 2.200000000000002, 'inertia': 2.3848891360943227, 'peak_coef': 1.3679999999999999}
#best for old selected {'pos_range': 87.28579816664298, 'age_coef': 0.2798657353773545, 'inhibition_coef': 0.32620979385574644, 'attraction_coef': 2.7563011926944743, 'neg_range': 11.711794061841385, 'straight_coef': 1.6000000000000019, 'inertia': 1.3848891360943227, 'peak_coef': 0.48510826869913243}
#best for old no Y selected {'pos_range': 54.46376405141337, 'age_coef': 0.1750504891345548, 'inhibition_coef': 0.9397275102984234, 'attraction_coef': 1.1008875744661841, 'neg_range': 27.686395469466905, 'straight_coef': 1.0000000000000018, 'inertia': 3.3848891360943227, 'peak_coef': 0.93814875}
# best for all values no Y {'pos_range': 60.04629986668324, 'age_coef': 0.13505048913455475, 'inhibition_coef': 1.0878520591092122, 'attraction_coef': 1.000887574466184, 'neg_range': 27.686395469466905, 'straight_coef': 1.3000000000000018, 'inertia': 2.8848891360943227, 'peak_coef': 0.99}
# best for all values {'pos_range': 66.20104560301827, 'age_coef': 0.12505048913455474, 'inhibition_coef': 0.931786483346064, 'attraction_coef': 1.3008875744661843, 'neg_range': 23.737623315634185, 'straight_coef': 1.3000000000000018, 'inertia': 1.8848891360943227, 'peak_coef': 0.987525}

#c.data_context.test_coefs={'pos_range': 57.28579816664298, 'age_coef': 0.2798657353773545, 'inhibition_coef': 0.32620979385574644, 'attraction_coef': 2.7563011926944743, 'neg_range': 11.711794061841385, 'straight_coef': 1.6000000000000019, 'inertia': 1.3848891360943227, 'peak_coef': 0.98510826869913243}
c.data_context.test_coefs={'pos_range': 71.71620795045503, 'age_coef': 0.07692824879311926, 'inhibition_coef': 3.399847997789268, 'attraction_coef': 3.837766152702213, 'neg_range': 18.91505374438373, 'straight_coef': 2.200000000000002, 'inertia': 5.384889136094323, 'peak_coef': 0.7875}
# best for x90 only {'pos_range': 71.71620795045503, 'age_coef': 0.07692824879311926, 'inhibition_coef': 3.399847997789268, 'attraction_coef': 3.645877845067102, 'neg_range': 18.91505374438373, 'straight_coef': 2.200000000000002, 'inertia': 5.384889136094323, 'peak_coef': 0.7875}
v={'pos_range': 107.89361255813142, 'age_coef': 0.14, 'inhibition_coef': 6.355953160070807, 'attraction_coef': 2.293621646025508, 'neg_range': 32.98578691659749, 'straight_coef': 2.3624899123526997, 'inertia': 2.7, 'peak_coef': 0.4152491906221574}

c.data_context.test_coefs={'pos_range': 67.89361255813142, 'age_coef': 0.14, 'inhibition_coef': 6.355953160070807, 'attraction_coef': 2.293621646025508, 'neg_range': 32.98578691659749, 'straight_coef': 2.3624899123526997, 'inertia': 2.7, 'peak_coef': 0.8152491906221574,'age_cut_off_coef':4}

# poki co {'pos_range': 146.77568061102758, 'age_coef': 0.1, 'inhibition_coef': 4.312741134783502, 'attraction_coef': 3.3008875744661847, 'neg_range': 33.23442245890604, 'straight_coef': 3.2300000000000018, 'inertia': 2.3, 'peak_coef': 0.6259216192894532}
#test bez atrakcji 
c.data_context.test_coefs={'pos_range': 15.00910389371612, 'age_coef': 0.15, 'inhibition_coef': 6.0953542885783545, 'attraction_coef': 0.0664938587611197, 'neg_range': 33.048488571771294, 'straight_coef': 1.033288516025809, 'inertia': 6.04168514422266, 'peak_coef': 0.958992855719747, 'age_cut_off_coef': 6.974547012901643}


def draw(labels,v):
    c.data_context._generate_lines(labels,**v,draw=True)


def draw_strength(v,age=5):
    i=0
    strength=-generate_line.calculate_strength(i,age, v['pos_range'], v['neg_range'], v['inhibition_coef'], v['attraction_coef'],v['age_coef'],v['age_cut_off_coef'],v['peak_coef'])
    result=[strength]
    while i<250:
        strength=-generate_line.calculate_strength(i,age, v['pos_range'], v['neg_range'], v['inhibition_coef'], v['attraction_coef'],v['age_coef'],v['age_cut_off_coef'],v['peak_coef'])
        result.append(strength)
        i+=1
    fig, ax = plt.subplots()
    ax.plot(result)
    ax.grid(True, which='both')
    ax.set_aspect(20)
    ax.axhline(y=0, color='k')
    fig.savefig(f'C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\age_{age}.png', dpi = 300)


#for age in (5,6,7,10,11,12):
#    draw_strength(v,age=age)




def remove_line(label):
    points=c.data_context.get_points_from_label(label)
    c.canvas.skeleton_graph[points[0]].remove(points[1])
    c.canvas.skeleton_graph[points[1]].remove(points[0])


from shape_calculation import *

#convincing x=self._generate_lines(labels,strongest=5,attraction_coef=3,neg_range=30,straight_coef=5,inertia=30)


# from wx.py import frame

# class MyCurst(CrustFrame):
#     def __init__(self, startup_script=None, parent=None, id=-1, title='MyCrust',
#                  pos=wx.DefaultPosition, size=wx.DefaultSize,
#                  style=wx.DEFAULT_FRAME_STYLE,    
#                  rootObject=None, rootLabel=None, rootIsNamespace=True,
#                  locals=None, InterpClass=None,
#                  config=None, dataDir=None,
#                  *args, **kwds):
#         """Create CrustFrame instance."""
#         frame.Frame.__init__(self, parent, id, title, pos, size, style,
#                              shellName='PyCrust')
#         frame.ShellFrameMixin.__init__(self, config, dataDir)

#         if size == wx.DefaultSize:
#             self.SetSize((800, 600))

#         intro = 'PyCrust %s - The Flakiest Python Shell' % VERSION
#         self.SetStatusText(intro.replace('\n', ', '))
#         self.crust = Crust(parent=self, intro=intro,
#                            rootObject=rootObject,
#                            rootLabel=rootLabel,
#                            rootIsNamespace=rootIsNamespace,
#                            locals=locals,
#                            InterpClass=InterpClass,
#                            startupScript='startup.py',
#                            execStartupScript='startup.py',
#                            *args, **kwds)
#         self.shell = self.crust.shell

#         # Override the filling so that status messages go to the status bar.
#         self.crust.filling.tree.setStatusText = self.SetStatusText

#         # Override the shell so that status messages go to the status bar.
#         self.shell.setStatusText = self.SetStatusText

#         self.shell.SetFocus()
#         self.LoadSettings()


def remove_line_label(remove,dont_remove,remove_higher=False):
    if remove_higher:
        if dont_remove[2]>remove[2]:
            remove, dont_remove = dont_remove, remove
    points=[remove,dont_remove]
    line = find_path_between_nodes(c.canvas.skeleton_graph,*points)[0]
    line.remove(dont_remove)
    c.data_context.remove(line,'point')
    c.data_context.remove([remove],'label')
    c.data_context.add_label(dont_remove)


def get_line_between_marked():
    a=find_path_between_nodes(c.canvas.skeleton_graph, *c.skeleton_distances.get_points(), find_only_primary=True)
    if c.canvas.point_distance_L2(a[-1],c.data_context.values['center'])>c.canvas.point_distance_L2(a[0],c.data_context.values['center']):
        a=a[::-1]
    c.graphic_context.mark_selection_points(a[:-1])
    return a

from global_simulation import YoungLine,GlobalSimulation

color_step=c.skeleton_distances.simulation_controller.color_step
# for xp448_90
gs : GlobalSimulation = c.data_context.global_simulation
def init_dr90():
    global gs

    gs.cone_coeffs={'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}
    gs.growth_coeffs_young=gs.growth_coeffs_old

    np.random.seed(10)
    c.skeleton_distances.load_skeleton('C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\xp48_dr5v2_90_ske_global_simulation_trimmed.ske')
    gs = GlobalSimulation(c)
    c.data_context.global_simulation=gs
    sk=c.canvas.skeleton_graph
    in_progres=[]    
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
    in_progres=[]
    #gs.center = c.data_context.values['center']
    in_progres.append(YoungLine(-2,(0,0,0)))
    line2r=find_path_between_nodes(sk,(866, 274, 17),(625, 462, 178),find_only_primary=True)
    in_progres[0].line=line2r
    list(filter(lambda x: len(sk[x])==3,line2r))
    line2l=find_path_between_nodes(sk,(647, 318, 103),(655.588, 371.548, 124.64),find_only_primary=True)

    in_progres.append(YoungLine(-2,None))
    a=in_progres[1]
    a.line=line2l[:-1]
    a.connection=in_progres[0].line.index(line2l[-1])
    in_progres[0].connected_secondary.append(in_progres[1])
    line3=find_path_between_nodes(sk,(461, 995, 64),(513, 727, 187), find_only_primary=True)

    in_progres.append(YoungLine(-3,(1,)))
    in_progres[-1].line=line3
    line4=find_path_between_nodes(sk,(261, 232, 119),(400, 436, 181),find_only_primary=True)

    in_progres.append(YoungLine(-4,5))
    in_progres[-1].line=line4
    in_progres[1].active=False
    for p in in_progres:
        p.line=[c.canvas._stack_coord_to_micron(x)  for x in p.line]
    for p in in_progres:
        if p.active:
            p.init_direction(c.data_context.global_simulation.center)
    gs.in_progres=in_progres
    #gs.simulation(50)

    a5=[(46.69581384063926, -67.8164255773282, 146.3456433768487), (44.83165359497071, -66.17872378733327, 142.9077453613281), (42.95730590820312, -64.51865600663498, 139.41290283203122), (41.10945129394535, -62.86538140370615, 135.9216461181641), (39.262599945068416, -61.22858113221224, 132.4606628417969), (37.430801391601584, -59.607966141956695, 129.02648925781253), (35.62085723876955, -58.004532155829665, 125.62010192871101), (33.83338165283205, -56.41882322332924, 122.24252319335939), (32.07751846313478, -54.848036205404696, 118.88588714599608), (30.350378036499038, -53.293886716025156, 115.55459594726567), (28.66338920593263, -51.754569993502, 112.24306488037105), (27.011932373046918, -50.23245921547482, 108.95755004882814), (25.414165496826143, -48.72253109927209, 105.68466949462888), (23.8722171783447, -47.22262462552405, 102.41984558105466), (22.38863945007322, -45.73129919434348, 99.16018676757815), (20.965747833251932, -44.247582215686506, 95.90397644042973), (19.61597442626956, -42.76830188851072, 92.64369964599607), (18.347719192504844, -41.291711635749124, 89.37575531005857), (17.165630340576147, -39.81690759106496, 86.09893798828128), (16.063993453979446, -38.34673320059545, 82.82134246826173), (15.034232139587377, -36.883151806903996, 79.54904174804689), (14.072859764099103, -35.42548778671585, 76.28159332275389), (13.176010131835913, -33.97403100719825, 73.02067565917969), (12.339735031127928, -32.52969749098693, 69.76929473876957), (11.555845260620124, -31.09533373135893, 66.53491210937501), (10.820261001586887, -29.672114103973154, 63.320869445800795), (10.129300117492656, -28.261440112288472, 60.13092803955079), (9.477180480957028, -26.86654208067893, 56.97298431396481), (8.859771728515655, -25.48915227236007, 53.85138320922852), (8.273654937744084, -24.131136246120835, 50.7706871032715), (7.715934753417906, -22.794433703980776, 47.735561370849624)]
    a6=[(119.88348388671874, -46.795786995648825, -176.68028259277347), (118.98269653320314, -45.345242689024204, -172.9399871826172), (118.45529937744146, -43.97165314201611, -169.03244018554688), (118.14882659912105, -42.66630361019915, -165.0619354248047), (117.9508438110352, -41.41079328193287, -161.07092285156247), (117.77670288085932, -40.18502049343148, -157.0749053955078), (117.54650878906247, -38.966413404856155, -153.0825958251953), (117.25009918212895, -37.752199355632726, -149.0950927734375), (116.8044204711914, -36.517874762171814, -145.1277160644531), (116.14944458007808, -35.24527428331008, -141.19728088378906), (115.24507141113276, -33.92257831952644, -137.3238525390625), (114.07823181152344, -32.545862695948614, -133.52453613281247), (112.6059265136719, -31.10673576757124, -129.84538269042972), (110.87147521972656, -29.612869099284215, -126.27153778076172), (108.92395019531251, -28.078050579441598, -122.7995452880859), (106.79097747802732, -26.510799549995273, -119.43418121337889), (104.45123291015625, -24.909487237336624, -116.21722412109376), (101.88426971435551, -23.271505316367605, -113.18872070312501), (99.09355926513665, -21.59855865256848, -110.3682098388672), (96.08906555175786, -19.896107342462503, -107.7800750732422)]
    a7=[(-162.39385986328125, -56.18419645535336, 11.684492111206076), (-159.6805267333984, -54.646106860476266, 9.108146667480487), (-156.51702880859378, -52.99335098341787, 6.917696952819806), (-153.07963562011722, -51.287591376130834, 5.070940971374515), (-149.47915649414062, -49.5641977382722, 3.5167372226715363), (-145.78189086914062, -47.838857527189234, 2.1883413791656197), (-142.03179931640625, -46.120067271617245, 1.027633666992183), (-138.26084899902344, -44.41150099000534, -0.023792266845665353), (-134.49497985839844, -42.71508602660299, -1.0193428993225369)]

    a5 = [generate_line.project_to_cone(point,(14.785861525386506, 24.7665, -17.67426238150849),**{'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}) for point in a5]
    a6 = [generate_line.project_to_cone(point,(14.785861525386506, 24.7665, -17.67426238150849),**{'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}) for point in a6]
    a7 = [generate_line.project_to_cone(point,(14.785861525386506, 24.7665, -17.67426238150849),**{'a1': -0.03, 'a2': -0.11999999999999998, 'c': 4.600000000000001, 'tan': 2.4699999999999998}) for point in a7]

    line5 = YoungLine(-5,None)
    line5.line=a5
    line5.init_direction(None)
    line6 = YoungLine(-6,None)
    line6.line=a6
    line6.init_direction(None)
    line7 = YoungLine(-7,None)
    line7.line=a7
    line7.init_direction(None)

    gs.in_progres.append(line5)
    gs.in_progres.append(line6)
    gs.in_progres.append(line7)


# gs.create_new_line([x.skeleton for x in gs.lines if abs(x.age-5)<0.05][0],gs.OLD_DISTANCE,-8,True)
# gs.create_new_line([x.skeleton for x in gs.lines if abs(x.age-3)<0.05][0],gs.YOUNG_DISTANCE,-5,False)


#c.skeleton_distances.simulation_controller.color_lines()

def init_random(_number=3):
    global gs
    #np.random.seed(10)
    np.random.seed(int(1000*random()))
    #c.skeleton_distances.load_skeleton('C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\xp41_dr35_new_global_simulation_trimmed2.ske')
    gs = GlobalSimulation(c)
    c.data_context.global_simulation=gs
    gs = c.data_context.global_simulation
    sk=c.canvas.skeleton_graph    
    gs.cone_coeffs={'a1': 0.034999999999999865, 'a2': 0.040000000000000306, 'c': 22.800000000000228, 'tan': 2.4199999996166066}

    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
    gs.growth_coeffs=dict(A=0.114, B=2.842, C=50.03)
    # gs.growth_coeffs=gs.growth_coeffs_young
    # gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)

    gs.growth_rate=30
    gs.distance_treshold=(39,1)
    gs.connection_distance=5
    gs.young_attraction_distance=10
    gs.center=(3.369800998809006, 35.510999999999996, -58.9715174791578) 

    in_progres=[]
    def radian_points(number = 5):
        result = []
        for i in range(number):
            point = (np.sin(((np.pi*2)/number*i)),np.cos(((np.pi*2)/number*i)))
            result.append(point)
        return result
    
    def starting_points(number = 5,max_distance=80,start_distance=80):
        points=radian_points(number=number)
        result=[]
        for i,point in enumerate(points):
            distance = 50+(i+1)*max_distance/number
            new_point= (gs.center[0]+distance*point[0],gs.center[1],gs.center[2]+distance*point[1])
            result.append(new_point)
        return result
        

    points=starting_points(_number)
    #starting_point   
    for point in points:
        line = YoungLine(-6,generate_line.project_to_cone(point,gs.center,**gs.cone_coeffs))
        #line31.line=a31
        line.init_direction(gs.center)
        in_progres.append(line)
    gs.in_progres=in_progres
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.time_step=0.3
    # c.simulation_controller.screenshots_number=600






# for dr35
def init_dr35():
    global gs
    np.random.seed(10)
    c.skeleton_distances.load_skeleton('C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\xp41_dr35_new_global_simulation_trimmed2.ske')
    gs = GlobalSimulation(c)
    c.data_context.global_simulation=gs
    gs = c.data_context.global_simulation
    sk=c.canvas.skeleton_graph
    in_progres=[]    
    c.data_context.init_global_simulation(labels=['-1','0', '1', '2', '3', '4', '5','6'])

    a21=[(472, 94, 35), (468, 141, 52), (451, 174, 75), (441, 193, 75)]
    a22=[(594, 283, 23), (543, 275, 42), (523, 269, 42), (500, 256, 52), (471, 237, 64), (447, 221, 73), (427, 210, 75), (418.44899999999996, 211.006, 75.0), (404.379, 214.04399999999998, 78.577), (399, 216, 82), (372, 229, 91), (346, 244, 94)]
    a2_index=a22.index((427, 210, 75))

    a31=[(47, 262, 40), (92, 235, 63)]
    a32=[(62, 48, 30), (82, 101, 52), (94, 170, 59), (112, 192, 63), (127, 205, 63), (144, 211, 70), (158, 206, 78), (170, 209, 87)]
    a3_index = a32.index((127, 205, 63))

    lines = [a21,a22,a31,a32]
    lines = [[c.canvas._stack_coord_to_micron(point) for point in line] for line in lines]
    lines = [[generate_line.project_to_cone(point,(3.369800998809006, 35.510999999999996, -58.9715174791578),**{'a1': 0.034999999999999865, 'a2': 0.040000000000000306, 'c': 22.800000000000228, 'tan': 2.4199999996166066}) for point in line] for line in lines]
    a21,a22,a31,a32 = lines

    line31 = YoungLine(-6,None)
    line31.line=a31
    line32 = YoungLine(-6,None)
    line32.line=a32
    line32.connected_secondary.append(line31)
    line31.connection=a3_index
    line32.init_direction(None)
    line31.init_direction(None)

    line21 = YoungLine(-4,None)
    line21.line=a21
    line22 = YoungLine(-4,None)
    line22.line=a22
    line22.connected_secondary.append(line21)
    line21.connection=a2_index
    line22.init_direction(None)
    line21.init_direction(None)

    


    gs.in_progres=[line21,line22,line31,line32]
    for line in gs.lines:
        if line.age ==3:
            gs.create_new_line(line.skeleton,gs.OLD_DISTANCE,-8,True)
        if line.age ==4:
            gs.create_new_line(line.skeleton,gs.OLD_DISTANCE+10,-8,True)
        line.age=(line.age-1) *2+1
        if line.maturing_time>0:
            line.maturing_time=line.maturing_time*1.5
    #gs.points = [generate_line.project_to_cone(point,(3.369800998809006, 35.510999999999996, -58.9715174791578),**{'a1': 0.034999999999999865, 'a2': 0.040000000000000306, 'c': 22.800000000000228, 'tan': 2.4199999996166066}) for point in gs.points]



    gs.cone_coeffs={'a1': 0.034999999999999865, 'a2': 0.040000000000000306, 'c': 22.800000000000228, 'tan': 2.4199999996166066}
    gs.growth_rate=30
    gs.distance_treshold=(39,1)
    gs.connection_distance=5
    gs.young_attraction_distance=10

    # gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    # gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
    #gs.growth_coeffs_young=dict(A=0.2, B=6, C=62.6)
    #gs.meristem_grow_speed=0.4



# c.simulation_controller.screenshots_number=200
# c.simulation_controller.time_step=30

#growth tests

# gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
# gs.growth_coeffs=gs.growth_coeffs_young
# gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)

#init_dr35()
def slower_growth():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=.4
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)

def faster_growth():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=2
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)

def normal_growth():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)

def grow_A_more():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.4, B=6, C=62.6)

def grow_C_more():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=72.6)

def grow_B_more():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=8, C=62.6)

def grow_BC_more():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=8, C=72.6)

def grow_A_less():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.15, B=6, C=62.6)

def grow_C_less():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=6, C=55.6)

def grow_B_less():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=5, C=62.6)

def grow_BC_less():
    gs = c.data_context.global_simulation
    gs.meristem_grow_speed=1
    gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
    gs.growth_coeffs=gs.growth_coeffs_young
    gs.growth_coeffs_old=dict(A=0.2, B=4, C=55.6)


def grow_factory(function,name):
    def f():
        init_dr35()
        function()
        c.simulation_controller.color_step(time_step=30)
        c.simulation_controller.make_screenshot(f'growths\\{name}')
    return f

def random_factory(var={'growth':4,'A':0.4,'B':10,'C':20,'small_growth':30}):
    growth=random()*var['growth']
    growth_rate=25+random()*var['small_growth']
    y=dict(A=0.114, B=2.842, C=50.03)
    old={k:var[k]*random()+v for k,v in y.items()}
    def f():
        gs = c.data_context.global_simulation
        gs.meristem_grow_speed=growth
        gs.growth_rate=growth_rate
        gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
        gs.growth_coeffs=gs.growth_coeffs_young
        gs.growth_coeffs_old=old
    f.name=f"young rate {growth_rate:.2f} growth speed {growth:.2f}, A {old['A']:.2f}, B {old['B']:.2f}, C {old['C']:.2f}"
    return f
        



def growth_speed(speed):
    def f():
        gs = c.data_context.global_simulation
        gs.meristem_grow_speed=speed
        gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
        gs.growth_coeffs=gs.growth_coeffs_young
        gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
    f.name=f'speed {speed}'
    return f

#functions_to_test = [normal_growth,slower_growth,faster_growth,grow_A_more,grow_B_more,grow_C_more,grow_BC_more,grow_A_less,grow_B_less,grow_C_less,grow_BC_less]
#functions_to_test = [random_factory(var={'growth':6,'A':0.8,'B':15,'C':20,'small_growth':47}) for i in range(30)]
#functions_to_test = [growth_speed(speed/10) for speed in range(8,10)]
# functions_to_test = [normal_growth,slower_growth,faster_growth]+[growth_speed(speed/10) for speed in range(5,16)]

# def test_functions(functions=functions_to_test):
#     for function in functions:
#         grow_factory(function,function.name)()

# c.simulation_controller.functions_to_test=functions_to_test
# c.simulation_controller.init_function=init_dr35


# c.simulation_controller.time_step=1


cmin,cmax=-322.9538809035303, 228.11805279419087
v=stream_scale = (cmax-cmin)/700 

# stream plot map functions 
def show_map(a):
    _a=a/(max(np.max(a),-np.min(a))*2)
    _a=_a+0.5
    from matplotlib.colors import ListedColormap
    newcolors = np.array([[1,0.15,0.15,1]]*5+[[1,1,1,0]]*8+[[0.0,0.0,0.0,1.]]*5)
    return plt.imshow(_a,cmap=ListedColormap(newcolors),vmin=0,vmax=1)

a=0
def fill_array_f():
    global a
    for i in range(1):
        a=generate_line.fill_array((700,700),i,_line,**gs.coeffs)
        show_map(a)
        plt.savefig(f'age_{i}.jpg',dpi=400)

def fill_array_cone(coeffs=None,density=700):
    global gs
    if coeffs is None:
        coeffs=gs.coeffs
    #a=generate_line.fill_array_cone(density,gs,center=gs.center,**coeffs,**gs.cone_coeffs)
    c : Context = Context()
    center = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label("c"))
    adult_lines=[]
    young_lines=[]
    for i in range(15):
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            break

        adult_lines.append((c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))),i))
    for i in range(-6,0):
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            continue
        young_lines.append(c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))))

    a,_,_=generate_line.fill_array_cone_from_lines(density,adult_lines,young_lines,center=center,**coeffs,**gs.cone_coeffs)
    x=np.zeros((a.shape[1],a.shape[0],a.shape[2]))
    x[:,:,0]=a[:,:,0].T
    x[:,:,1]=a[:,:,1].T
    x[:,:,2]=a[:,:,2].T
    a=x
    return a

def stream_plot(coeffs=None,density=3):
    for line in gs.in_progres:
        xxs,yys =line.get_segments()
    a=fill_array_cone(coeffs)

    _stream_plot(a,density=density)
    return a
    plt.text(0, 0, pprint.pformat(coeffs), fontsize = 10, 
         bbox = dict(facecolor = 'white', alpha = 0.5))
    #plt.quiver(X,Y,a[:,:,2],a[:,:,0])
    #plt.show()
def _stream_plot(a,density=3):
    plt.close()
    plt.clf()
    plt.subplots(figsize=(10,10))
    X=np.linspace(0,a.shape[1],a.shape[1])
    Y=np.linspace(0,a.shape[0],a.shape[0])
    #lines skel
    for line in gs.lines:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([(x-cmin)/v for x in xx],[(y-cmin)/v for y in yy],c=color,linewidth=5.0)
    for line in gs.in_progres:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="red"
            plt.plot([(x-cmin)/v for x in xx],[(y-cmin)/v for y in yy],c=color,linewidth=5.0)


    plt.streamplot(X,Y,a[:,:,0],a[:,:,2],density=density,color=(0.0,0.59*0.8,1*0.8,0.7))
    #show_map(a[:,:,1])
    # plt.text(20, a.shape[1]-20, '\n'.join([f"{k}:{' '*max(0,15-len(k))} {v:.2f}" for k,v in coeffs.items()]), fontsize = 10, 
    #      bbox = dict(facecolor = 'white', alpha = 0.75))
def stream_plot2(coeffs=None,density=700,streamplot_density=4):
    c : Context = Context()
    gs = c.data_context.global_simulation
    points = [c.canvas._stack_coord_to_micron(p) for p in c.canvas.skeleton_graph]
    cmin = min(min(p[0] for p in points),min(p[2] for p in points))
    cmax = max(max(p[0] for p in points),max(p[2] for p in points))
    if coeffs is None:
        coeffs=gs.coeffs
    #a=generate_line.fill_array_cone(density,gs,center=gs.center,**coeffs,**gs.cone_coeffs)
    center = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label("c"))
    adult_lines=[]
    young_lines=[]
    for i in range(15):
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            break

        adult_lines.append((c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))),i))
    for i in range(-6,0):
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            continue
        young_lines.append(c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))))
    print(cmin,cmax)
    a,_cmin,_cmax=generate_line.fill_array_cone_from_lines(density,adult_lines,young_lines,center=center,**coeffs,**gs.cone_coeffs)
    v=stream_scale = (cmax-cmin)/700 
    print(cmin,cmax)
    x=np.zeros((a.shape[1],a.shape[0],a.shape[2]))
    x[:,:,0]=a[:,:,0].T
    x[:,:,1]=a[:,:,1].T
    x[:,:,2]=a[:,:,2].T
    plt.close()
    plt.clf()
    plt.subplots(figsize=(10,10))
    X=np.linspace(0,a.shape[1],a.shape[1])
    Y=np.linspace(0,a.shape[0],a.shape[0])
    #lines skel
    for line in gs.lines:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([(x-cmin)/v for x in xx],[(y-cmin)/v for y in yy],c=color,linewidth=5.0)
    for line in gs.in_progres:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="red"
            plt.plot([(x-cmin)/v for x in xx],[(y-cmin)/v for y in yy],c=color,linewidth=5.0)
    #plt.text(0, 0, pprint.pformat(coeffs), fontsize = 10, 
    #     bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.streamplot(X,Y,a[:,:,0],a[:,:,2],density=streamplot_density,color=(0.0,0.59*0.8,1*0.8,0.7))



def save_streamplot(path):
    #plt.savefig("C:\\Users\\Andrzej\\Desktop\\2send_1\\skeletons_model_mediumSAMs\\stream plots\\test.jpg",bbox_inches="tight")
    plt.savefig(path,bbox_inches="tight")




from simulation_initialization import *

# # c=Context()
# #stream plots
# path="C:\\Users\\Andrzej\\Desktop\\2send_1\\PISANIE\\ilustracje"
# init_stream_plot()
# # stream_plot(gs.coeffs)
# # save_streamplot(f"{path}\\normal.svg")
# # gs.coeffs["inhibition_coef"]*=0.7
# # stream_plot(gs.coeffs)
# # save_streamplot(f"{path}\\inhibition07.svg")
# # gs.coeffs["inhibition_coef"]*=0.7
# # stream_plot(gs.coeffs)
# # save_streamplot(f"{path}\\inhibition07.svg")
# gs.coeffs["inhibition_coef"]
# gs.coeffs['straight_coef']*=0.5
# stream_plot(gs.coeffs)
# save_streamplot(f"{path}\\center05.svg")
#c.data_context.load_data(pickle.load(open("C:\\Users\\Andrzej\\Desktop\\2send_1\\skeletons_model_mediumSAMs\\calculation\\plots\\xp14_25_corr_S_c_model_e.ske","rb")))
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 1.9380879937972648,
#  'attraction_coef': 0,
#  'neg_range': 45.504265466109196,
#  'straight_coef': 2.6968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.7007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}repulsion.jpg")
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 1.9380879937972648,
#  'attraction_coef': 0,
#  'neg_range': 45.504265466109196,
#  'straight_coef': 5.6968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.7007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}high acropetal extension.jpg")
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 1.9380879937972648,
#  'attraction_coef': 0,
#  'neg_range': 45.504265466109196,
#  'straight_coef': 0.3968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.7007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}low acropetal extension.jpg")
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 0.9380879937972648,
#  'attraction_coef': 0.7,
#  'neg_range': 0.0,
#  'straight_coef': 0.6968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.0007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}only positive.jpg")
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 1.9380879937972648,
#  'attraction_coef': 6,
#  'neg_range': 45.504265466109196,
#  'straight_coef': 2.6968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.7007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}high positive.jpg")
# coeffs = {'pos_range': 94.6139871413999,
#  'age_coef': 0.0898168675836104,
#  'inhibition_coef': 3.9380879937972648,
#  'attraction_coef': 0,
#  'neg_range': 45.504265466109196,
#  'straight_coef': 2.6968647751699355,
#  'inertia': 4.968567709663606,
#  'peak_coef': 0.7007414561054597,
#  'age_cut_off_coef': 1.3488342990477376}
# stream_plot(coeffs)
# save_streamplot(f"{path}high negative.jpg")



# c.skeleton_distances.add_button(
#             "Calculate", (20, 0), (1, 10), lambda x: fill_array_cone())

#plt.savefig('aaa.jpg',dpi=300)





#random syntetic generation test

import copy
def _f():
    def normal_growth():
        gs = c.data_context.global_simulation
        gs.meristem_grow_speed=1
        gs.growth_coeffs_young=dict(A=0.114, B=2.842, C=50.03)
        gs.growth_coeffs=gs.growth_coeffs_young
        gs.growth_coeffs_old=dict(A=0.2, B=6, C=62.6)
    return normal_growth

ft=[_f() for i in range(20)]
for i,x in enumerate(ft):
    x.name='test ' + str(i)
    
c.simulation_controller.functions_to_test=ft
c.simulation_controller.init_function=init_random


def load_coefs():
    indices=[]
    with open('results_2.txt','r') as file:
        lines=[]
        lines = file.readlines()
        indices=[]
        for i,l in enumerate(lines):
            if 'new test' in l and (i>0 and 'best so far' in lines[i-1]):
                indices.append(i-2)
    best=[]
    for i in indices:
        a=eval(lines[i])
        best.append((a,float(lines[i+1][len('best so far '):-1])))

    best.sort(key=lambda x: x[1])

    logs=open('stream_plots/log.txt','w')
    for i,(coeffs,distance) in enumerate(best):
        print(f'{i}/{len(best)}')
        fill_array_cone(coeffs=coeffs,density=500)
        stream_plot()
        plt.savefig(f'stream_plots/{i}.jpg',dpi=300)
        plt.clf()
        logs.writelines([str(coeffs),str(distance),'\n'])

#init_dr90()
#load_coefs()

def failed(filename,outpath):
    failed=pickle.load(open(filename,'rb'))[:10_000][-100:]
    for i,(coeffs,distance) in enumerate(failed):
            print(f'{i}/{len(failed)}')
            fill_array_cone(coeffs=coeffs,density=500)
            stream_plot()
            plt.savefig(f'{outpath}{i}.jpg',dpi=300)
            plt.clf()

def correct():
    for v,i in list(c.data_context.labels['point'].items()):
        if len(i) == 4:
            print(i)
            c.data_context.labels['point'][v]=(i[:2]+i[3]+i[2])
    for v,i in list(c.data_context.labels['label'].items()):
        if len(i) == 4:
            print(i)
            c.data_context.labels['label'][v]=(i[:2]+i[3]+i[2])

import pyautogui
import pickle
def screen(name):
    pyautogui.screenshot(f"exploration_results\\comparison\\{name}.jpg",region=(0,30,2050,2050))

def get_best(name,sufix='max'):
    x=pickle.load(open(f".\\exploration_results\\optimalization\\dump_{name}_{sufix}",'rb'))
    x.sort(key=lambda a: a[1])
    return x[0]

def ff(name,sufix='max'):
    from  calculation_config import config_dict
    v,dist=get_best(name,sufix)
    print(dist)
    l=config_dict[name][0].labels_list[0]
    c.data_context.CIM._generate_lines(l,**v,draw=True)

def get_edge(distances):
    max_value = max(distances)
    min_value = min(distances)
    if -min_value>max_value:
        return min_value
    else:
        return max_value

def get_sum(distances,key):
    r=list(filter(key,distances))
    if r:
        return sum(r)/len(r)
    else:
        return 0

#calculate_all()
# #histograms
def calculate_histograms(name):
    distance_data=pickle.load(open(f"C:\\Users\\Andrzej\\Desktop\\2send_1\\{name}\\checked\\dump","rb"))
    _calculate_histograms(distance_data,name)

def calculate_combined(*args,out_name="combined"):
    distance_data={"center":{}}
    for name in args:
        distance_data["center"].update(pickle.load(open(f"C:\\Users\\Andrzej\\Desktop\\2send_1\\{name}\\checked\\dump","rb"))["center"])
        print(len(distance_data["center"].items()))
    _calculate_histograms(distance_data,out_name)


def _calculate_histograms(distance_data,name):
    _taken =list(distance_data["center"].items())
    for random_type in ["center"]:
        for i in ["5","8","13","all"]:
        #for i in ["all"]:
            if i!="all":
                taken = list(filter (lambda x: x[0][2]-x[0][1]==i,list(distance_data[random_type].items())))
            else:
                taken = list(distance_data[random_type].items())
            distances = [generate_line.distance_lines_iv(v[0],v[-4]) for k,v in taken if len(k)==4]


            dabs = [max(map(abs,d)) for d in distances]
            dmax = [max(d) for d in distances if max(d)>1]
            dmin = [min(d) for d in distances if min(d)<-1]
            apos = [get_sum(d,lambda x:x>0) for d in distances if get_sum(d,lambda x:x>0)!=0]
            aneg = [get_sum(d,lambda x:x<0) for d in distances if get_sum(d,lambda x:x>0)!=0]
            de = [get_edge(d) for d in distances]
            #calculate_all()
            distances2 = [generate_line.distance_lines_iv(v[-1][1],v[-4]) for k,v in taken if len(k)==4]
            dabs2 = [max(map(abs,d)) for d in distances2]
            de2 = [get_edge(d) for d in distances2]
            dmax2 = [max(d) for d in distances2 if max(d)>1]
            dmin2 = [min(d) for d in distances2 if min(d)<-1]
            apos2 = [get_sum(d,lambda x:x>0) for d in distances2 if get_sum(d,lambda x:x>0)!=0]
            aneg2 = [get_sum(d,lambda x:x<0) for d in distances2 if get_sum(d,lambda x:x>0)!=0]
            path = f"C:\\Users\\Andrzej\\Desktop\\2send_1\\{name}\\hist\\"


            import matplotlib.patches as mpatches

            c1 = 'green'
            c2 = 'violet'

            plt.hist(dabs, bins=20,color=c1)
            plt.hist(dabs2,bins=20,alpha=0.5,color=c2)

            empiric_patch = mpatches.Patch(color=c1, label='empiric data')
            model_patch = mpatches.Patch(color=c2, label='modeled data')
            plt.legend(handles=[empiric_patch,model_patch])
                    

            
            plt.savefig(f"{path}{random_type} comparison for dabs n+{i}.png")

            plt.clf()

            plt.hist(dmax, bins=20,color=c1)
            plt.hist(dmax2,bins=20,alpha=0.5,color=c2)

            empiric_patch = mpatches.Patch(color=c1, label='empiric data')
            model_patch = mpatches.Patch(color=c2, label='modeled data')
            plt.legend(handles=[empiric_patch,model_patch])
                    

            plt.savefig(f"{path}{random_type} comparison for dmax n+{i}.png")
            plt.clf()

            plt.hist(dmin, bins=20,color=c1)
            plt.hist(dmin2,bins=20,alpha=0.5,color=c2)

            empiric_patch = mpatches.Patch(color=c1, label='empiric data')
            model_patch = mpatches.Patch(color=c2, label='modeled data')
            plt.legend(handles=[empiric_patch,model_patch])
                    
            plt.savefig(f"{path}{random_type} comparison for dmin n+{i}.png")
            plt.clf()
            
            plt.hist(apos, bins=20,color=c1)
            plt.hist(apos2,bins=20,alpha=0.5,color=c2)

            empiric_patch = mpatches.Patch(color=c1, label='empiric data')
            model_patch = mpatches.Patch(color=c2, label='modeled data')
            plt.legend(handles=[empiric_patch,model_patch])

            plt.savefig(f"{path}{random_type} comparison for apos n+{i}.png")
            plt.clf()

            plt.hist(aneg, bins=20,color=c1)
            plt.hist(aneg2,bins=20,alpha=0.5,color=c2)

            empiric_patch = mpatches.Patch(color=c1, label='empiric data')
            model_patch = mpatches.Patch(color=c2, label='modeled data')
            plt.legend(handles=[empiric_patch,model_patch])
            
            plt.savefig(f"{path}{random_type} comparison for aneg n+{i}.png")
            plt.clf()


#calculate_combined("smallSAM","mediumSAM","bigSAM")

# coeffs = {'pos_range': 87.08772940705788,
#   'age_coef': 0.09272969738881419,
#   'inhibition_coef': 2.3262779227846035,
#   'attraction_coef': 1.2606430163271114,
#   'neg_range': 49.92900966738538,
#   'straight_coef': 2.098113713185219,
#   'inertia': 5.312337338544549,
#   'peak_coef': 0.6897034724068661,
#   'age_cut_off_coef': 1.1001373558417338}

#coeffs = {'pos_range': 87.08772940705788,
#  'age_coef': 0.09272969738881419,
#  'inhibition_coef': 1.1262779227846035,
#  'attraction_coef': 1.2606430163271114,
#  'neg_range': 49.92900966738538,
#  'straight_coef': 2.098113713185219,
#  'inertia': 5.312337338544549,
#  'peak_coef': 0.6897034724068661,
#  'age_cut_off_coef': 1.1001373558417338}
#init_dr90()
#a=stream_plot(coeffs,density=5)




path="C:\\Users\\Andrzej\\Desktop\\2send_1\\skeletons_model_mediumSAMs\\calculation\\"+"xp14_21_corr_S_c_model.ske"
labels=('-1', '4', '7', '12')
config={'pos_range': 63.65102397236054,
  'age_coef': 0.1032757066968491,
  'inhibition_coef': 2.1413542425547676,
  'attraction_coef': 0.01,
  'neg_range': 44.09566909178031,
  'straight_coef': 2.998260929784881,
  'inertia': 4.914092611985757,
  'peak_coef': 0.5533656945287674,
  'age_cut_off_coef': 1.5883074050424635}
#c.data_context.CIM.generate_single_line(path,labels,config)
#c.data_context.CIM._generate_lines(labels_list[-1],**coeffs,draw=True)
coeffs={'pos_range': 132.0, 'age_coef': 0.07000000000000005, 'inhibition_coef': 1.848421052631579, 'attraction_coef': 0.27263157894736845, 'neg_range': 47.373684210526314, 'straight_coef': 2.5560733141578544, 'inertia': 1.5, 'peak_coef': 0.4991447183847797, 'age_cut_off_coef': 1.3757699971976245}
labels_list=[
                [('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7'),('0', '5', '8', '13')],
                [('-5', '0', '3', '8'),('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7','12'),('0', '5', '8', '13')],
                [('-5', '0', '3'),('-4', '1', '4', '9'),('-3', '2', '5'),('-2', '3', '6', '11'),('-1', '4', '7', '12'),('0', '5', '8', '13')],
                [('-4', '1', '4','9'),('-3', '2', '5', '10'),('-2', '3', '6'),('-1', '4', '7','12')],
                [('-4', '1', '4'),('-3', '2', '5'),('-2', '3', '6'),('-1', '4', '7', '12'),('0', '5', '8')],
                [('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7'),('0', '5', '8','13')],
                [('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7', '12'),('0', '5', '8', '13')],
                [('-5', '3', '8'),('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7', '12'),('0', '5', '8', '13')],
                [('-5', '0', '3', '8'),('-4', '1', '4', '9'),('-2', '3', '6', '11'),('-1', '4', '7', '12')],
                [('-6', '-1', '2', '7'),('-5', '0', '3', '8'),('-4', '1', '4', '9'),('-3', '2', '5', '10'),('-2', '3', '6', '11'),('-1', '4', '7', '12'),('0', '5', '8', '13')],
            ]

from simulation_initialization import *

#init_small_idealized()
#init_xp41_dr35()
#init_xp42_dr48()
#init_maximum_attraction(init_xp41_dr35)
#init_maximum_attraction()
#init_medium_idealized()
# init_small_idealized(suffix="inhibition_coef_32",coeffs={'pos_range': 20.6139871413999,
#             'age_coef': 0.03963136053954481,
#             'inhibition_coef': 3.2873971411850995,
#             'attraction_coef': 0.0,
#             'neg_range': 59.22591211160096,
#             'straight_coef': 3.0624121912173043,
#             'inertia': 0.5,
#             #'inertia': 1.,
#             'peak_coef': 0.99,
#             'age_cut_off_coef': 0.8961433833351786})
c.simulation_controller.screenshots_number=220
#init_small_idealized()
#init_xp5_dr5()
#c.simulation_controller.screenshots_number=440
c.simulation_controller.screenshots_number=220
#c.simulation_controller.screenshots_number=332
#c.simulation_controller.step(190,6)

#run lucas
#init_lucas()
#init_bijugate_idealized()
c.simulation_controller._screen_reset_zoom=1.1
# c.simulation_controller.fullHD=True

c.main_sidebar.scale_field.SetValue("1.1")

gs=c.data_context.global_simulation

c.simulation_controller.do_save_images=True
step=c.simulation_controller.step

run = c.simulation_controller.run_calculation_thread

def step2(i):
    for _ in range(i):
        step(3,0.1)

def load_gs():
    global gs
    gs = pickle.load(open("global_simulation_dump","rb"))
    gs.context = c
    c.data_context.global_simulation=gs



#knees = sorted([(k,gs.neighbour_angle(skeleton,k),gs.distance_points(skeleton,start,k)) for k in skeleton.keys() if gs.neighbour_angle(skeleton,k)<2.7],key=lambda x: x[1])


def set_as_real_time():
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)
set_as_real_time()

#run()

import run_experiment

c.data_context.settings.PRINT_MODE=PrintMode.INDEX_ONLY



def run_simulation():
    c.data_context.new_simulation_iterator=run_experiment.run_lucas()
    next(c.data_context.new_simulation_iterator)

#init_small_idealized(suffix="_test")
# step2(300)
# x=c.simulation_controller.calculate_n_8_adult()
#run()

def run_average_ring(steps=700):
    gs.growth_coeff_development = growth_coeffs_no_change(2)
    gs.growth_coeff_development.growth_rate=[5]
    c.simulation_controller.screenshots_number=steps
    c.data_context.settings.HIGHEST_AGE=True
    c.data_context.settings.SAVE_IMAGES=True
    c.simulation_controller.run_calculation_thread()


# def move_along(a,b,t):
#     x=np.array(a)
#     y=np.array(b)
#     v=np.round((x-y)/np.linalg.norm(x-y)*t,4)
#     print(v)
#     print(x-y)
#     print(a,tuple(x+v))
#     c.data_context.move_point(a,tuple(x+v))
#     return tuple(x+v)
# ma=move_along
# b,a=[(369.0, 616.0, 1.3634),(323.0, 727.0, 1.2671)]
# gp = c.skeleton_distances.get_points
#with open("C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions visualization\\clearing\\img8_exp8_Dr5v2_8_ske5.ske","rb") as file:
#    c.data_context.load_data(pickle.load(file))


# import threading
# class stack_smooth(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         print("xxx")
#         pass
#     def run(self):
#         path = "C:\\Users\Andrzej\\Desktop\\2send_1\\infl\\smooth w=1.tif"
#         print("xxxx")
#         save_stack(c.data_context.create_skeleton_stack_smooth(1),name=path)
#         path = "C:\\Users\Andrzej\\Desktop\\2send_1\\infl\\smooth w=2.tif"
#         print("xxxx")
#         save_stack(c.data_context.create_skeleton_stack_smooth(2),name=path)
#         path = "C:\\Users\Andrzej\\Desktop\\2send_1\\infl\\smooth w=3.tif"
#         print("xxxx")
#         save_stack(c.data_context.create_skeleton_stack_smooth(3),name=path)

# c.graphic_context.curve_tension=0.2
#stack_smooth().start()


# TESTING POINTS TO SIMULATION DIR
# Code for stramplots
vv3=Vector3
from matplotlib import cm
def test_display(coeffs,center,radius_step,angular_density,start,end):
    P=generate_cone_points(coeffs,center,radius_step,angular_density,start,end)
    PS=[c.canvas._micron_coord_to_stack(x) for x in P]
    #positions_to_simulation_vectors
    c.graphic_context.mark_temporary_points(PS)
#test_display(gs.cone_coeffs,Vector3(gs.center)-Vector3((0,4,0)),4,4,20,300)
def draw_points_map():
    P=list(map(tuple,generate_cone_points(gs.cone_coeffs,Vector3(gs.center),4,4,40,300)))
    coeffs = dict(gs.coeffs)
    V=gs.positions_to_simulation_vectors(P,coeffs)
    colors=[
    cm.viridis(np.arccos(min(0.999,max(0.001,float((vv3(gs.center)-vv3(point)).normalised@vv3(d)))))/np.pi*2)[:3]
        for point,d in zip(P,V)
    ]
    P=list(map(tuple,generate_cone_points(gs.cone_coeffs,Vector3(gs.center)+Vector3((0,4,0)),4,4,40,300)))
    PS=[c.canvas._micron_coord_to_stack(x) for x in P]
    c.graphic_context.mark_color_points(PS,colors)
    c.data_context.settings.DEPTH_TEST_POINTS=True


# P=list(map(tuple,generate_cone_points(gs.cone_coeffs,Vector3(gs.center),10,10,40,300)))
# coeffs = dict(gs.coeffs)
# V=gs.positions_to_simulation_vectors(P,coeffs)
# P=list(map(tuple,generate_cone_points(gs.cone_coeffs,Vector3(gs.center)+Vector3((0,4,0)),10,10,40,300)))
# PS=[c.canvas._micron_coord_to_stack(x) for x in P]


# import shaders as sh
# def init_arrows():
#     gs= OpenGL.GL.shaders.compileShader(sh.arrow["geometry"], GL_GEOMETRY_SHADER)
#     vs= OpenGL.GL.shaders.compileShader(sh.arrow["vertex"], GL_VERTEX_SHADER)
#     fs=OpenGL.GL.shaders.compileShader(sh.arrow["fragment"], GL_FRAGMENT_SHADER)
#     print(glGetShaderInfoLog(vs))
#     print(glGetShaderInfoLog(gs))
#     print(glGetShaderInfoLog(fs))
#     program_arrow = OpenGL.GL.shaders.compileProgram(
#         vs,
#         gs,
#         fs,
#         #validate=False
#     )

#     PA=[c.canvas._stack_coord_to_opengl(c.canvas._micron_coord_to_stack(x)) for x in P]
#     VA=[tuple(vv3(c.canvas._stack_coord_to_opengl(
#         c.canvas._micron_coord_to_stack(tuple(vv3(p)+vv3(v)*1))
#         ))-vv3(pa)) 
#     for pa,p,v in zip(PA,P,V)]
#     arrow_vao = glGenVertexArrays(1)
#     vbo_points, vbo_directions = glGenBuffers(2)
#     arrow_size=len(PA)
#     c.canvas.skeleton_bundle._init_buffers(arrow_vao,vbo_points,vbo_directions,np.array(PA, dtype=np.float32),np.array(VA, dtype=np.float32),arrow_size)

#     def arrow_draw():
#         glUseProgram(program_arrow)
#         glBindVertexArray(arrow_vao)
#         glLineWidth(1)
#         glDrawArrays(GL_POINTS, 0, arrow_size)
#         glLineWidth(c.data_context.settings.LINE_WIDTH)
#     c.canvas.additional_draw.append((program_arrow,arrow_draw))
# init_arrows()
#init_streamplot_gs()
# with open("streamplot_data","rb") as file:
#     coeffs_list=pickle.load(file)
# def save_figs():
#     for i,(coeffs,cost) in enumerate(coeffs_list):
#         if i<449:
#             continue
#         stream_plot.stream_plot(coeffs=coeffs,density=2,streamplot_density=300,adult_lines_indices=[2,7,15],young_lines_indices=[-6])
#         plt.savefig(f"streamplots/{i}.png")

# kwargs = dict(
#     arrowsize=2,
#     arrowstyle="simple"
# )
# with open("streamplot_data","rb") as file:
#     coeffs_list=pickle.load(file)
# import stream_plot
# stream_plot.stream_plot(coeffs=coeffs_list[27][0],density=[1,1],streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
# plt.savefig(f"streamplots/1/bad.png")
# stream_plot.stream_plot(coeffs=coeffs_list[27][0],density=[1,1],streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
# plt.savefig(f"streamplots/1/bad.png")
# stream_plot.stream_plot(coeffs=gs.coeffs,density=[1,1],streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
# plt.savefig(f"streamplots/1/good.png")


# from importlib import reload
# def d():
#     reload(stream_plot)
#     for i in range(1,4):
#         stream_plot.stream_plot(coeffs=coeffs_list[27][0],density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/bad_{i}.eps")
#         stream_plot.stream_plot(coeffs=coeffs_list[27][0],density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/bad_{i}.svg")
#         stream_plot.stream_plot(coeffs=coeffs_list[27][0],density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/bad_{i}.png")
#         stream_plot.stream_plot(coeffs=gs.coeffs,density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/good_{i}.png")
#         stream_plot.stream_plot(coeffs=gs.coeffs,density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/good_{i}.eps")
#         stream_plot.stream_plot(coeffs=gs.coeffs,density=i,streamplot_density=301,adult_lines_indices=[2,7,15],young_lines_indices=[-6],**kwargs)
#         plt.savefig(f"streamplots/1/good_{i}.svg")
# d()
#for points

# i=7
# x=find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i)))
# e=[a for a,b in x.items() if len(b)==1]
# e
# c.graphic_context.mark_temporary_points(c.canvas.get_equal_points(*e,10))
