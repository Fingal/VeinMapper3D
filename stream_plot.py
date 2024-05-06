import generate_line
import matplotlib.pyplot as plt
from context import *
from skeleton import *
cmin,cmax=-322.9538809035303, 228.11805279419087
v=stream_scale = (cmax-cmin)/700 
c_min_x,c_min_y,c_max_x,c_max_y=0,0,0,0
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

def fill_array_cone(coeffs=None,streamplot_density=700,margin=20,adult_lines_indices=range(15),young_lines_indices=range(-6,0)):
    c : Context = Context()
    global c_min_x,c_min_y,c_max_x,c_max_y,v
    gs = c.data_context.global_simulation
    if coeffs is None:
        coeffs=gs.coeffs
    #a=generate_line.fill_array_cone(density,gs,center=gs.center,**coeffs,**gs.cone_coeffs)
    center = c.canvas._stack_coord_to_micron(c.data_context.get_point_from_label("c"))
    adult_lines=[]
    young_lines=[]
    for i in adult_lines_indices:
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            break

        adult_lines.append((c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))),i))
    for i in young_lines_indices:
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            continue
        young_lines.append(c.canvas._lines_coord_to_microns(graph_to_lines(find_compact_component(c.canvas.skeleton_graph,c.data_context.get_point_from_label(str(i))))))

    a,c_min_x,c_min_y,c_max_x,c_max_y=generate_line.fill_array_cone_from_lines(streamplot_density,adult_lines,young_lines,margin=margin,center=center,**coeffs,**gs.cone_coeffs)
    v = max(c_max_x-c_min_x,c_max_y-c_min_y)/streamplot_density
    x=np.zeros((a.shape[1],a.shape[0],a.shape[2]))
    x[:,:,0]=a[:,:,0].T
    x[:,:,1]=a[:,:,1].T
    x[:,:,2]=a[:,:,2].T
    a=x
    return a

def stream_plot(coeffs=None,density=3,streamplot_density=700,margin=20,adult_lines_indices=range(15),young_lines_indices=range(-6,0),**kwargs):
    c : Context = Context()
    gs = c.data_context.global_simulation
    for line in gs.in_progres:
        xxs,yys =line.get_segments()
    a=fill_array_cone(coeffs,streamplot_density,margin=margin,adult_lines_indices=adult_lines_indices,young_lines_indices=young_lines_indices)

    
    for i in adult_lines_indices:
        point = c.data_context.get_point_from_label(str(i))
        if point is None:
            continue


    _stream_plot(a,density=density,**kwargs)
    return a
    plt.text(0, 0, pprint.pformat(coeffs), fontsize = 10, 
         bbox = dict(facecolor = 'white', alpha = 0.5))
    #plt.quiver(X,Y,a[:,:,2],a[:,:,0])
    #plt.show()
def _stream_plot(a,density=3,start_x=40,end_x=225,start_y=80,end_y=300,**kwargs):
#start, end, x, y are temporary for one specific streamplot
    size_x=end_x-start_x
    size_y=end_y-start_y
    c : Context = Context()
    gs = c.data_context.global_simulation
    x_range=int((c_max_y-c_min_y)/v)
    y_range=int((c_max_x-c_min_x)/v)
    r = max(x_range,y_range)
    plt.close()
    plt.clf()
    plt.subplots(figsize=(10,(size_y)/size_x*10))
    plt.xlim((start_x,end_x))
    plt.ylim((start_y,end_y))
    X=np.linspace(0,r,r)
    Y=np.linspace(0,r,r)
    #lines skel
    c_min_x,c_min_y,c_max_x,c_max_y
    for line in gs.lines:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([(x-c_min_x)/v for x in xx],[(y-c_min_y)/v for y in yy],c=color,linewidth=5.0)
    for line in gs.in_progres:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="red"
            plt.plot([(x-c_min_x)/v for x in xx],[(y-c_min_y)/v for y in yy],c=color,linewidth=5.0)


    plt.streamplot(Y[start_x:end_x],X[start_y:end_y],a[start_y:end_y,start_x:end_x,0],a[start_y:end_y,start_x:end_x,2],density=density,color=(0.0,0.59*0.8,1*0.8,0.7),**kwargs)
    #show_map(a[:,:,1])
    # plt.text(20, a.shape[1]-20, '\n'.join([f"{k}:{' '*max(0,15-len(k))} {v:.2f}" for k,v in coeffs.items()]), fontsize = 10, 
    #      bbox = dict(facecolor = 'white', alpha = 0.75))
def stream_plot2(coeffs=None,density=700,streamplot_density=3):
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

from pyrr import Matrix33
from pyrr import Vector3

def rotate_points_around_center(theta,points):
    c = Context()
    center = Vector3(c.data_context.get_point_from_label("c"))
    rotation=Matrix33.from_z_rotation(theta)
    points_rotated = [Vector3(rotation@(Vector3(p)-center))+center for p in points]
    points_rotated = [tuple(round(x,2) for x in p) for p in points_rotated]
    return points_rotated

def rot_all(theta):
    c = Context()
    points=list(set(c.data_context.values["point"]) | set(c.data_context.values["label"])| set(c.canvas.skeleton_graph.keys()) )

    for l in {((602, 329, 65), (632, 353, 61)), ((602, 329, 65), (589, 316, 65)), ((516, 94, 47), (500, 101, 47)), ((637, 341, 65), (602, 329, 65)), ((543, 278, 70), (562, 295, 67)), ((551, 322, 74), (532, 336, 79)), ((429, 207, 70), (444, 196, 70)), ((453, 86, 31), (478, 47, 10)), ((665, 379, 26), (639, 374, 43)), ((632, 353, 61), (653, 363, 29)), ((532, 336, 79), (507, 349, 82)), ((411, 220, 73), (429, 207, 70)), ((589, 316, 65), (562, 295, 67)), ((560, 299, 67), (551, 322, 74)), ((444, 196, 70), (465, 174, 64)), ((602, 329, 65), (638, 345, 61))}:
        try:
            del c.data_context.labels["line"][l]
        except:
            pass
    points_rotated=rotate_points_around_center(theta,points)
    points_dict = {p:pr for p,pr in zip(points,points_rotated)}
    print("aaa1")
    c.data_context.values["line"]=[(points_dict[a],points_dict[b]) for a,b in c.data_context.values["line"]]
    print("aaa2")
    c.data_context.values["point"]=[points_dict[a] for a in c.data_context.values["point"]]
    print("aaa3")
    c.data_context.values["label"]=[points_dict[a] for a in c.data_context.values["label"]]
    print("aaa4")

    c.data_context.labels["line"]={(points_dict[a],points_dict[b]):v for (a,b),v in c.data_context.labels["line"].items()}
    print("aaa5")
    c.data_context.labels["point"]={points_dict[a]:v for a,v in c.data_context.labels["point"].items()}
    print("aaa6")
    c.data_context.labels["label"]={points_dict[a]:v for a,v in c.data_context.labels["label"].items()}
    print("aaa7")
    n = {}
    for a,b in c.canvas.skeleton_graph.items():
        n[points_dict[a]]=set(points_dict[x] for x in b)
    c.canvas.skeleton_graph=n
    #c.canvas.skeleton_graph={points_dict[a]:points_dict[b] for a,b in c.canvas.skeleton_graph.items()}
    print("aaa8")
    c.graphic_context.refresh()
    c.graphic_context.redraw_graph()
    #points_rotated=rotate_points_around_center(theta)


#rot_all(-145/180*3.14)

#p=rotate_points_around_center(90,points)
#p=dup_rot_labels(90)

# def r(a):
#     p=rotate_points_around_center(a/180*3.14,points)
#     c.graphic_context.mark_temporary_points(p)

def refleciton(points):
    c = Context()
    c1,c2,c3 = c.data_context.get_point_from_label("c")
    return [(a,-b+2*c2,c) for a,b,c in points]

def mod_skel(fun):
    c = Context()
    points=list(set(c.data_context.values["point"]) | set(c.data_context.values["label"])| set(c.canvas.skeleton_graph.keys()) )

    for l in {((602, 329, 65), (632, 353, 61)), ((602, 329, 65), (589, 316, 65)), ((516, 94, 47), (500, 101, 47)), ((637, 341, 65), (602, 329, 65)), ((543, 278, 70), (562, 295, 67)), ((551, 322, 74), (532, 336, 79)), ((429, 207, 70), (444, 196, 70)), ((453, 86, 31), (478, 47, 10)), ((665, 379, 26), (639, 374, 43)), ((632, 353, 61), (653, 363, 29)), ((532, 336, 79), (507, 349, 82)), ((411, 220, 73), (429, 207, 70)), ((589, 316, 65), (562, 295, 67)), ((560, 299, 67), (551, 322, 74)), ((444, 196, 70), (465, 174, 64)), ((602, 329, 65), (638, 345, 61))}:
        try:
            del c.data_context.labels["line"][l]
        except:
            pass
    points_rotated=fun(points) 
    points_dict = {p:pr for p,pr in zip(points,points_rotated)}
    print("aaa1")
    c.data_context.values["line"]=[(points_dict[a],points_dict[b]) for a,b in c.data_context.values["line"]]
    print("aaa2")
    c.data_context.values["point"]=[points_dict[a] for a in c.data_context.values["point"]]
    print("aaa3")
    c.data_context.values["label"]=[points_dict[a] for a in c.data_context.values["label"]]
    print("aaa4")

    c.data_context.labels["line"]={(points_dict[a],points_dict[b]):v for (a,b),v in c.data_context.labels["line"].items()}
    print("aaa5")
    c.data_context.labels["point"]={points_dict[a]:v for a,v in c.data_context.labels["point"].items()}
    print("aaa6")
    c.data_context.labels["label"]={points_dict[a]:v for a,v in c.data_context.labels["label"].items()}
    print("aaa7")
    n = {}
    for a,b in c.canvas.skeleton_graph.items():
        n[points_dict[a]]=set(points_dict[x] for x in b)
    c.canvas.skeleton_graph=n
    #c.canvas.skeleton_graph={points_dict[a]:points_dict[b] for a,b in c.canvas.skeleton_graph.items()}
    print("aaa8")
    c.graphic_context.refresh()
    c.graphic_context.redraw_graph()


# init_streamplot_gs()

# gs=c.data_context.global_simulation
# #stream_plot()

# import stream_plot
# #stream_plot.stream_plot(adult_lines_indices=[2,7,15],young_lines_indices=[-6])

# from importlib import reload
# def redraw_stream_plot():
#     reload(stream_plot)
#     stream_plot.stream_plot(adult_lines_indices=[2,7,15],young_lines_indices=[-6])
#     plt.show()

# coeffs={'pos_range': 20.6139871413999,
#         'age_coef': 0.03963136053954481,
#         'inhibition_coef': 2.7873971411850995,
#         'attraction_coef': 0.0,
#         'neg_range': 59.22591211160096,
#         'straight_coef': 3.0624121912173043,
#         'inertia': 0.5,
#         #'inertia': 1.,
#         'peak_coef': 0.99,
#         'age_cut_off_coef': 0.8961433833351786}

# def stream_plot_coeffs():
#     stream_plot.stream_plot(coeffs=coeffs,density=2,streamplot_density=300,adult_lines_indices=[2,7,15],young_lines_indices=[-6])
#     plt.show()

# import pickle
# with open("streamplot_data","rb") as file:
#     coeffs_list=pickle.load(file)
# def save_figs():
#     for i,(coeffs,cost) in enumerate(coeffs_list):
#         if i<449:
#             continue
#         stream_plot.stream_plot(coeffs=coeffs,density=2,streamplot_density=300,adult_lines_indices=[2,7,15],young_lines_indices=[-6])
#         plt.savefig(f"streamplots/{i}.png")


# save_figs()



def skeleton_to_vector_graphics(gs:GlobalSimulation,path:str):
    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    primordia = []
    for line in gs.lines:
        primordia.append(gs.points[line.label_pos])
    
    plt.scatter([x for x,y,z in primordia],[z for x,y,z in primordia],c="grey",s=3,zorder=3)

    for line in gs.lines:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([x for x in xx],[y for y in yy],c=color,linewidth=1.0)

    for line in gs.in_progres:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([x for x in xx],[y for y in yy],c=color,linewidth=1.0)
    plt.savefig(f"{path}.eps")
    plt.savefig(f"{path}.svg")
