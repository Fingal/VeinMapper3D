import pdb
from typing import *
import context
import global_simulation
import skeleton
import generate_line
from canvas import PointsMapDrawer

import random
from PIL import ImageGrab
from time import sleep
from math import floor
import pickle
import threading
import copy
import pyautogui
from math import ceil
from pyrr import Vector3
import wx
import threading
import time
import winsound
import matplotlib.pyplot as plt
#ffmpeg -r 8 -i step_%03d.png -c:v libx264 -vf fps=24 -pix_fmt yuv420p out.mp4
#ffmpeg -r 8 -i rotated_step_%03d.png -c:v libx264 -vf fps=24 -pix_fmt yuv420p rotated_out.mp4

c=context.Context()

def skeleton_to_vector_graphics(gs:global_simulation.GlobalSimulation,path:str):
    
    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    primordia = []
    color="black"
    plt.xlim=(-500,500)
    plt.ylim=(-500,500)
    plt.plot([-25,25],[-450,-450],c="blue",linewidth=1.0)
    for line in gs.lines:
        primordia.append(gs.points[line.label_pos])
    for p in gs.surface_points.existing_points:
        primordia.append(p.position)
    plt.scatter([x for x,y,z in primordia],[-z for x,y,z in primordia],c="grey",s=3,zorder=3)
    plt.scatter([gs.center[0]],[-gs.center[2]],c="black",s=3,zorder=3)

    for line in gs.lines:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            plt.plot([x for x in xx],[-y for y in yy],c=color,linewidth=1.0)

    for line in gs.in_progres:
        xxs,yys =line.get_segments()
        for xx,yy in zip(xxs,yys):
            color="black"
            plt.plot([x for x in xx],[-y for y in yy],c=color,linewidth=1.0)
    plt.savefig(f"{path}.eps")
    plt.savefig(f"{path}.svg")

def make_screenshots(index):
    pyautogui.click(x=3552, y=1423, button='left')
    #pyautogui.click(x=3545, y=1456, button='left')
    pyautogui.moveTo(x=914, y=1010)  
    pyautogui.dragRel(10, -100, duration=0.4)
    sleep(0.2)
    pyautogui.screenshot(f"{c.data_context.image_path}\\rotated_step_{index:03}.png",region=(0,30,3100,2050))  
    pyautogui.dragRel(-10, 100, duration=0.4)
    pyautogui.click(x=914, y=1010, button='left')
    sleep(0.2)
    pyautogui.screenshot(f"{c.data_context.image_path}\\step_{index:03}.png",region=(0,30,3100,2000))  

def make_single_screenshot(filename,path='images'):
    sleep(0.25)
    pyautogui.click(x=3552, y=1423, button='left')
    #pyautogui.click(x=3545, y=1456, button='left')
    pyautogui.moveTo(x=2014, y=1010)  
    pyautogui.dragRel(10, -100, duration=0.4)
    sleep(0.2)
    pyautogui.dragRel(-10, 100, duration=0.4)
    sleep(0.2)
    pyautogui.screenshot(f"{c.data_context.image_path}\\{filename}.png",region=(0,30,3100,2050))


def make_single_screenshot_full_path(filename):
    sleep(0.25)
    pyautogui.click(x=3552, y=1423, button='left')
    #pyautogui.click(x=3545, y=1456, button='left')
    pyautogui.moveTo(x=2014, y=1010)  
    pyautogui.dragRel(10, -100, duration=0.4)
    sleep(0.2)
    pyautogui.dragRel(-10, 100, duration=0.4)
    sleep(0.2)
    pyautogui.screenshot(filename,region=(0,30,3400,2050))

    
def thread_function(arg,function=make_screenshots,**kwargs):
    function(arg,**kwargs)
    pyautogui.click(x=3487, y=1289, button='left') 
    pyautogui.click(x=3331, y=1143, button='left')    
    #pyautogui.click(x=3553, y=1265, button='left')    

colors=[(0.3, 0.0, 0.0),
 (0.7, 0.0, 0.0),
 (1.0, 0.0, 0.0),
 (0.0, 0.3, 0.0),
 (0.0, 0.7, 0.0),
 (0.0, 1.0, 0.0),
 (0.0, 0.0, 0.3),
 (0.0, 0.0, 0.7),
 (0.0, 0.0, 1.0),
 (0.3, 0.3, 0.0),
 (0.7, 0.7, 0.0),
 (0.9, 0.9, 0.0),
 (0.3, 0.0, 0.3),
 (0.7, 0.0, 0.7),
 (1.0, 0.0, 1.0),
 (0.0, 0.3, 0.3),
 (0.0, 0.7, 0.7),
 (0.0, 1.0, 1.0),
 (0.3, 0.7, 0),
 (0.3, 1, 0),
 (0.7, 0.3, 0),
 (0.7, 1, 0),
 (1, 0.3, 0),
 (1, 0.7, 0),
 (0.3, 0, 0.7),
 (0.3, 0, 1),
 (0.7, 0, 0.3),
 (0.7, 0, 1),
 (1, 0, 0.3),
 (1, 0, 0.7),
 (0, 0.3, 0.7),
 (0, 0.3, 1),
 (0, 0.7, 0.3),
 (0, 0.7, 1),
 (0, 1, 0.3),
 (0, 1, 0.7)]

class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""
    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(context.Context().data_context.EVT_RESULT_ID)
        self.data = data

class SimulatedData:
    def __init__(self):
        self.angles_labels=[]
        self.distances_data = []
        self.distances_data_adult = []

class SimStep(threading.Thread):
    def __init__(self,notify_window,gs,steps,time_step,save_images=True,run_calculation=False):
        threading.Thread.__init__(self)
        self.notify_window=notify_window
        self.steps=steps
        self.time_step=time_step
        self.gs=gs
        self.save_images=save_images
        self.run_calculation=run_calculation
    def run(self):
        #force simulation steps number based on growth speed
        if self.run_calculation:
            self.gs.simulation(self.steps,time_step=self.time_step)
        if self.save_images:
            time.sleep(1)
        else:
            time.sleep(0.05)
        wx.PostEvent(self.notify_window,ResultEvent(self.time_step))

class TestFunction(threading.Thread):
    def __init__(self,func):
        threading.Thread.__init__(self)
        self.func=func
    def run(self):
        self.func()

  

class SimulationController:
    def __init__(self):
        self.simulated_data = SimulatedData()
        self.do_save_images=False
        self.print_screaner = None
        self.colors=colors
        self.index = 0
        self.conducted_steps = 0
        self.context : context.Context = context.Context()
        self.context.simulation_controller = self
        self.time_step = 0.1

        self.screenshots_number=-1

        self.color_map = {}

        self.functions_to_test=[]
        self.init_function = None
        self.growth_statistics={'length_difference':[],'absolute_distance':[],'line_growth':[]}

        self.filenames_iterator = self._get_filenames()

        self._screen_reset_zoom=1.23
        self.fullHD=False

        self.point_map_drawer=PointsMapDrawer()

    
    def save_prediction(self,sufix=''):
        path=f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\predicted_steps\\skeleton_step_{self.conducted_steps}-{sufix}.ske"
        file=open(path,'wb')
        self.context.data_context.save_data(file)
        file.close()

    def step(self,steps : int =32, time_step : float = 1.,sufix=''):
        import pdb; pdb.stack_trace()
        gs = self.context.data_context.global_simulation
        #force simulation steps number based on growth speed
        pdb;pdb.set_trace()
        gs.simulation(steps,time_step=time_step)
        self.context.data_context.load_from_global_simulation()
        self.conducted_steps+=time_step
        self.save_prediction(sufix=sufix)
        print(self.conducted_steps)
        points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in gs.in_progres if x.is_left]
        surface_points = [self.context.canvas._micron_coord_to_stack(point.position) for  point in gs.surface_points.existing_points]
        
        if self.context.data_context.settings.SHOW_SURFACE_POINTS:
            self.context.graphic_context.mark_temporary_points(points+surface_points)
        else:
            self.context.graphic_context.mark_temporary_points(points)


        
    
    def make_screenshot_wx(self,filename="test.png",screen_size=(3192,2030)):
        c=self.context
        #c.canvas.SetFocus()
        c.canvas.reset_position(self._screen_reset_zoom)
        c.canvas.Update()
        c.canvas.Refresh()
        screen_offset=0,33
        bmp =wx.Bitmap(*screen_size)
        memDC = wx.MemoryDC()
        memDC.SelectObject(bmp)
        memDC.Blit(0,0,*screen_size,c.canvas.dc,*screen_offset)
        img = bmp.ConvertToImage()
        img.SaveFile(filename, wx.BITMAP_TYPE_PNG)

    def post_proscessing(self):
        self.context.data_context.load_from_global_simulation()
        self.conducted_steps+=0.1
        self.make_screenshot_wx(f"./test/{self.index:03}")
        self.index += 1
        print(self.conducted_steps)
        points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in gs.in_progres if x.is_left]
        surface_points = [self.context.canvas._micron_coord_to_stack(point.position) for  point in gs.surface_points.existing_points if self.context.data_context.settings.SHOW_SURFACE_POINTS]
        young_points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in  gs.in_progres in x.maturing==True]
        
        self.context.graphic_context.mark_temporary_points(points+surface_points+young_points)
        self.make_screenshot_wx()
        self.make_screenshot_wx()
        self.make_screenshot_wx()

    def set_color_map(self,color_map):
        self.color_map=color_map
        
        #self.context.graphic_context.mark_temporary_points(list(self.context.canvas.skeleton_graph.keys()))


    #step()

    # save_prediction()
    # for i in range(10):
    #     step()
    #     save_prediction()
    # x=skeleton.graph_to_lines(skeleton.find_compact_component(self.context.canvas.skeleton_graph,self.context.data_context.get_point_from_label('7')))
    # print(len(x))
    # self.context.graphic_context.set_lines_color(x,(1,1,0))

    def _calculate_distance_young(self,_other_line,line_8):
        gs : global_simulation.GlobalSimulation = self.context.data_context.global_simulation
        points=[]
        try:
            connected : global_simulation.YoungLine = max(_other_line.connected_secondary,key = lambda x: x.connection)
            points.append(generate_line.point_on_path_distance_reversed(connected.line,len(connected.line)-1,gs.YOUNG_DISTANCE))
            points.append(generate_line.point_on_path_distance_reversed(_other_line.line,connected.connection,gs.YOUNG_DISTANCE))
        except:
            points.append(generate_line.point_on_path_distance_reversed(_other_line.line,len(_other_line.line)-1,gs.YOUNG_DISTANCE+40))
        if line_8.line_n_8 is not None:
            l_8=copy.copy(line_8.line_n_8.line)
            try:
                l_8.extend(line_8.line_n_8.primary.line[line_8.line_n_8.connection:])
            except:
                pass
            distance = min(generate_line.calculate_distance(points,[(a,b) for a,b in zip(l_8,l_8[1:])]))
        else:
            distance = min(sum((a-b)**2 for a,b in zip(p,line_8.position_n_8))**0.5 for p in points)
        return distance

    def _calculate_distance_adult(self,line_5 : global_simulation.AdultLine,line_8 : global_simulation.AdultLine):
        distance=-1
        if line_5.line_n_5 is not None:
            if line_8.line_n_8 is not None:
                    if line_5.line_n_5 in line_8.line_n_8.connected_secondary or line_8.line_n_8 in line_5.line_n_5.connected_secondary:
                        return 0
                    l_8=copy.copy(line_8.line_n_8.line)
                    if line_8.line_n_8.primary is not None:
                        l_8.extend(line_8.line_n_8.primary.line[line_8.line_n_8.connection:])
                    distance=min(generate_line.calculate_distance(line_5.line_n_5.line,[(a,b) for a,b in zip(l_8,l_8[1:])]))
            else:
                l_5=line_5.line_n_5.line
                distance=generate_line.calculate_distance([line_8.position_n_8],[(a,b) for a,b in zip(l_5,l_5[1:])])[0]
        else:
            if line_8.line_n_8 is not None:
                l_8=copy.copy(line_8.line_n_8.line)
                if line_8.line_n_8.primary is not None:
                    l_8.extend(line_8.line_n_8.primary.line[line_8.line_n_8.connection:])
                distance=generate_line.calculate_distance([line_5.position_n_5],[(a,b) for a,b in zip(l_8,l_8[1:])])[0]
            else:
                distance = sum((a-b)**2 for a,b in zip(line_5.position_n_5,line_8.position_n_8))**0.5
        return distance

    def calculate_n_8_adult(self,test=None) -> List[Tuple[float,float,float]]:
        gs : global_simulation.GlobalSimulation = self.context.data_context.global_simulation
        lines = sorted(gs.lines,key=lambda x: x.primordium_label)
        gs.calculate_new_lines_positions()
        result : List[Tuple[float,float,float]] =[]

        for i,line_8 in enumerate(lines):
            if line_8.primordium_label==test:
                import pdb;pdb.set_trace()
            if line_8.primordium_label>3:
                continue
            if line_8.line_n_8 is not None and line_8.line_n_8.adult:
                continue
            distances = []
            other_lines = [yl for yl in gs.in_progres if yl.primordium_label==line_8.primordium_label-3]
            if len(other_lines)>0:
                distances.extend([self._calculate_distance_young(other_line,line_8) for other_line in other_lines])

            if i>=3:
                distances.append(self._calculate_distance_adult(lines[i-3],line_8))
            if len(distances)==0:
                continue
            
            age=0
            if line_8.line_n_8 is not None:
                age=line_8.line_n_8.age
            else:
                age=line_8.matured_age-gs.growth_coeff_development.calculate_n8(gs.simulation_time)
            
            distance = min(distances)
            result.append((gs.simulation_time,age,distance,line_8.primordium_label))


        for i,line in enumerate(lines):
            if line.primordium_label==test:
                import pdb;pdb.set_trace()
            if line.primordium_label>3:
                continue
            if line.line_n_8 is None:
                continue
            if lines[i-3].line_n_5 is None:
                continue
            if not line.line_n_8.adult:
                continue
            print(line.primordium_label)
            try:
                line_8 : global_simulation.AdultLine= [_line for _line in lines if _line.primordium_label==line.line_n_8.primordium_label][0]
            except:
                continue

            age=line.line_n_8.age

            if lines[i-3].line_n_5.primordium_label==line_8.primordium_label:
                distance=0
                result.append((gs.simulation_time,age,distance,line.primordium_label))
                continue
            else:
                start = gs.get_closest_to_center(line_8.skeleton)
                end = gs.get_closest_end_to_point(line_8.skeleton,line.position_n_8)
                import skeleton
                path = skeleton.find_path_between_nodes(line_8.skeleton,start,end,find_only_primary=True)
                points_line = [(gs.points[a],gs.points[b]) for a,b in zip(path,path[1:])]

                distance = 9999

                if lines[i-3].line_n_5.adult:
                    line_5 : global_simulation.AdultLine= [_line for _line in lines if _line.primordium_label==lines[i-3].line_n_5.primordium_label][0]
                    start = gs.get_closest_to_center(line_5.skeleton)
                    end = gs.get_closest_end_to_point(line_5.skeleton,lines[i-3].position_n_5)
                    other_path = skeleton.find_path_between_nodes(line_5.skeleton,start,end,find_only_primary=True)
                    points = [gs.points[a] for a in other_path]
                    distance = min(generate_line.calculate_distance(points,points_line))
                else:
                    distance = min(generate_line.calculate_distance(lines[i-3].line_n_5.line,points_line))

                
                result.append((gs.simulation_time,age,distance,line.primordium_label))

        return result





    # def calculate_n_8_distance_min(self) -> List[Tuple[float,float,float]]:
    #     gs : global_simulation.GlobalSimulation = self.context.data_context.global_simulation
    #     lines = sorted(gs.lines,key=lambda x: x.primordium_label)
    #     gs.calculate_new_lines_positions()
    #     result : List[Tuple[float,float,float]] =[]

    #     for i,line_8 in enumerate(lines):
    #         if line_8.primordium_label>3:
    #             continue
    #         if line_8.line_n_8 is not None and line_8.line_n_8.adult==True:
    #             continue
    #         distances = []
    #         other_lines = [yl for yl in gs.in_progres if yl.primordium_label==line_8.primordium_label-3 or yl.primordium_label==line_8.primordium_label+5]
    #         if len(other_lines)>0:
    #             distances.extend([self._calculate_distance_young(other_line,line_8) for other_line in other_lines])

    #         if i>=3:
    #             distances.append(self._calculate_distance_adult(lines[i-3],line_8))
    #         if i+5<len(lines):
    #             if lines[i+5].line_n_5 is not None and lines[i+5].line_n_5.adult != True:
    #                 distances.append(self._calculate_distance_adult(lines[i+5],line_8))

    #         if len(distances)==0:
    #             continue
            
    #         age=0
    #         if line_8.line_n_8 is not None:
    #             age=line_8.line_n_8.age
    #         else:
    #             age=line_8.matured_age-gs.growth_coeff_development.calculate_n8(gs.simulation_time)
            
    #         distance = min(distances)

                
    #         result.append((gs.simulation_time,age,distance,line_8.primordium_label))
    #     return result

    def calculate_n_8_distance(self) -> List[Tuple[float,float,float]]:
        gs : global_simulation.GlobalSimulation = self.context.data_context.global_simulation
        lines = sorted(gs.lines,key=lambda x: x.primordium_label)
        gs.calculate_new_lines_positions()
        result : List[Tuple[float,float,float]] =[]

        for line_8 in lines[:3]:
            if line_8.primordium_label>3:
                continue
            if line_8.line_n_8 is not None and line_8.line_n_8.adult==True:
                continue
            other_line = [yl for yl in gs.in_progres if yl.primordium_label==line_8.primordium_label-3]
            if len(other_line)==0:
                continue
            other_line=other_line[0]
            points=[]
            try:
                connected : global_simulation.YoungLine = max(other_line.connected_secondary,key = lambda x: x.connection)
                points.append(generate_line.point_on_path_distance_reversed(connected.line,len(connected.line)-1,gs.YOUNG_DISTANCE))
                points.append(generate_line.point_on_path_distance_reversed(other_line.line,connected.connection,gs.YOUNG_DISTANCE))
            except:
                points.append(generate_line.point_on_path_distance_reversed(other_line.line,len(other_line.line)-1,gs.YOUNG_DISTANCE+40))
            if line_8.line_n_8 is not None:
                l_8=line_8.line_n_8.line
                distance = min(generate_line.calculate_distance(points,[(a,b) for a,b in zip(l_8,l_8[1:])]))
            else:
                distance = min(sum((a-b)**2 for a,b in zip(p,line_8.position_n_8))**0.5 for p in points)

            age=0
            if line_8.line_n_8 is not None:
                age=line_8.line_n_8.age
            else:
                age=line_8.matured_age-gs.growth_coeff_development.calculate_n8(gs.simulation_time)
            result.append((gs.simulation_time,age,distance,line_8.primordium_label))

        
        for line_5,line_8 in zip(lines,lines[3:]):
            if line_8.primordium_label>3:
                continue
            if line_8.line_n_8 is not None and line_8.line_n_8.adult==True:
                continue
            distance=-1
            if line_5.line_n_5 is not None:
                if line_8.line_n_8 is not None:
                        l_8=line_8.line_n_8.line
                        distance=min(generate_line.calculate_distance(line_5.line_n_5.line,[(a,b) for a,b in zip(l_8,l_8[1:])]))
                else:
                    l_5=line_5.line_n_5.line
                    distance=generate_line.calculate_distance([line_8.position_n_8],[(a,b) for a,b in zip(l_5,l_5[1:])])[0]
            else:
                if line_8.line_n_8 is not None:
                    l_8=line_8.line_n_8.line
                    distance=generate_line.calculate_distance([line_5.position_n_5],[(a,b) for a,b in zip(l_8,l_8[1:])])[0]
                else:
                    distance = sum((a-b)**2 for a,b in zip(line_5.position_n_5,line_8.position_n_8))**0.5

            age=0
            if line_8.line_n_8 is not None:
                age=line_8.line_n_8.age
            else:
                age=line_8.matured_age-gs.growth_coeff_development.calculate_n8(gs.simulation_time)
            result.append((gs.simulation_time,age,distance,line_8.primordium_label))
        return result


            


    def color_lines(self):
        def _pos_to_index(x):
            result = 0
            xx=x
            for v in xx:
                result=result*10000+round(v,1)
            return result
        gs = self.context.data_context.global_simulation



        for point,label in self.context.data_context.labels['label'].items():
            if label == "center":
                continue
            if point not in self.context.canvas.skeleton_graph or self.context.canvas.skeleton_graph[point]==0:
                self.context.graphic_context.mark_temporary_points_add([point])
                continue
            if point:
                try:
                    x=skeleton.graph_to_lines(skeleton.find_compact_component(self.context.canvas.skeleton_graph,point))
                    
                    color_index = self.color_map.get(_pos_to_index(point),None)
                    if color_index is None:
                        print('Color not found!',label)
                        i = int(floor(float(label[:3])))
                        color_index=(int(floor(len(self.colors)+i-self.conducted_steps)))%len(self.colors)
                    else:
                        color_index = color_index%len(self.colors)
                    color=self.colors[color_index]
                    # JET for distance
                    import matplotlib.cm as cm
                    if _pos_to_index(point) not in self.color_range:
                        print('Color map not found!!!',label)
                        print(min((k for k in self.color_range),key=lambda x: abs(x-_pos_to_index(point))))
                        print(point)
                    color_jet=cm.jet(self.color_range[_pos_to_index(point)])[:3]
                    #setting color of vein???
                    #print(label,self.color_range[_pos_to_index(point)],"jet",color_jet)
                    

                    #self.context.graphic_context.set_lines_color(x,color)
                    if len(self.context.canvas.skeleton_graph[point])<3:
                       self.context.graphic_context.set_lines_color(x,(0,0,0))
                    
                except Exception as e:
                    print('Color not found! error',label)
                    self.context.graphic_context.mark_temporary_points([point])
                    #print(e)
                    raise e
                    pass

        points = [self.context.canvas._micron_coord_to_stack(gs.points[gs.get_closest_to_center(line.skeleton)]) for line in gs.lines]
        surface_points = [self.context.canvas._micron_coord_to_stack(point.position) for  point in gs.surface_points.existing_points]
        young_points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in  gs.in_progres if x.maturing==True]
        #self.context.graphic_context.mark_temporary_points(points+surface_points+young_points)
        if self.context.data_context.settings.SHOW_SURFACE_POINTS:
            self.context.graphic_context.mark_temporary_points(points+surface_points)
        else:
            self.context.graphic_context.mark_temporary_points([point for point in points if len(self.context.canvas.skeleton_graph[point])>0])
        self.context.graphic_context.mark_color_points([self.context.data_context.values['center']],[(0,0,0)])


        #draw influence ring
        if self.context.data_context.settings.DRAW_RING:
            self.context.data_context.global_simulation.distance_treshold[0]
            import shape_calculation
            
            ring_points = [shape_calculation.translate_from_cone_coordinate(gs.cone_coeffs,gs.center,gs.distance_treshold[0],(i/50)*360) for i in range(51)]
            ring_points = [self.context.canvas._micron_coord_to_stack((x[0],x[1],x[2])) for x in ring_points]
            ring_lines = [((gs.center),(b[0],b[1],b[2])) for a,b in zip(ring_points,ring_points[1:]+[ring_points[0]])]
            self.context.graphic_context.mark_temporary_paths([ring_points])
        #self.context.graphic_context.mark_temporary_points(points)

    def statistics(self):
        gs = self.context.data_context.global_simulation
        ages = sorted([line.age for line in gs.lines],key=lambda x: -x)
        younger = list(filter(lambda x: x<10,ages))
        if len(younger)>1:
            ages=younger
        diffrences = [i-j for i,j in zip(ages,ages[1:])]
        avg = sum(diffrences)/len(diffrences)
        var = sum((i-avg)**2 for i in diffrences)/len(diffrences)
        return avg,var,len([1 for i in ages if i<10]),diffrences

    #f"{c.data_context.image_path}\\step_{index:03}.png"

    def color_step(self,steps=40, time_step=None):
        if time_step is None:
            time_step=self.time_step
        if self.print_screaner:
            self.print_screaner.join()
        # if time_step>1:
        #     for _ in range(floor(time_step)):
        #         self.index += 1
        #         self.step(int(steps*time_step),time_step/floor(time_step))
        #         avg,var,number,distances=self.statistics()
        #         print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
        # else:
        #     self.index += 1
        #     self.step(int(steps*time_step),time_step)
        #     avg,var,number,distances=self.statistics()
        #     print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
        self.index += 1
        self.step(ceil(steps*time_step),time_step)
        # try:
        #     avg,var,number,distances=self.statistics()
        #     print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
        #     print(f'time passed {self.index*time_step}')
        # except ZeroDivisionError:
        #     pass

        self.color_lines()
        labels,angles=self.context.data_context.global_simulation.calculate_consecutive_angles()

        

        self.simulated_data.angles_labels.append((self.context.data_context.global_simulation.simulation_time,labels,angles))
        with open(f"{self.context.data_context.image_path}/angles","wb") as file:
            pickle.dump(self.simulated_data.angles_labels,file)
        #add data for n+5 n+8
        print("\n NEW STEP")

        if self.do_save_images and (self.screenshots_number<0 or self.index<self.screenshots_number):
            self.print_screaner = threading.Thread(target=thread_function,args=(self.index,))
            self.print_screaner.start()
        #im = ImageGrab.grab()
        #im.save(f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\predicted_steps\\screen_nr{self.conducted_steps}.png")

    def run_calculation_thread(self,steps=None, time_step=0.1,run_calculation=False):
        gs = self.context.data_context.global_simulation
        if steps == None:
            steps = int(ceil(gs.growth_rate/3*2))
            steps = 4
        #gs.simulation(steps,time_step=time_step)
        SimStep(self.context.main_frame,gs,steps,time_step,self.context.data_context.settings.SAVE_IMAGES,run_calculation=run_calculation).start()



    def color_step_post(self,steps=40, time_step=0.1):
        self.context.data_context.load_from_global_simulation()
        self.conducted_steps+=time_step
        print(self.conducted_steps)
        gs = self.context.data_context.global_simulation
        points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in gs.in_progres if x.is_left]
        surface_points = [self.context.canvas._micron_coord_to_stack(point.position) for  point in gs.surface_points.existing_points]
        young_points = [self.context.canvas._micron_coord_to_stack(x.line[0]) for x in  gs.in_progres if x.maturing==True]
        if self.context.data_context.settings.SHOW_SURFACE_POINTS:
            self.context.graphic_context.mark_temporary_points(points+surface_points+young_points)
        else:
            self.context.graphic_context.mark_temporary_points(points+young_points)

        # if time_step>1:
        #     for _ in range(floor(time_step)):
        #         self.index += 1
        #         self.step(int(steps*time_step),time_step/floor(time_step))
        #         avg,var,number,distances=self.statistics()
        #         print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
        # else:
        #     self.index += 1
        #     self.step(int(steps*time_step),time_step)
        #     avg,var,number,distances=self.statistics()
        #     print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
        try:
            avg,var,number,distances=self.statistics()
            #print(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}')
            #print(f'time passed {self.index*time_step}')
        except ZeroDivisionError:
            pass

        self.color_lines()
        labels,angles=self.context.data_context.global_simulation.calculate_consecutive_angles()


        surface_points = [self.context.canvas._micron_coord_to_stack(point.position) for  point in gs.surface_points.existing_points]
        young_points = [self.context.canvas._micron_coord_to_stack(x.line[-1]) for x in  gs.in_progres if x.maturing==True]

        if self.context.data_context.settings.SHOW_SURFACE_POINTS:
            self.context.graphic_context.mark_selection_points(surface_points+young_points)
        else:
            self.context.graphic_context.mark_selection_points(young_points)


        self.simulated_data.angles_labels.append((self.context.data_context.global_simulation.simulation_time,labels,angles))
        with open(f"{self.context.data_context.image_path}/angles","wb") as file:
            pickle.dump(self.simulated_data.angles_labels,file)
        if self.context.data_context.settings.CALCULATE_8_DISTANCES:
            self.simulated_data.distances_data.append(self.calculate_n_8_distance())
            with open(f"{self.context.data_context.image_path}/distances_8","wb") as file:
                        pickle.dump(self.simulated_data.distances_data,file)
            self.simulated_data.distances_data_adult.append(self.calculate_n_8_adult())
            with open(f"{self.context.data_context.image_path}/distances_8_min","wb") as file:
                        pickle.dump(self.simulated_data.distances_data_adult,file)

        if self.context.data_context.settings.DRAW_ARROWS:
            self.context.canvas.arrows_drawer.calculate_directions()
        if self.context.data_context.settings.DRAW_POINTS_MAP:
            self.point_map_drawer.draw_dots()

        if self.context.data_context.settings.SAVE_VECTOR_IMAGES:
            skeleton_to_vector_graphics(gs,f"{c.data_context.image_path}\\v_step_{self.index:03}")
        if self.context.data_context.settings.SAVE_IMAGES:
            if self.context.data_context.settings.FULL_HD:
                self.make_screenshot_wx(f"{c.data_context.image_path}\\step_{self.index:03}.png",screen_size=(1500,980))
                self.make_screenshot_wx(f"{c.data_context.image_path}\\step_{self.index:03}.png",screen_size=(1500,980))
            else:
                self.make_screenshot_wx(f"{c.data_context.image_path}\\step_{self.index:03}.png")
                self.make_screenshot_wx(f"{c.data_context.image_path}\\step_{self.index:03}.png")

        if self.do_save_images and (self.screenshots_number<0 or self.index<self.screenshots_number):
            self.run_calculation_thread(run_calculation=True)
        else:
            winsound.Beep(80,1000)
            winsound.Beep(160,500)
            winsound.Beep(320,500)
            #for multiple simulation
            try:
                next(self.context.data_context.new_simulation_iterator)
            except:
                winsound.Beep(320,500)
                winsound.Beep(80,1000)
                winsound.Beep(160,500)

        self.index += 1
            
        #im = ImageGrab.grab()
        #im.save(f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\predicted_steps\\screen_nr{self.conducted_steps}.png")

    def make_screenshot(self,name):
        if self.print_screaner:
            self.print_screaner.join()
        self.print_screaner = threading.Thread(target=make_single_screenshot,args=(name,))
        self.print_screaner.start()

    def test_functions(self):
        self.do_save_images=False
        if self.functions_to_test:
            if self.print_screaner:
                self.print_screaner.join()
            if self.init_function:
                self.init_function()
            f=self.functions_to_test.pop()
            f()
            self.conducted_steps=0
            time_step=30
            steps=40
            self.index += 1
            self.step(ceil(steps*time_step),time_step,sufix=f.__name__)
            self.color_lines()
            try:
                with open('images\\growths\\statistics.txt','a') as file:
                    avg,var,number,distances=self.statistics()
                    file.write(f'\nfunction_name: {f.__name__}\n')
                    file.write(f'statistics\n\t avg:{avg}\n\t var:{var}\n\t number:{number}\n\t distances:{distances}\n')
            except:
                pass
            print(f'functions left {len(self.functions_to_test)}')

            try:
                name=f.name
            except:
                name=f.__name__

            
            path=f"C:\\growths\\skeletons\\{name}.ske"
            file=open(path,'wb')
            self.context.data_context.save_data(file)
            file.close()

            self.print_screaner = threading.Thread(target=thread_function,args=(f'growths\\{name}',make_single_screenshot))
            self.print_screaner.start()



            

    def run(self):
        self.test_functions()

        #self.test_functions()


        # def line_distance(line):
        #     length=0
        #     for a,b in zip(line,line[1:]):
        #         length+=(Vector3(a)-Vector3(b)).length
        #     return length
        # first = copy.copy(self.context.data_context.global_simulation.in_progres)
        # cache = [{'line':yl,'point':yl.line[-1],'size':len(yl.line),'length':line_distance(yl.line)} for yl in first]
        # self.cache=cache
        # self.color_step()
        
        # self.growth_statistics['length_difference'].append([])
        # self.growth_statistics['absolute_distance'].append([])
        # self.growth_statistics['line_growth'].append([])

        # for _c in cache:
        #     if _c['line'].active:
        #         line=_c['line']
        #         self.growth_statistics['length_difference'][-1].append(line_distance(line.line)-_c['length'])

        #         self.growth_statistics['absolute_distance'][-1].append((Vector3(_c['point'])-Vector3(line.line[-1])).length)

        #         self.growth_statistics['line_growth'][-1].append(line_distance(line.line[max(0,_c['size']-1):]))

#Drawing from file section
    def _get_filenames(self):
        for sufix in ['max','avg']:
            for in_file, name in zip(['young']*3+['old']*3+['older']*3,['dr34','dr35','dr37','dr79','dr86','dr87','dr7', 'dr8', 'dr9']):
                yield (name,sufix,in_file)
                #yield (name,sufix)

    def generate_images(self):
        def get_best(name,sufix='max'):
            x=pickle.load(open(f".\\exploration_results\\optimalization\\dump_{name}_{sufix}",'rb'))
            x.sort(key=lambda a: a[1])
            return x[0]

        def ff(name,sufix='max',in_file=None):
            from  calculation_config import config_dict
            if in_file is None:
                v,dist=get_best(name,sufix)
            else:
                v,dist=get_best(in_file,sufix)
            print(dist)
            config=config_dict[name][0]
            l=config.labels_list[0]
            path=f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{config.number}_dr{config.file_numbers[0]}_mark.ske"

            self.context.data_context.load_data(pickle.load(open(path,'rb')))
            self.context.data_context.CIM._generate_lines(l,**v,draw=True)
        try:
            name,sufix,in_file=self.filenames_iterator.__next__()
        except StopIteration:
            return
        if self.print_screaner:
            self.print_screaner.join()
            
        ff(name,sufix,in_file)
            
        self.print_screaner = threading.Thread(target=thread_function,args=(f'.\\exploration_results\\comparison\\{name}_{sufix}_{in_file}.png',make_single_screenshot_full_path))
        self.print_screaner.start()

        