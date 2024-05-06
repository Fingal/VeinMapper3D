from typing import List
from numpy.lib.function_base import angle
import pyrr
from pyrr.objects import vector3
from pyrr.objects.base import BaseVector3
import wx
import wx.lib.scrolledpanel
import copy
import numpy as np
import math
from math import floor, ceil
from importlib import reload
import test_functions as tf
from types import MethodType
from stack_io import *
import pickle
import re
from point_display import *
from auto_node import AutoWindow
from semi_auto_node import SemiAutoWindow
from simulation_controller import SimulationController
import calculate_line
import cone_calculation
from time import time
from context import Context
from measurement_data import *
import skeleton


from stack_io import save_stack


#TODO distance computation 

class SkeletonDistances(wx.Panel):
    def __init__(self, parent, canvas):
        wx.Panel.__init__(self, parent)
        self.canvas = canvas
        self.show_points = False
        #self.SetupScrolling()
        self.sizer = wx.GridBagSizer(2, 0)
        self.SetSizer(self.sizer)
        self._grid_cache=[]
        self.context=Context()
        self.context.skeleton_distances=self
        self.simulation_controller=SimulationController()

        self.concentration_exporter : ConcentrationExporter = ConcentrationExporter()
        self.angle_exporter : AngleExporter = AngleExporter()

        self.distance_exporter : LineLineDistanceExporter = LineLineDistanceExporter()

        self.exporters = [self.concentration_exporter,self.angle_exporter,self.distance_exporter]

        self.grid = wx.grid.Grid(self,size=(150,200))

        #self.values = {'point':[],'line':[],'center':None, 'label':[]}
        self.hidden = set()
        self.point_display=PointDisplay(self)
        #self.grid_window=AllPointsWindow(self)
        self.grid.EnableEditing(True)
        self.grid.CreateGrid(10, 3)
        self.grid.SetColLabelValue(0, "Label")
        self.grid.SetColLabelValue(1, "Type")
        self.grid.SetColLabelValue(2, "Positions")
        self.grid.SetColSize(0, 40)
        self.grid.SetColSize(1, 60)
        self.grid.SetColSize(2, 120)
        self.grid.SetRowLabelSize(25)
        self.grid.GetGridWindow().Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)
        self.grid.GetGridWindow().Bind(wx.EVT_LEFT_UP, self.show_marked)
        self.grid.SetSelectionMode(1)
        self.grid.Bind(wx.grid.EVT_GRID_CELL_CHANGING, self.on_edit)
        self.sizer.Add(
            self.grid, pos=(0, 0), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=(10, 25)
        )
        self.button_panel=wx.lib.scrolledpanel.ScrolledPanel(self)
        #self.button_panel.SetupScrolling(scroll_x = False)
        self.sizer.Add(self.button_panel, pos=(10,0), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=(16.99,26))

        self.button_sizer = wx.GridBagSizer(2, 0)
        self.button_panel.SetSizer(self.button_sizer)

        self.show_marked_button = self.add_button(
            "Show marked",
            (0, 0),
            (1, 10),
            self.show_marked_event,
            button_type=wx.ToggleButton,
        )
        self.set_center_button = self.add_button(
            #"Set center", (1, 0), (1, 10), self.set_center
            "Slider point", (1, 0), (1, 10), self.add_slider_point
        )
        
        self.add_skeleton_point_button = self.add_button(
            "Add point", (2, 0), (1, 10), self.add_skeleton_point
        )
        self.auto_node_button = self.add_button(
            "Auto nodes", (3, 0), (1, 10), self.auto_node
        )
        
        
        self.load = self.add_button(
            "Load Skeleton", (6, 0), (1, 10), self.load_skeleton_event
        )
        self.save = self.add_button(
            "Save Skeleton", (7, 0), (1, 10), self.save_skeleton
        )
        self.sort_button = self.add_button(
            "Convert to tiff", (8, 0), (1, 10), self.convert_to_tiff
        )
        self.sort_button = self.add_button(
            "Sort", (9, 0), (1, 10), self.sort
        )

        #TODO move to graphic context
        def f(e):
            self.canvas.draw_text=(self.canvas.draw_text+1)%2
            self.canvas.Refresh()
        self.label_button = self.add_button("show label", (10, 0), (1, 10), f)

        self.debug_button = self.add_button("debug", (11, 0), (1, 10), self.debug)


        self.compute = self.add_button(
            "Join points", (0, 10), (1, 10), self.connect_points
        )

        self.line_point = self.add_button(
            "Join point-line", (1, 10), (1, 10), self.connect_point_line
        )
        

        self.add_point_button = self.add_button(
            "Point on line", (2, 10), (1, 10), self.add_point_on_line
        )
        self.auto_direction_button = self.add_button(
            "Semi-auto node", (3, 10), (1, 10), self.semi_auto_node
        )

        self.line_point = self.add_button(
            "Project point to line", (6, 10), (1, 10), self.project_point_line
        )

        self.line_point = self.add_button(
            "Remove line", (7, 10), (1, 10), self.remove_line
        )


        initial_label = wx.StaticText(
            self.button_panel, style=wx.ALIGN_CENTRE_HORIZONTAL, label="point label for angles:"
        )
        self.button_sizer.Add(
            initial_label,
            pos=(8, 10),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            span=(1, 10),
        )

        self.initial_label = wx.TextCtrl(self.button_panel)
        self.button_sizer.Add(
            self.initial_label,
            pos=(9, 11),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            span=(1, 10),
        )

        self.height_distance = wx.CheckBox(self.button_panel, label="Height distance")
        self.button_sizer.Add(
            self.height_distance,
            pos=(10, 12),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            span=(1, 14),
        )
        self.compute = self.add_button(
            "Compute distances", (11, 10), (1, 10), self.compute_distances_event
        )


        self.value_labels = [wx.StaticText(self.button_panel),wx.StaticText(self.button_panel),wx.StaticText(self.button_panel)]
        #for max values on surface
        self.value_fields = [wx.TextCtrl(self.button_panel),wx.TextCtrl(self.button_panel),wx.TextCtrl(self.button_panel)]
        for i,(label,field) in enumerate(zip(self.value_labels,self.value_fields)):
            size=5
            self.button_sizer.Add(
                label,
                pos=(13, i*size),
                flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
                span=(1, size),
            )
            self.button_sizer.Add(
                field,
                pos=(14, i*size),
                flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
                span=(1, size),
            )

        #for average values on surface
        self.value_fields_average = [wx.TextCtrl(self.button_panel),wx.TextCtrl(self.button_panel),wx.TextCtrl(self.button_panel)]
        for i,field in enumerate(self.value_fields_average):
            size=5
            self.button_sizer.Add(
                field,
                pos=(15, i*size),
                flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
                span=(1, size),
            )
        
        self.add_skeleton_point_button = self.add_button(
            "Compute distance lines", (16, 0), (1, 10), self.compute_distance_line_event
        )
        self.clear_button = self.add_button(
            "Clear", (16, 10), (1, 10), self.clear_table
        )
        self.compute_distance_button = self.add_button(
            "Compute distance", (17, 0), (1, 10), self.compute_single_distance_event
        )
        self.compute_distance_button = self.add_button(
            "Compute angles", (17, 10), (1, 10), self.compute_angles_event
        )
        self.radius_field = wx.StaticText(self.button_panel,label='radius')
        self.button_sizer.Add(
            self.radius_field,
            pos=(18, 0),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            span=(1, 5),
        )
        self.radius_field = wx.TextCtrl(self.button_panel)
        self.button_sizer.Add(
            self.radius_field,
            pos=(18, 5),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            span=(1, 2),
        )
        self.compute_distance_button = self.add_button(
            "Concentration sur.", (18, 10), (1, 10), self.concentration_surface_chosen
        )


        self.export_calculation_button = self.add_button(
            "Export measurement", (20, 10), (1, 10), self.export_measurement
        )

        self.compute_distance_button = self.add_button(
            "point to surface", (19, 0), (1, 10), self.add_surface_point
        )
        # self.compute_distance_button = self.add_button(
        #     "Calculate", (19, 0), (1, 10), lambda x: self.context.data_context._test(values=
        #             self.context.data_context.test_coefs
        #                ))

        # self.compute_distance_button = self.add_button(
        #     "Calculate gs", (19, 0), (1, 10), lambda x: self.simulation_controller.run())

        #values
        def _optimalization():
            results=[]
            with open('results_walk_2_max','rb') as file:
                walk_results=pickle.load(file)
            start_time = time()
            for i,(values,_) in enumerate(walk_results[116:1000]):
                print(f'full time {time()-start_time}\t\t {i+1+116}/1000 in progres')
                result=self.context.data_context.CIM._test(values=values)
                results.append(result)
            with open('results_optimalisation','wb') as file:
                pickle.dump(results,file)
        # self.compute_distance_button = self.add_button(
        #     "Calculate", (19, 0), (1, 10), lambda x: _optimalization())
        self.compute_distance_button = self.add_button(
    "Calculate", (20, 0), (1, 10), lambda x: self.simulation_controller.color_step(6,0.1))




    def on_right_click(self, event):
        """"""
        points,lines = self.get_marked_lines()
        if points or lines:
            self.context.data_context.hidden.update(points)
            self.context.data_context.hidden.update(lines)
        else:
            x, y = self.grid.CalcUnscrolledPosition(event.GetX(), event.GetY())
            row, col = self.grid.XYToCell(x, y)
            # print(row, col)
            if row==len(self._grid_cache):
                self.context.data_context.values['center']=None
            assert row>len(self._grid_cache),'No marked'
        self.grid.ClearSelection()
            
        self.reload_grid()

    def add_button(self, label, pos, span, funct, button_type=wx.Button):
        button = button_type(self.button_panel, label=label)
        self.button_sizer.Add(button, pos=pos, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=span)
        if button_type == wx.Button:
            self.Bind(wx.EVT_BUTTON, funct, button)
        if button_type == wx.ToggleButton:
            self.Bind(wx.EVT_TOGGLEBUTTON, funct, button)

        return button

    def add_line(self, start, end, label=""):
        assert False,"deprecated"
        self.context.data_context.add_lines([(start,end)])

    def add_point(self, point, label=""):
        assert False,"deprecated"
        # if point in self.hidden:
        #     self.hidden.remove(point)
        # elif point not in [p for label,p in self.values['point']]:
        #     self.values["point"].append((label,point))
        # self.canvas.add_point(point)
        # self.reload_grid()

    def _add_to_grid(self,i,label,typ,positions_string):
        self.grid.SetCellValue(i, 0, label)
        self.grid.SetCellValue(i, 1, typ)
        self.grid.SetCellValue(i, 2, positions_string)

    def sort(self,event):
        self.context.data_context.sort()

    def reload_grid(self,reload_window=True):
        new_row_number = (len(self.context.data_context.values['point'])+
                                  len(self.context.data_context.values['line'])+1-len(self.context.data_context.hidden)+15)
        if self.grid.GetNumberRows()>new_row_number:
            self.grid.DeleteRows(new_row_number, self.grid.GetNumberRows())
        else:
            self.grid.AppendRows(new_row_number-self.grid.GetNumberRows())
        
        index=0
        self._grid_cache=[]
        for positions in self.context.data_context.values['point']:
            if positions not in self.context.data_context.hidden:
                positions_string = "{:},{:},{:}".format(*positions)
                label=self.context.data_context.labels['point'][positions]
                self._add_to_grid(index,label,'point',positions_string)
                index+=1
                self._grid_cache.append(positions)
        for positions in self.context.data_context.values['line']:
            if positions not in self.context.data_context.hidden:
                positions_string = ""
                for position in positions:
                    positions_string = positions_string + "({:},{:},{:}) ".format(
                        *position
                    )
                label=self.context.data_context.labels['line'][positions]
                self._add_to_grid(index,label,'line',positions_string)
                index+=1
                self._grid_cache.append(positions)
        if self.context.data_context.values['center']:
            positions = self.context.data_context.values['center']
            positions_string = "{:},{:},{:}".format(*positions)
            label='CENTER'
            self._add_to_grid(index,label,'center',positions_string)
        for i in range(index,new_row_number):
            self._add_to_grid(i,'','','')

        #self.show_marked('')

    def on_edit(self, evnt):
        # import pdb; pdb.set_trace()
        if evnt.GetCol() == 0 and evnt.GetRow() <= len(self._grid_cache):
            edited = self._grid_cache[evnt.GetRow()]
            type='line'
            if len(edited)==3:
                type='point'
            if edited in self.context.data_context.values['point']:
                self.context.data_context.labels['point'][edited]=evnt.GetString() 
            #self.grid_window.value_update(*result)
            self.context.data_context.reload_all()
        else:
            evnt.Veto()

    
    def get_marked_lines(self, take_from_point_display=True):
        selected = [self._grid_cache[i] for i in self.grid.GetSelectedRows() if i < len(self._grid_cache)]
        selected_lines = []
        selected_points = []
        for i in selected:
            print(i,len(i))
            if len(i)==2:
                selected_lines.append(i)
            elif len(i)==3:
                selected_points.append(i)
                
        if self._is_center_selected():
            selected_points.append(self.context.data_context.values['center'])
        if take_from_point_display:
            s,type=self.point_display.get_selected()
            if type == 'line':
                selected_lines.extend(s)
            elif type == 'point' or type == 'label':
                selected_points.extend(s)
        return selected_points, selected_lines


    def _is_center_selected(self):
        return self.context.data_context.values['center']!= None and len(self._grid_cache) in self.grid.GetSelectedRows()

    def show_marked(self, evnt):
        if self.show_points:
            selected_points, selected_lines = self.get_marked_lines()
            self.context.graphic_context.mark_selection_points(selected_points)
            self.context.graphic_context.mark_selection_lines(selected_lines)
            # self.canvas.mark_points([])
            # self.canvas.mark_lines([])

    def show_marked_event(self, evnt):
        self.show_points = evnt.IsChecked()
        if evnt.IsChecked() and self.grid.GetSelectedRows():
            self.show_marked("")
        else:
            self.context.graphic_context.mark_selection_points([])
            self.context.graphic_context.mark_selection_lines([])
            # self.canvas.mark_points([])
            # self.canvas.mark_lines([])

    def on_click(self, evnt):
        if self.show_points:
            self.show_marked("")

    def _add_skeleton_point(self,point):
        self.context.data_context.show_data(set([point]))
        self.reload_grid()
        for i in range(self.grid.GetNumberRows()):
            if f"{point[0]},{point[1]},{point[2]}"==self.grid.GetCellValue(i,2):
                self.grid.MakeCellVisible(i,0)
                self.grid.SelectRow(i)
                continue

    def add_skeleton_point(self,evnt):
        #assert False, 'not implemented'
        #TODO no idea
        self.canvas.set_mode("find_any_point",self._add_skeleton_point)

        
    
    def connect_points(self, evnt):
        points = tuple(self.get_points(2))
        if points:
            self.context.data_context.add_lines([points])
        for i in range(self.grid.GetNumberRows()):
            if f"{points[0]} {points[1]} " in self.grid.GetCellValue(i,2) or f"{points[1]} {points[0]} " in self.grid.GetCellValue(i,2):
                self.grid.MakeCellVisible(i,0)
                self.grid.SelectRow(i)
                continue



            
    def add_point_on_line(self, evnt):
        line = tuple(self.get_lines(1))
        if line:
            line=line[0]
            def get_connection(coef):
                junction=tuple(start*(1-coef)+end*coef for start,end in zip(*line))
                self.context.graphic_context.mark_temporary_points([junction])
            
            def result(coef):
                junction=tuple(start*(1-coef)+end*coef for start,end in zip(*line))
                junction=self.context.data_context.add_junction(line,junction)
                self.context.graphic_context.mark_temporary_points([])
                self.context.data_context.add_points([junction])
            SliderFrame(self,get_connection,result)
            # self.canvas.connect_(point[0],line[0])
            # self.add_line(*point)            

    def connect_point_line(self, evnt):
        point = tuple(self.get_points(1))
        line = tuple(self.get_lines(1))
        if point and line:
            point=point[0]
            line=line[0]
            def get_connection(coef):
                junction=tuple(start*(1-coef)+end*coef for start,end in zip(*line))
                self.context.graphic_context.mark_temporary_points([junction])
            
            def result(coef):
                junction=tuple(start*(1-coef)+end*coef for start,end in zip(*line))
                junction=self.context.data_context.add_junction(line,junction)
                self.context.data_context.add_lines([(point,junction)])
                self.context.graphic_context.mark_temporary_points([])

            SliderFrame(self,get_connection,result)
            # self.canvas.connect_(point[0],line[0])
            # self.add_line(*point)

    def project_point_line(self, evnt):
        from calculation_skeleton_features import calculate_all
        calculate_all()
        assert False,"not used"
        point = tuple(self.get_points(1))
        line = tuple(self.get_lines(1))
        if point and line:
            point=point[0]
            line=line[0]
            self.values['point']=[(label,p) for (label,p) in self.values['point'] if p!=point]
            new_point = self.canvas.project_point_line(point,line)
            new_lines=[(label,line) for label,line in self.values['line'] if point not in line] + [(label,(new_point,line[1])) for label,line in self.values['line'] if point == line[0]] + [(label,(line[0],new_point)) for label,line in self.values['line'] if point == line[1]]
            self.values['line']=new_lines
            self.reload_grid()
            # if new_point:
            #     self.values['point'].append(new_point)

    def remove_line(self,event):
        try:
            line = tuple(self.get_lines(1))[0]
            if line:
                self.context.data_context.remove_line(*line)
        except:
            pass
        start,end = tuple(self.get_points(2))
        self.remove_path_pass_label(start,end)
        # if len(self.canvas.skeleton_graph[start])==1:
        #     end,start=start,end
        # self.context.canvas.remove_line(start,end)
        # self.context.data_context.labels["point"][start]=self.context.data_context.labels["point"][end]
        # if len(self.canvas.skeleton_graph[end])==0:
        #     print("removing end",end, self.canvas.skeleton_graph[end])
        #     self.context.data_context.remove([end],"point")
        #     if end in self.context.data_context.values["label"]:
        #         self.context.data_context.remove_label(end)
        #         self.context.data_context.add_label(start)
    
    def remove_path_pass_label(self,start,end):
        if len(self.canvas.skeleton_graph[start])==1:
            end,start=start,end
        self.context.canvas.remove_line(start,end)
        self.context.data_context.labels["point"][start]=self.context.data_context.labels["point"][end]
        if len(self.canvas.skeleton_graph[end])==0:
            print("removing end",end, self.canvas.skeleton_graph[end])
            self.context.data_context.remove([end],"point")
            if end in self.context.data_context.values["label"]:
                self.context.data_context.remove_label(end)
                self.context.data_context.add_label(start)




        
        

    def compute_distances_event(self, event):
        saveFileDialog = wx.FileDialog(
            self, "Save", "", "", "txt files *.txt | *.txt", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()
        saveFileDialog.Destroy()
        points=self.context.data_context.values['label']
        #lines=self.context.data_context.values['line']
        values=self.context.data_context.values

        with open(path, "w") as file:
            file.write('points_index={')
            for i,(pos) in enumerate(points):
                label=self.context.data_context.labels['label'][pos]
                file.write(f'"{label}":{i},')
            file.write('}')

            # file.write('lines_index={')
            # for i,(label,_,) in enumerate(lines):
            #     file.write(f'{label}:{i},')
            # file.write('}\n')

            if center and self.initial_label.GetValue().strip():
                center = values['center']
                center_pos = center[1]
                first_point = [
                    x for x in values['label'] if self.context.labels['ponint'][x] == self.initial_label.GetValue().strip()
                ][0]
                first_point_pos = first_point
                direction = np.array([first_point_pos[0], first_point_pos[2]]) - np.array(
                    [center_pos[0], center_pos[1]]
                )
                direction = direction / np.linalg.norm(direction)
                
                #file.write("Distaces and angles between center and points:\n")
                file.write("Distace_center=[")
                for positions in points:
                    file.write(
                        f"{round(self.canvas.point_distance_L2(center_pos,positions),3)}, "
                    )
                file.write(']\n')

                file.write("angle=[")
                for positions in points:
                    new_direction = np.array([positions[0], positions[2]]) - np.array(
                        [center_pos[0], center_pos[2]]
                    )
                    new_direction = new_direction / np.linalg.norm(new_direction)
                    angle = np.arccos(np.dot(direction, new_direction)) / np.pi * 180
                    file.write(f"{round(angle,1)},")
                file.write(']\n')

            if self.height_distance.GetValue():
                self.data_context.compute_height_distance()
                file.write('point_height_distance=[')
                for i, (label, positions) in enumerate(points):
                     file.write(f"{round(self.canvas.height_distance(positions),3)},")
                file.write(']\n')

                # file.write('line_height_distance=[')
                # for i, (label, positions) in enumerate(lines):
                #     min_distance, max_distance = self.canvas.line_height_distance(
                #         *positions
                #     )
                #     file.write(f"({round(min_distance,3)},{round(max_distance,3)}),")
                # file.write(']\n')
            all_straight_distances=[]
            all_graph_distances=[]

    def debug(self, evt):
        # p,_=self.get_marked_lines()
        # point=p[0]
        # print(self.canvas.height_distance(point))
        import pdb

        pdb.set_trace()
        # coord=self.canvas.stack_lookup_coordinates.get(point,None)
        # self.canvas.heightmap[coord[0],coord[1]]
        # abs(self.canvas.heightmap[coord[0],coord[1]]-coord[2])*self.canvas.scaling_ratio[2]/self.canvas.shape[2]*self.canvas.unit_to_microns_coef

    def set_center(self, event):
        point = self.get_points(1)[0]
        self.context.data_context.set_center(point)
    def add_slider_point(self,event):
        def _(point):
            self.context.data_context.add_points([point])
            self.context.graphic_context.mark_temporary_points([])
        CenterChooser(self.context,self,_)
    
    def _get_item(self, typ, number=0):
        if typ=='line':
            selected=[self._grid_cache[x] for x in  self.grid.GetSelectedRows() if len(self._grid_cache[x])==2]

        else:
            selected=[self._grid_cache[x] for x in  self.grid.GetSelectedRows() if len(self._grid_cache[x])==3]

        print(selected)
        if number:
            if len(selected) == 0:
                if typ=='line':
                    [x for x in  self._grid_cache if len(x)==2][:-number]
                else:
                    [x for x in  self._grid_cache if len(x)==1][:-number]
            elif len(selected) != number:
                return
        return selected    
    
    def get_points(self, number=0):
        return self._get_item('point', number)

    def get_lines(self, number=0):
        return self._get_item('line', number)


    def save_skeleton(self, event):
        saveFileDialog = wx.FileDialog(
            self,
            "Save",
            "",
            "",
            "Skeleton files (*.ske)|*.ske|Any type (*.*)|*.*",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()
        saveFileDialog.Destroy()
        if "ske" not in path[-4:]:
            path = path + ".ske"
        file=open(path,'wb')
        self.context.data_context.save_data(file)
        file.close()

    def load_skeleton_event(self, event):
        openFileDialog = wx.FileDialog(
            self,
            "Open",
            "",
            "",
            "Skeleton files(*.ske)| *.ske|Any type (*.*)|*.*",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = openFileDialog.GetPath()
        self.load_skeleton(path)

    def load_skeleton(self,path):
        file=open(path,'rb')
        save=pickle.load(file)
        self.context.data_context.load_data(save)
        #self.grid_window.clear()
        self.reload_grid()

    def value_update(self,label,point,type):
        assert False,"deprecated"
        if type=='center':
            self.values['center']=(label,point)
        else:
            index=-1
            for i,(label,p) in enumerate(self.values[type]):
                if p==point:
                    index=i
                    break
            if index>=0:
                self.values[type][index]=(label,point)

    # def auto_node(self,evnt):
    #     start=self.get_points(1)[0]
    #     l=self.get_lines(1)[0]
    #     print(l)
    #     if start==l[0]:
    #         direction=(np.array(l[0])-np.array(l[1]))
    #     elif start==l[1]:
    #         direction=(np.array(l[1])-np.array(l[0]))
    #     else:
    #         print('xxxx')
    #         return
        
    #     positions=calculate_line.calculate_line(self.canvas.points,start,direction=direction,step_size=10,step_number=20,
    #                                             distance=20,angle=np.pi/1.5)
    #     for pos in positions:
    #         self.add_point(pos,"")
    #         # TODO self.added_points.append(pos)
    #         # TODO self._added_point(pos)
    #     for line in zip(positions,positions[1:]):
    #         self.add_line(*line,"")
    #     self.canvas.mark_stack_points([])
    
    def auto_node(self,_):
        points=self.get_points()
        AutoWindow(self,points)

    def semi_auto_node(self,_):
        start=self.get_points(1)[0]
        l=self.get_lines(1)[0]
        print(l)
        if start==l[0]:
            direction=(np.array(l[0])-np.array(l[1]))
        elif start==l[1]:
            direction=(np.array(l[1])-np.array(l[0]))
        else:
            print('xxxx')
            return
        SemiAutoWindow(self,start,direction)

        
    def clear_table(self,evnt):
        self.context.data_context.hide_all()

    def compute_single_distance_event(self,evnt):
        
        a,b = self.get_points(2)
        x,y,z,_all = self.canvas.point_distance_coords_L2(a,b)
        *_,s = self.canvas.point_distance_skeleton(a,b)
        self.value_labels[0].SetLabel("2d dist")
        self.value_labels[1].SetLabel("3d dist")
        self.value_labels[2].SetLabel("sk dist")

        self.value_fields[0].SetValue(f"{round(float((x**2+z**2)**0.5),1)}")
        self.value_fields[1].SetValue(f"{round(float(_all),1)}")
        self.value_fields[2].SetValue(f"{round(float(s),1)}")

    def convert_to_tiff(self,event):
        saveFileDialog = wx.FileDialog(
            self, "Save", "", "", "tiff files *.tif | *.tif", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()
        saveFileDialog.Destroy()
        try:
            save_stack(self.context.data_context.create_skeleton_stack_smooth(float(self.radius_field.GetValue())),name=path)
        except:
            for r in (1,2,3):
                p=path[:-4]+f"r={r}.tif"
                print(f"path {p}")
                save_stack(self.context.data_context.create_skeleton_stack_smooth(r),name=p)
        #save_stack(self.context.data_context.create_skeleton_stack(),name=path)


    def compute_distance_line_event(self,evnt):
        points = self.get_points(4)
        distance,first_line,second_line = self.canvas.line_line_distance_points(points)
        self.value_labels[0].SetLabel("line dist")
        self.value_fields[0].SetValue(f"{distance}")
        
        first_name = self.context.data_context.get_closest_label(first_line[0])
        second_name = self.context.data_context.get_closest_label(second_line[1])
        self.distance_exporter.add(first_name,second_name,distance)

        
    def compute_angles_event(self,evnt):
        _points = self.get_points(2)
        #_stack_coord_to_micron
        
        center = self.canvas._stack_coord_to_opengl(self.context.data_context.get_point_from_label('c'))
        points= [self.canvas._stack_coord_to_opengl(x) for x in _points]

        center = vector3.Vector3((center[0],0,center[2]))
        print("points",points)
        a = (vector3.Vector3((points[0][0],0,points[0][2]))-center).normalized
        b = (vector3.Vector3((points[1][0],0,points[1][2]))-center).normalized

        
        angle=(math.acos(a.dot(b))/math.pi*180)
        self.value_labels[0].SetLabel(f"angle")
        self.value_fields[0].SetLabel(f"{round(angle,2)}")

        first_name = self.context.data_context.get_closest_label(_points[0])
        second_name = self.context.data_context.get_closest_label(_points[1])

        self.angle_exporter.add(first_name,second_name,angle)
        #assert False, 'not implemented'

    def _concentration_surface(self,r):
        b = self.get_points(1)[0]
        self.value_labels[0].SetLabel(f"r{r} d7")
        self.value_labels[1].SetLabel(f"r{r} d15")
        self.value_labels[2].SetLabel(f"r{r} depth")


        d_7,d_7_avg = self.context.data_context.calculate_approx_survace_with_average(b,r=r,depth=7)
        d_15,d_15_avg = self.context.data_context.calculate_approx_survace_with_average(b,r=r,depth=15)
        h,h_avg = self.context.data_context.calculate_min_concentration_height(b,r=r)

        self.value_fields[0].SetValue(f"{round(d_7,1)}")
        self.value_fields[1].SetValue(f"{round(d_15,1)}")
        self.value_fields[2].SetValue(f"{h}")
        self.value_fields_average[0].SetValue(f"{round(d_7_avg,1)}")
        self.value_fields_average[1].SetValue(f"{round(d_15_avg,1)}")
        self.value_fields_average[2].SetValue(f"{round(h_avg,1)}")

        
        name = self.context.data_context.get_closest_label(b)
        self.concentration_exporter.add(name,d_7_avg,d_15_avg,h_avg,d_7,d_15,h,r)

    def export_measurement(self,evnt):
        saveFileDialog = wx.FileDialog(
            self, "Save", "", "", "tsv files *.tsv | *.tsv", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )
        
        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()

        with open(path,"w") as file:
            for exporter in self.exporters:
                exporter.export(file)
                file.write("\n")


    def concentration_surface_r3(self,evnt):
        self._concentration_surface(3)

    def concentration_surface_r1(self,evnt):
        self._concentration_surface(1)

    def concentration_surface_chosen(self,evnt):
        radius = int(self.radius_field.GetValue())
        self._concentration_surface(radius)
        
    def add_surface_point(self,evnt):
        old_point = self.get_points(1)[0]

        x=self.context.data_context.compute_height_distance(old_point)
        new_point= (old_point[0],old_point[1],x)
        name='s'
        if self.context.data_context.labels['point'][old_point]:
            name=self.context.data_context.labels['point'][old_point]+name
        self.context.data_context.add_points([new_point],label=name)
        self.context.data_context.add_label(new_point,name)
        

class CenterChooser(wx.Frame):
    def __init__(self, context,parent,funct):
        wx.Panel.__init__(self, parent, size=(550, 300))
        self.context = context
        self.funct = funct
        self.canvas=self.context.canvas

        self.sizer = wx.GridBagSizer(9, 5)
        # self.slider_x = RangeSlider(0,1)
        self.sliders = {}
        self.add_sliders("X", 0)
        self.add_sliders("Y", 1)
        self.add_sliders("Z", 2)
        self.button = wx.Button(self, label="Set")
        self.sizer.Add(self.button, pos=(7, 14), span=(0, 9), flag=wx.EXPAND)
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.button_event, self.button)
        self.context.graphic_context.mark_temporary_points([(0.0, 0.0, 0.0)])
        self.Show()

    def add_sliders(self, axis, pos):
        label = wx.StaticText(self, style=wx.ALIGN_CENTRE_HORIZONTAL, label=axis + " ")
        self.sliders["label_" + axis] = label
        self.sizer.Add(
            label, pos=(1 + 2 * pos, 0), span=(1, 37), flag=wx.EXPAND, border=3
        )

        slider = wx.Slider(self, minValue=0, maxValue=1000, name=(axis + " pos"))
        slider.SetValue(500)
        self.sliders["slider_" + axis] = slider
        self.sizer.Add(
            slider, pos=(2 + 2 * pos, 0), span=(1, 37), flag=wx.EXPAND, border=3
        )

        self.Bind(wx.EVT_SCROLL, self.on_slider, slider)

    def get_point(self):
        return (
            self.sliders["slider_X"].GetValue(),
            self.sliders["slider_Y"].GetValue(),
            #self.sliders["slider_Z"].GetValue() / 1000 - 0.5,
            100
        )

    def on_slider(self, evnt):
        self.context.graphic_context.mark_temporary_points([self.get_point()])

    #TODO turn to marked point
    def button_event(self, evnt):
        self.funct(self.get_point())
        self.Close()


class SliderFrame(wx.Frame):
    def __init__(self, parent,update_funct,result_funct):
        wx.Panel.__init__(self, parent, size=(550, 300))
        self.parent = parent
        self.update_funct=update_funct
        self.result_funct=result_funct

        self.sizer = wx.GridBagSizer(9, 5)
        # self.slider_x = RangeSlider(0,1)
        label = wx.StaticText(self, style=wx.ALIGN_CENTRE_HORIZONTAL, label="Choose junction position")
        self.sizer.Add(
            label, pos=(1 + 0, 0), span=(1, 37), flag=wx.EXPAND, border=3
        )
        
        self.slider = wx.Slider(self, minValue=0, maxValue=1000, name=("pos"))
        self.sizer.Add(
            self.slider, pos=(2 + 0, 0), span=(1, 37), flag=wx.EXPAND, border=3
        )

        self.button = wx.Button(self, label="Choose")
        self.sizer.Add(self.button, pos=(3, 14), span=(0, 9), flag=wx.EXPAND)
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.button_event, self.button)
        self.Bind(wx.EVT_SCROLL, self.on_slider, self.slider)

        self.update_funct(self.slider.GetValue())
        self.Show()

    def on_slider(self,evnt):
        self.update_funct(self.slider.GetValue()/1000)


    def button_event(self, evnt):
        self.result_funct(self.slider.GetValue()/1000)
        self.Close()



