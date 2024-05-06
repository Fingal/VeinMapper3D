import wx
import calculate_line
import cone_calculation
import numpy as np
from context import Context
from auto_node import RefineWindow

class SemiAutoWindow(wx.Frame):
    def __init__(self, parent,start,direction):
        wx.Panel.__init__(self, parent, size=(700, 700))
        self.parent = parent
        self.context=Context()
        self.start=start
        self.direction=direction
        self.initial_direction=direction
        self.path=[start]
        
        self.sizer = wx.GridBagSizer(9, 5)
        self.SetSizer(self.sizer)
        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="cone angle:")
        self.sizer.Add(label,pos=(1,1))
        self.angle_field=wx.TextCtrl(self,value="180")
        self.sizer.Add(self.angle_field,pos=(1,2))

        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="cone distance:")
        self.sizer.Add(label,pos=(2,1))
        self.distance_field=wx.TextCtrl(self,value="40")
        self.sizer.Add(self.distance_field,pos=(2,2))
        
        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="step length:")
        self.sizer.Add(label,pos=(3,1))
        self.step_field=wx.TextCtrl(self,value="10")
        self.sizer.Add(self.step_field,pos=(3,2))

        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="steps number:")
        self.sizer.Add(label,pos=(4,1))
        self.step_number_field=wx.TextCtrl(self,value="10")
        self.sizer.Add(self.step_number_field,pos=(4,2))

        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="inertia:")
        self.sizer.Add(label,pos=(5,1))
        self.inertia=wx.TextCtrl(self,value="1")
        self.sizer.Add(self.inertia,pos=(5,2))

        self.step_button = wx.Button(self,label="Next step")
        self.Bind(
            wx.EVT_BUTTON, self.calculate_next_step, self.step_button
        )
        self.sizer.Add(self.step_button,pos=(7,1))

        # TODO NOT WORKING
        self.back_button = wx.Button(self,label="Back")
        self.Bind(
            wx.EVT_BUTTON, lambda x,y: 0, self.back_button
        )
        self.sizer.Add(self.back_button,pos=(7,2))

        self.save_button = wx.Button(self,label="Save")
        self.Bind(
            wx.EVT_BUTTON, self.save, self.save_button
        )
        self.sizer.Add(self.save_button,pos=(8,1))

        self.reset_button = wx.Button(self,label="reset button")
        self.Bind(
            wx.EVT_BUTTON, self.reset, self.reset_button
        )
        self.sizer.Add(self.reset_button,pos=(8,2))

        self.do_flood_fill=wx.CheckBox(self, label="use flood-fill")
        self.sizer.Add(self.do_flood_fill,pos=(6,1))

        
        # self.slider_x = RangeSlider(0,1)

        self.Show()


    def reset(self,evt):
        self.path=[self.path[0]]
        self.direction=self.initial_direction
        self.clear_temporary_line()


    def save(self,event):
        self.clear_temporary_line()
        RefineWindow(self,[self.path])
        

        
    def calculate_next_step(self,evt):
        new_path,direction=calculate_line.calculate_line(self.context.data_context.dr_stack,
            self.path[-1],direction=self.direction,
            step_size=int(self.step_field.GetValue()),
            distance=float(self.distance_field.GetValue()),
            angle=float(self.angle_field.GetValue()),
            inertia=float(self.inertia.GetValue()),
            steps_number=int(self.step_number_field.GetValue()),
            flood_fill=self.do_flood_fill.GetValue())
        self.path.extend(new_path[1:])
        self.direction=direction

        self.context.graphic_context.mark_temporary_paths([self.path])

    def _reload_iter(self):
        self.point_iter=calculate_line.iter_calculate_line(self.canvas.points,
                self.start,direction=self.direction,
                step_size=int(self.step_field.GetValue()),
                distance=float(self.distance_field.GetValue()),
                angle=float(self.angle_field.GetValue()),
                inertia=float(self.inertia.GetValue()))


    def clear_temporary_line(self):
        self.context.graphic_context.mark_temporary_paths([])
