import wx
import calculate_line
import cone_calculation
import numpy as np
from context import Context
import copy

class AutoWindow(wx.Frame):
    def __init__(self, parent,positions):
        wx.Panel.__init__(self, parent, size=(300, 300))
        self.parent = parent
        self.context=Context()
        self.paths=[]

        self.directions=[]
        for point in positions:
            direction = cone_calculation.calculate_max_direction(self.context.data_context.dr_stack,point)
            print(direction)
            if direction[2]>0:
                direction=-np.array(direction)
            else:
                direction=np.array(direction)
                
            self.paths.append([point])
            self.directions.append(direction)

            if point[2]<0.8*self.context.data_context.dr_stack.shape[2]:
                self.paths.append([point])
                self.directions.append(-direction)
        self.initial_directions=copy.deepcopy(self.directions)
        print('directions',self.directions)


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
        self.sizer.Add(self.step_button,pos=(6,1))

        # TODO NOT WORKING
        self.back_button = wx.Button(self,label="Back")
        self.Bind(
            wx.EVT_BUTTON, lambda x,y: 0, self.back_button
        )
        self.sizer.Add(self.back_button,pos=(6,2))

        self.save_button = wx.Button(self,label="Refine")
        self.Bind(
            wx.EVT_BUTTON, self.save, self.save_button
        )
        self.sizer.Add(self.save_button,pos=(7,1))

        self.reset_button = wx.Button(self,label="reset button")
        self.Bind(
            wx.EVT_BUTTON, self.reset, self.reset_button
        )
        self.sizer.Add(self.reset_button,pos=(7,2))
        
        # self.slider_x = RangeSlider(0,1)

        self.Show()

    def reset(self,evt):
        self.paths=[[p[0]] for p in self.paths]
        self.directions=self.initial_directions
        print('directions',self.directions)
        self.clear_temporary_line()

    def save(self,event):
        RefineWindow(self,self.paths)
        # self.clear_temporary_line()
        # for path in self.paths:
        #     p=list(zip(path,path[1:]))
        #     self.context.data_context.add_lines(p)
        

    def calculate_next_step(self,evt):
        for i,path in enumerate(self.paths):
            if type(self.directions[i]) != type(None):
                new_path,direction=calculate_line.calculate_line(self.context.data_context.dr_stack,
                    path[-1],direction=self.directions[i],
                    step_size=int(self.step_field.GetValue()),
                    distance=float(self.distance_field.GetValue()),
                    angle=float(self.angle_field.GetValue()),
                    inertia=float(self.inertia.GetValue()),
                    steps_number=int(self.step_number_field.GetValue()))
                path.extend(new_path[1:])
                if path[-1][2]<0.8*self.context.data_context.dr_stack.shape[2]:
                    self.directions[i]=direction
                else:
                    self.directions[i]=None
        self.context.graphic_context.mark_temporary_paths(self.paths)

    
    def clear_temporary_line(self):
        self.context.graphic_context.mark_temporary_paths([])

def normalize(a):
    return a/np.linalg.norm(a)

def angle_value(path):
    def f(point):
        i=path.index(point)
        values=[]
        if i==0 or i==len(path)-1:
            return 100
        for j in range(1,10):
            a=np.array(path[max(0,i-j)])-np.array(path[i])
            b=np.array(path[min(len(path)-1,i+j)])-np.array(path[i])
            values.append(np.dot(normalize(a),normalize(b)))
        return max(values)
    return f

class RefineWindow(wx.Frame):
    def __init__(self, parent,paths):
        wx.Panel.__init__(self, parent, size=(500, 500))
        self.context=Context()

        self.paths=copy.deepcopy(paths)
        self.original_paths=paths
        self.simplification=[0]*len(paths)
        self.current_path=0
        self.sizer = wx.GridBagSizer(9, 5)
        self.SetSizer(self.sizer)

        self.lengths=[len(path) for path in paths]

        label = wx.StaticText(self, style=wx.ALIGN_CENTER, label="Choose line")
        self.sizer.Add(label,pos=(1,1))

        self.path_slider=wx.Slider(self, minValue=0, maxValue=max(1,(len(self.paths)-1)), name='path',size=(400,50))
        self.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.choose_path, self.path_slider)
        self.sizer.Add(self.path_slider,pos=(2,1))


        label = wx.StaticText(self, style=wx.ALIGN_CENTER, label="Line length")
        self.sizer.Add(label,pos=(3,1))

        self.length_slider=wx.Slider(self, minValue=0, maxValue=(len(self.paths[0])), name='length',size=(400,50))
        self.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.set_length, self.length_slider)
        self.sizer.Add(self.length_slider,pos=(4,1))


        label = wx.StaticText(self, style=wx.ALIGN_CENTER, label="Simplificaton")
        self.sizer.Add(label,pos=(5,1))

        self.simplification_slider=wx.Slider(self, minValue=0, maxValue=(len(self.paths[0]))-1, name='simplification',size=(400,50))
        self.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.simplify, self.simplification_slider)
        self.sizer.Add(self.simplification_slider,pos=(6,1))

        self.save_button = wx.Button(self,label="Save all")
        self.Bind(
            wx.EVT_BUTTON, self.save, self.save_button
        )
        self.sizer.Add(self.save_button,pos=(7,1))

        self.choose_path("")
        self.Show()

    def save(self,event):
        self.context.graphic_context.mark_temporary_paths([])
        for i,path in enumerate(self.paths):
            simplified_path=self._simplify(path[:self.lengths[i]],self.simplification[i])
            p=list(zip(simplified_path,simplified_path[1:]))
            self.context.data_context.add_lines(p)
        self.Parent.Close()
        self.Close()
    
    def choose_path(self,e):
        self.current_path=self.path_slider.GetValue()
        self.length_slider.SetRange(0,len(self.paths[self.current_path]))
        self.length_slider.SetValue(self.lengths[self.current_path])

        self.simplification_slider.SetRange(0,max(self.lengths[self.current_path]-1,0))
        self.simplification_slider.SetValue(min(self.simplification[self.current_path],self.lengths[self.current_path]-1))

        self.context.graphic_context.mark_temporary_paths([self.paths[self.current_path][:self.lengths[self.current_path]]])

    def set_length(self,e):
        self.lengths[self.current_path]=self.length_slider.GetValue()
        self.simplification_slider.SetRange(0,max(self.lengths[self.current_path]-1,0))
        self.simplification_slider.SetValue(0)
        self.simplification[self.current_path]=0
        self.context.graphic_context.mark_temporary_paths([self.paths[self.current_path][:self.lengths[self.current_path]]])

    def simplify(self,e):
        self.simplification[self.current_path]=self.simplification_slider.GetValue()
        path=self._simplify(self.paths[self.current_path][:self.lengths[self.current_path]],self.simplification_slider.GetValue())
        self.context.graphic_context.mark_temporary_paths([path])


    def _simplify(self,path,value):
        x=sorted(path,key=angle_value(path))
        for i in range(value):
            path.remove(x[i])
        return path
        


    