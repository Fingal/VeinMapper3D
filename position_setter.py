from stack_io import *
import calculate_line  
from context import Context

import wx
import wx.lib.scrolledpanel
from itertools import product
from math import floor, ceil
import PIL
import numpy as np
import time

from importlib import reload
import test_functions as tf
from types import MethodType

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom



def PIL2wx (image):
    width, height = image.size
    return wx.Bitmap.FromBuffer(width, height, image.tobytes())


class ImagePanel(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent,**kwargs):
        
        self.context=Context()
        self.context.image_slicer=self
        self._timer=time.time()
        pi_array = self.context.data_context.pi_stack
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, **kwargs)
        self.SetupScrolling()
        first_time=time.time()
        self.dr_array= self.context.data_context.dr_stack
        self.pi_array=np.minimum(pi_array/ (np.amax(pi_array) / 255),255).astype(np.float16)
        print(self.pi_array.dtype)
        self.parent=parent
        self.is_pi_showed=False
        self._x=0.2
        self.pos=None
        self.temporary_points=[]
        self.scale=1
        
        # print('time',time.time()-first_time)
        # self.compressed_array=zoom(gaussian_filter(dr_array,4),0.5)
        # print('time',time.time()-first_time)
        self.vbox = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.vbox)
        self.max_value=np.max(self.dr_array)
        png = PIL2wx(get_image(self.dr_array,0))
        self.img=wx.StaticBitmap(self, -1, png)
        self.vbox.Add(self.img)
        
        

        self.slider=wx.Slider(self, minValue=0, maxValue=self.dr_array.shape[2]-1, name='height',size=(20,self.img.GetSize()[0]//3),style=wx.VERTICAL)
        self.Bind(wx.EVT_SCROLL, self.on_slide, self.slider)
        
        self.intensification_slider=wx.Slider(self, minValue=0, maxValue=self.max_value-5, name='intensification',size=(20,self.img.GetSize()[0]//3),style=wx.VERTICAL)
        self.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.set_max, self.intensification_slider)

        self.scale_slider=wx.Slider(self, minValue=0, maxValue=2000, name='scale',size=(20,self.img.GetSize()[0]//3),style=wx.VERTICAL)
        self.scale_slider.SetValue(self.scale_slider.GetMax()//2)
        self.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.set_scale, self.scale_slider)
        
        self.img.Bind(wx.EVT_MOUSE_EVENTS,self.on_mouse)

        self.layer_label=wx.StaticText(self, label='layer 0:   ',size=(80,20))
        
        hbox1=wx.BoxSizer(wx.VERTICAL)
        hbox1.Add(wx.StaticText(self, label='   ',size=(80,self.img.GetSize()[0]//5)))
        hbox1.Add(self.layer_label)
        hbox1.Add(self.slider)

        self.vbox.Add(hbox1)

        pi_checkbox=wx.CheckBox(self,label='pi layer')
        self.Bind(wx.EVT_CHECKBOX, self.on_checkbox, pi_checkbox)
        hbox=wx.BoxSizer(wx.VERTICAL)

        hbox.Add(wx.StaticText(self, label='   ',size=(80,self.img.GetSize()[0]//5)))
        hbox.Add(wx.StaticText(self, label='intensification: ',size=(80,20)))
        hbox.Add(self.intensification_slider)
        hbox.Add(pi_checkbox)
        self.vbox.Add(hbox)

        hbox=wx.BoxSizer(wx.VERTICAL)
        hbox.Add(wx.StaticText(self, label='   ',size=(80,self.img.GetSize()[0]//5)))
        hbox.Add(wx.StaticText(self, label='scale: ',size=(80,20)))
        hbox.Add(self.scale_slider)

        reset_scale_button = wx.Button(self,label="reset")
        self.Bind(wx.EVT_BUTTON, self.reset_scale, reset_scale_button)
        hbox.Add(reset_scale_button)
        self.vbox.Add(hbox)

        
        hbox=wx.BoxSizer(wx.VERTICAL)
        hbox.Add(wx.StaticText(self, label='',size=(30,self.img.GetSize()[0]//3)))
        hbox.Add(wx.StaticText(self, label='label: '), wx.EXPAND | wx.ALL)
        self.label = wx.TextCtrl(self)
        hbox.Add(self.label)
        self.vbox.Add(hbox, wx.EXPAND | wx.LEFT | wx.RIGHT)
        self._precalculate_images()


        
        dc = wx.ClientDC(self.img)
        self.overlay=wx.Overlay()
        odc = wx.DCOverlay(self.overlay, dc)


    def on_checkbox(self,event):
        self.is_pi_showed = event.IsChecked()
        self._precalculate_images()

    def on_slide(self,event):
        self.layer_label.SetLabel(f'layer {(self.slider.GetValue())}: ')
        self.set_image(self.slider.GetValue())

    def set_scale(self,event):
        self.scale=self.scale_slider.GetValue()/self.scale_slider.GetMax()+0.5
        self._set_image(self.slider.GetValue()) 
        self.parent.SetSize((int(self.context.data_context.dr_stack.shape[1]*self.scale)+600,int(self.context.data_context.dr_stack.shape[0]*self.scale)+50))
        self.SetSize((int(self.context.data_context.dr_stack.shape[1]*self.scale)+700,int(self.context.data_context.dr_stack.shape[0]*self.scale)+10))

    def reset_scale(self,event):
        self.scale_slider.SetValue(self.scale_slider.GetMax()//2)
        self.set_scale(event)

    def set_max(self,event):
        self._timer=time.time()
        self._set_image(self.slider.GetValue())
    
    def set_image(self,height):
        #arbitrary time
        if abs(time.time()-self._timer)>0.08:
            self._set_image(height) 
    
    def _set_image(self,height):
        #print("aaa")
        self._timer=time.time()
        #calculate base image
        arr=np.zeros((*self.context.data_context.dr_stack[:,:,height].shape,3),dtype=np.uint8)
        if not self.is_pi_showed:
            _arr=(np.minimum(self.context.data_context.dr_stack[:,:,height],self.max_value-self.intensification_slider.GetValue()))/ (self.max_value-self.intensification_slider.GetValue()) * 255
            #arr=(self.dr_array>self.intensification_slider.GetValue()).astype(np.int8)
            #print('value',self.intensification_slider.GetValue())
            # I=arr>3
            # arr[I]=np.log2(arr[I])
            # print(arr)
            # print(np.max(arr))

            arr[:,:,0]=_arr
            arr[:,:,1]=_arr
            arr[:,:,2]=_arr
            #print(np.amax(_arr),self.max_value-self.intensification_slider.GetValue())
        else:
            _arr=(np.minimum(self.context.data_context.dr_stack[:,:,height],self.max_value-self.intensification_slider.GetValue()))/ (self.max_value-self.intensification_slider.GetValue()) * 255
            pi_arr=self.pi_array[:,:,height]*(340-_arr)/300
            
            _pi=pi_arr.astype(np.uint8)
            arr[:,:,0]=np.minimum(_arr + pi_arr,255).astype(np.uint8)
            arr[:,:,1]=np.maximum(_arr - pi_arr,0).astype(np.uint8)
            arr[:,:,2]=np.maximum(_arr - pi_arr,0).astype(np.uint8)

        



        #arr=np.copy(self._array_cache[:,:,height,:])


        surface_points=[]
        other_points=[]
        for point in self.context.data_context.values['point']:
            if abs(point[2]-self.slider.GetValue())<2:
                if 's' in self.context.data_context.labels['point'][point]:
                    surface_points.append(list(map(int,point)))
                else:
                    other_points.append(list(map(int,point)))
        if self.pos:
            #self.parent.canvas.mark_stack_points([(*self.pos,self.slider.GetValue())])
            self.context.graphic_context.mark_temporary_points([(*self.pos,self.slider.GetValue())])
            #point colour
            arr[self.pos[0]-2:self.pos[0]+2,self.pos[1]-2:self.pos[1]+2,1]=255
            arr[self.pos[0]-2:self.pos[0]+2,self.pos[1]-2:self.pos[1]+2,0]=0
            arr[self.pos[0]-2:self.pos[0]+2,self.pos[1]-2:self.pos[1]+2,2]=0
        for _point in self.temporary_points:
            point = [int(p) for p in _point]
            if abs(self.slider.GetValue()-point[2])<2:
                arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,0]=255
                arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,1]=255
                arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,2]=0
        for point in surface_points:
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,0]=255
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,1]=0
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,2]=255
        for point in other_points:
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,0]=0
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,1]=0
            arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,2]=255



        
        # print(self.added_points,self.slider.GetValue())
        # for point in filter(lambda x: x[2]==self.slider.GetValue(),self.added_points):
        #     arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,1]=0
        #     arr[point[0]-2:point[0]+2,point[1]-2:point[1]+2,2]=250
        #bitmap=wx.BitmapFromImage(wx.ImageFromBuffer(arr.shape[0], arr.shape[1],arr).Scale(arr.shape[0]*self.scale, arr.shape[1]*self.scale, wx.IMAGE_QUALITY_HIGH))
        #print(arr.shape)
        bitmap=wx.Bitmap(wx.ImageFromBuffer(arr.shape[1], arr.shape[0],arr).Scale(arr.shape[1]*self.scale, arr.shape[0]*self.scale, wx.IMAGE_QUALITY_HIGH))
        self.img.SetBitmap(bitmap)

    def _added_point(self,point):
        self._array_cache[max(0,point[0]-2):point[0]+2,max(0,point[1]-2):point[1]+2,max(0,point[2]-2):point[2]+2,0]=0
        self._array_cache[max(0,point[0]-2):point[0]+2,max(0,point[1]-2):point[1]+2,max(0,point[2]-2):point[2]+2,1]=0
        self._array_cache[max(0,point[0]-2):point[0]+2,max(0,point[1]-2):point[1]+2,max(0,point[2]-2):point[2]+2,2]=250

    def _precalculate_images(self):
        self.set_image(self.slider.GetValue())
        return
        self._array_cache=np.zeros((*self.dr_array.shape,3),dtype=np.uint8)
        if not self.is_pi_showed:
            arr=(np.minimum(self.dr_array,self.max_value-self.intensification_slider.GetValue()))
            #arr=(self.dr_array>self.intensification_slider.GetValue()).astype(np.int8)
            #print('value',self.intensification_slider.GetValue())
            # I=arr>3
            # arr[I]=np.log2(arr[I])
            # print(arr)
            # print(np.max(arr))

            self._array_cache[:,:,:,0]=arr / (np.max(arr) / 255)
            self._array_cache[:,:,:,1]=arr / (np.max(arr) / 255)
            self._array_cache[:,:,:,2]=arr / (np.max(arr) / 255)
            #print(np.amax(arr),self.max_value-self.intensification_slider.GetValue())
        else:
            arr=(np.minimum(self.dr_array,self.max_value-self.intensification_slider.GetValue()))
            arr=arr / (np.amax(arr) / 255)
            pi_arr=self.pi_array*(340-arr)/300
            
            _pi=pi_arr.astype(np.uint8)
            self._array_cache[:,:,:,0]=np.minimum(arr + pi_arr,255).astype(np.uint8)
            self._array_cache[:,:,:,1]=np.maximum(arr - pi_arr,0).astype(np.uint8)
            self._array_cache[:,:,:,2]=np.maximum(arr - pi_arr,0).astype(np.uint8)

        for point in self.context.data_context.values['point']:
            self._added_point(list(map(int,point)))
        
        self.set_image(self.slider.GetValue())



    def on_mouse(self, event):
        if event.LeftDown() or event.Dragging():
            self.pos = event.GetPosition()     
            self.pos=(int(self.pos[1]/self.scale),int(self.pos[0]/self.scale))
            self.set_image(self.slider.GetValue()) 
        if event.LeftUp():  
            self._set_image(self.slider.GetValue()) 
            #print(event.GetPosition())
            end_pos = event.GetPosition()
            # if self.on_rectangle:
            #     self.on_rectangle(tuple(reversed(self.pos)),tuple(reversed(end_pos)),self.slider.GetValue())

    def add_point(self,_):
        self.context.data_context.add_points([(*self.pos,self.slider.GetValue())],label=self.label.GetValue())
        self.pos=None
        self.context.graphic_context.mark_temporary_points([])
        self._set_image(self.slider.GetValue())
        self._set_image(self.slider.GetValue())

    def add_point_surface(self,_):
        #self.context.data_context.add_points([(*self.pos,self.slider.GetValue())],label=self.label.GetValue())
        x=self.context.data_context.compute_height_distance((*self.pos,self.slider.GetValue()))
        new_point= (self.pos[0],self.pos[1],x)
        name=self.label.GetValue()+' s'
        self.context.data_context.add_points([new_point],label=name)
        self.context.data_context.add_label(new_point,name)


        self.pos=None
        self.context.graphic_context.mark_temporary_points([])
        self._set_image(self.slider.GetValue())
        self._set_image(self.slider.GetValue())
        #self.context.graphic_context.mark_point_setter(pos)

    # def add_point(self,_):
    #     self._add_point((*self.pos,self.slider.GetValue()))

    # # def _add_point(self,position):
    # #     positions=calculate_line.calculate_line(self.dr_array,position,step_size=10,step_number=30,
    # #                                             distance=20,angle=np.pi/1.5)
    # #     for pos in positions:
    # #         self.parent.parent.skeleton_distances.add_point(pos,self.label.GetValue())
    # #         self.added_points.append(pos)
    # #         self._added_point(pos)
    # #     for line in zip(positions,positions[1:]):
    # #         self.parent.parent.skeleton_distances.add_line(*line,"")
    # #     self.pos=None
    # #     self._set_image(self.slider.GetValue())
    # #     self.parent.canvas.mark_stack_points([])
        
    # # import pdb; pdb.set_trace()



class PositionSetter(wx.Frame):
    def __init__(self,parent):
        self.context=Context()
        dr = self.context.data_context.dr_stack
        self.size = (dr.shape[1]+500, dr.shape[0]+100)
        wx.Frame.__init__(self, None,size=self.size)
        self.parent=parent
        self.Show()
        self.box = wx.BoxSizer(wx.HORIZONTAL)
        self.bbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.box)
        self.image_panel=ImagePanel(self,size=(dr.shape[1]+500,dr.shape[0]+5000))
        self.box.Add(self.image_panel, wx.EXPAND | wx.ALL)

        self.box.Add(self.bbox, border=0, flag=wx.EXPAND | wx.ALL)
        self.button = wx.Button(self, label='Add point')
        self.bbox.Add(self.button, border=5, flag=wx.EXPAND | wx.ALL)
        self.button_surface = wx.Button(self, label='Add point\non surface')
        self.bbox.Add(self.button_surface, border=5, flag=wx.EXPAND | wx.ALL)
        self.Bind(wx.EVT_BUTTON, self.image_panel.add_point, self.button)
        self.Bind(wx.EVT_BUTTON, self.image_panel.add_point_surface, self.button_surface)
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)

    def OnKeyDown(self, event=None):
        print(f"event, {event}")

    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_UP:
            self.image_panel.slider.SetValue(min(self.image_panel.slider.GetMax(),self.image_panel.slider.GetValue()+1))
        if keycode == wx.WXK_DOWN:
            self.image_panel.slider.SetValue(max(0,self.image_panel.slider.GetValue()-1))
