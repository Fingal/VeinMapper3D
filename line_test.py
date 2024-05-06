from stack_io import *
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


def PIL2wx (image):
    width, height = image.size
    return wx.Bitmap.FromBuffer(width, height, image.tobytes())

class ImagePanel(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent,array,func,**kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, **kwargs)
        self.SetupScrolling()
        self.array=array.astype(np.float16)
        self.on_rectangle=func
        self.parent=parent
        self.is_pi_showed=False
        self._x=0.2
        self.pos=None
        self.added_points=[]

        self._timer=time.time()

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.vbox)
        png = PIL2wx(Image.fromarray(self.array.astype(np.uint8)))
        self.img=wx.StaticBitmap(self, -1, png)
        self.vbox.Add(self.img)
        

        self.img.Bind(wx.EVT_MOUSE_EVENTS,self.on_mouse)
        self.points=[]
        

        
        dc = wx.ClientDC(self.img)
        self.overlay=wx.Overlay()
        odc = wx.DCOverlay(self.overlay, dc)

    def set_image(self):
        png = PIL2wx(Image.fromarray(self.array.astype(np.uint8)))
        self.img.SetBitmap(png)
    
    def _add_point(self,point):
        self.points.append(point)
        print(len(self.points))
        if len(self.points)>=2:
            a,b = map(np.array,self.points[-2:])
            length=sum(np.abs(a-b))*2
            for i in range(length+1):
                point = b*i/length+a*(1-i/length)
                self.array[int(point[0]),int(point[1]),:]=0
            self.set_image()

    def add_point(self,point):
        def p_fun(t,a,b,m,i):
            p = (2*t**3 - 3*t**2+1) *a + (t**3 - 2*t**2+t)*m[i]+((-2)*t**3 + 3*t**2)*b+(t**3 - t**2)*m[i+1]
            p=np.maximum(p,0)
            p=np.minimum(p,max(self.array.shape)-2)
            return p

        C=0.50
        self.points.append(point)
        m=[]
        if len(self.points)>=2:
            points = [self.points[0]]+self.points+[self.points[-1]]
            self.array=np.ones(self.array.shape,self.array.dtype)*250
            for a,b in zip(map(np.array,points[:-1]),map(np.array,points[2:])):
                m.append((1-C)*(b-a)*(2))
            old_p=None
            for i,(a,b)  in enumerate(zip(map(np.array,self.points[0:-1]),map(np.array,self.points[1:]))):
                step=100
                for t in np.linspace(0,1,step):
                    p = p_fun(t,a,b,m,i)
                    self.array[int(p[0]),int(p[1]),:]=0
                    if type(old_p)!=type(None):
                        distance = np.sum((p-old_p)**2)**0.5
                        if distance>=2:
                            for t2 in range(1,int(distance*1.5)):
                                new_p = p_fun(t-1/(step*int(distance*1.5))*t2,a,b,m,i)
                                self.array[int(new_p[0]),int(new_p[1]),:]=0
                    old_p=p


                    
                    
            self.set_image()


            



    def on_mouse(self, event):
        if event.LeftUp():
            self.pos = event.GetPosition()     
            self.pos=(self.pos[1],self.pos[0])
            self.add_point(self.pos)
            


class MyFrame(wx.Frame):
    def __init__(self):
        self.size = (1280, 1280)
        wx.Frame.__init__(
            self,
            None,
            title="My wx frame",
            style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE,
        )
        self.SetMinSize(self.size)

        # self.panel = wx.Panel(self)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        # self.canvas = mockup.OpenGLCanvasMockup(self)
        self.canvas = ImagePanel(self,np.ones((1280,1280,3))*250,None)


class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame()
        frame.Show()
        return True


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('MyApp().MainLoop()')
    app = MyApp()
    app.MainLoop()
