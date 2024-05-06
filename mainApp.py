import wx
import wx.grid
from wx.py.crust import CrustFrame,Crust
from wx import glcanvas
from OpenGL.GL import *
import OpenGL.GL.shaders
from pyrr import Matrix44, matrix44, Vector3
import time, sysconfig
import os.path
import numpy as np
import copy

import wx.lib.scrolledpanel

import context
from Cube import *
from stack_io import *
import shaders as sh
import mockup
from canvas import *

from scipy import ndimage
from skimage.morphology import skeletonize_3d
from skimage.transform import downscale_local_mean

from side_panels import *
from side_panel_skeleton_distances import SkeletonDistances
from position_setter import PositionSetter


TEST=False

from wx.py import frame

#crust frame override to get startup script a bit hax
class MyCrust(CrustFrame):
    def __init__(self, startup_script=None, parent=None, id=-1, title='MyCrust',
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE,    
                 rootObject=None, rootLabel=None, rootIsNamespace=True,
                 locals=None, InterpClass=None,
                 config=None, dataDir=None,
                 *args, **kwds):
        """Create CrustFrame instance."""
        frame.Frame.__init__(self, parent, id, title, pos, size, style,
                             shellName='PyCrust')
        frame.ShellFrameMixin.__init__(self, config, dataDir)

        if size == wx.DefaultSize:
            self.SetSize((800, 600))

        intro = 'PyCrust - The Flakiest Python Shell'
        self.SetStatusText(intro.replace('\n', ', '))
        self.crust = Crust(parent=self, intro=intro,
                           rootObject=rootObject,
                           rootLabel=rootLabel,
                           rootIsNamespace=rootIsNamespace,
                           locals=locals,
                           InterpClass=InterpClass,
                           startupScript='startup.py',
                           execStartupScript='startup.py',
                           *args, **kwds)
        self.shell = self.crust.shell

        # Override the filling so that status messages go to the status bar.
        self.crust.filling.tree.setStatusText = self.SetStatusText

        # Override the shell so that status messages go to the status bar.
        self.shell.setStatusText = self.SetStatusText

        self.shell.SetFocus()
        self.LoadSettings()

class Sidebar(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent, canvas):
        """Constructor"""
        super(Sidebar, self).__init__(parent)
        self.context = Context()
        self.context.main_sidebar = self
        self.canvas = canvas
        self.init_sidebar()
        self.SetupScrolling(scroll_x = False)
        self.SetSizerAndFit(self.sidebar)

        if TEST:
            self.directory_field_dr.SetValue("C:\\Users\\Andrzej\\Desktop\\2send_1\\dr79-1-7_uncorr_dr.tif")
            self.directory_field_pi.SetValue("C:\\Users\\Andrzej\\Desktop\\2send_1\\dr79-1-7_uncorr_pi.tif")
            #self.directory_field_dr.SetValue("I:\\symulacje\\stacks_vascular_1\\xp14_25_dr5v2_dr.tif")
            #self.directory_field_pi.SetValue("I:\\symulacje\\stacks_vascular_1\\xp14_25_dr5v2_dr.tif")
            #self.directory_field_dr.SetValue("I:\\symulacje\\stacks_vascular_1\\2019\\16.10\\xp5_5_dr.tif")
            #self.directory_field_pi.SetValue("I:\\symulacje\\stacks_vascular_1\\2019\\16.10\\xp5_5_pi.tif")
            self.do_downscale.SetValue(True)
            self.load_file_event(None)


    def init_sidebar(self):
        self.sidebar = wx.GridBagSizer(3, 3)

        # pi location
        self.directory_field_pi = wx.TextCtrl(self)
        self.directory_field_pi.SetToolTip("Directory")
        self.sidebar.Add(
            self.directory_field_pi,
            pos=(0, 0),
            span=(1, 4),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )

        self.open_button = wx.Button(self, label="Open pi file")
        self.sidebar.Add(
            self.open_button,
            pos=(0, 4),
            span=(1, 2),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )
        self.Bind(
            wx.EVT_BUTTON, self.open_dialog(self.directory_field_pi), self.open_button
        )

        # dr location
        self.directory_field_dr = wx.TextCtrl(self)
        self.directory_field_dr.SetToolTip("Directory")
        self.sidebar.Add(
            self.directory_field_dr,
            pos=(1, 0),
            span=(1, 4),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )

        self.open_button = wx.Button(self, label="Open dr file")
        self.sidebar.Add(
            self.open_button,
            pos=(1, 4),
            span=(1, 2),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )
        self.Bind(
            wx.EVT_BUTTON, self.open_dialog(self.directory_field_dr), self.open_button
        )

        self.do_downscale = wx.CheckBox(self, label="Downscale")
        self.sidebar.Add(
            self.do_downscale,
            pos=(2, 0),
            span=(1, 4),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )


        self.load_button = wx.Button(self, label="initilize files")
        self.sidebar.Add(
            self.load_button,
            pos=(2, 4),
            span=(1, 2),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )
        self.Bind(wx.EVT_BUTTON, self.load_file_event, self.load_button)

        # tension attribute
        self.tenision_field=wx.TextCtrl(self, name="tension") 
        self.sidebar.Add(
            self.tenision_field,
            pos=(3, 0),
            span=(1, 4),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )
        self.Bind(wx.EVT_TEXT, is_type(float, self.tenision_field), self.tenision_field)


        self.tension_button = wx.Button(self, label="set tension")
        self.sidebar.Add(
            self.tension_button,
            pos=(3, 4),
            span=(1, 2),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=5,
        )
        self.context=Context()
        self.context.side_bar=self
        global CONTEXT
        CONTEXT=self.context
        def set_tension(evt):
            self.context.graphic_context.curve_tension=float(self.tenision_field.GetValue())
            self.context.graphic_context.redraw_graph()
        self.Bind(wx.EVT_BUTTON, set_tension, self.tension_button)
        


        #scale of stack and reseting position button 
        self.scale_field=wx.TextCtrl(self, name="scale") 
        self.reset_button = wx.Button(self, label="Reset position")
        self.sidebar.Add(self.reset_button, pos=(34, 0), flag=wx.EXPAND, span=(1, 8))
        self.Bind(
            wx.EVT_BUTTON, lambda x: self.canvas.reset_position(float(self.scale_field.GetValue())), self.reset_button
        )
        self.sidebar.Add(
            self.scale_field,
            pos=(35, 0),
            span=(1, 1),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
        )
        self.scale_field.SetValue('1.7')
        self.Bind(wx.EVT_TEXT, is_type(float, self.scale_field), self.scale_field)
        self.scale_label=wx.StaticText(self, label='1.7')
        self.sidebar.Add(
            self.scale_label,
            pos=(35, 1),
            span=(1, 1),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT,
        )


        # self.test = wx.Button(self, label="Test 0")
        # self.sidebar.Add(self.test, pos=(14, 0), flag=wx.EXPAND, span=(1, 6))
        # def on_test(event):
        #     number=self.canvas.test()
        #     self.test.SetLabel(f"Test {number}")
        # self.Bind(
        #     wx.EVT_BUTTON, on_test, self.test
        # )
        self.pages = wx.Notebook(self)
        self.pages.parent=self
        self.skeleton_distances = SkeletonDistances(self.pages, self.canvas)
        #self.skeleton_manipulation = SkeletonManipulation(self.pages, self.canvas, self.skeleton_distances)
        self.sidebar.Add(self.pages, pos=(5, 0), flag=wx.EXPAND|wx.DOWN|wx.UP, span=(28, 6))
        #self.pages.AddPage(self.skeleton_manipulation,"Skeleton manipulation")
        self.pages.AddPage(self.skeleton_distances, "Skeleton distances")
        self.pages.AddPage(ClipPlanes(self.pages, self.canvas), "Clipping")
        self.pages.AddPage(
            GradientSetter(self.pages, self.canvas, self.canvas.reload_dr_gradinet),
            "dr gradient",
        )
        self.pages.AddPage(
            GradientSetter(self.pages, self.canvas, self.canvas.reload_pi_gradinet),
            "pi gradient",
        )

    def open_dialog(self, directory_field):
        def f(event):
            openFileDialog = wx.FileDialog(
                self,
                "Open",
                "",
                "",
                "Tiff files (*.tif)|*.tif*|Any type (*.*)|*.*",
                wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            )
            if openFileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind
            directory_field.SetLabelText(openFileDialog.GetPath())
            openFileDialog.Destroy()

        return f

    def load_file_event(self, event):
        if (os.path.isfile(self.directory_field_dr.GetValue()) and 
            os.path.isfile(self.directory_field_pi.GetValue())):
            dr_array,pi_array=self.canvas.init_concentration(
                self.directory_field_dr.GetValue(),
                self.directory_field_pi.GetValue(),
                downscale=self.do_downscale.GetValue(),
            )
            self.position_setter=PositionSetter(self)

    def save_skeleton(self, event):
        saveFileDialog = wx.FileDialog(
            self,
            "Save",
            "",
            "",
            "Tiff files (*.tif)|*.tif*|Any type (*.*)|*.*",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()
        saveFileDialog.Destroy()
        if "tif" not in path[-4:]:
            path = path + ".tif"
        save_stack(self.canvas.skeleton, name=path)

    def load_skeleton(self, event):
        openFileDialog = wx.FileDialog(
            self,
            "Open",
            "",
            "",
            "Tiff files (*.tif)|*.tif*|Any type (*.*)|*.*",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind

        self.canvas.load_skeleton_from_file(openFileDialog.GetPath())
        openFileDialog.Destroy()


class MyFrame(wx.Frame):
    def __init__(self):
        self.context=Context()
        self.context.main_frame=self
        self.size = (1280, 880)
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
        self.canvas = OpenGLCanvas(self)
        self.SetSizer(self.hbox, wx.EXPAND | wx.ALL)
        self.hbox.Add(self.canvas, wx.EXPAND | wx.ALL)
        self.sidebar = Sidebar(self, self.canvas)

        self.hbox.Add(self.sidebar)
        self.Connect(-1, -1, self.context.data_context.EVT_RESULT_ID, self.OnResult)

    def on_resize(self, event):
        pass
        print(self.size)

    def on_close(self, event):
        self.Destroy()
        sys.exit(0)

    def OnResult(self, event):
        self.context.simulation_controller.color_step_post(time_step=event.data)

    def OnIdle(self, event):
        self.canvas.Refresh()         # refresh self and all its children

    def OnKeyDown(self, event=None):
        print(f"event, {event}")

class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame()
        frame.Show()
        frame.Maximize(True)


        frame = MyCrust()
        frame.Show()
        return True


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('MyApp().MainLoop()')
    app = MyApp()
    app.MainLoop()

