import wx
import wx.lib.scrolledpanel
import copy
import numpy as np
from math import floor, ceil
from importlib import reload
import test_functions as tf
import pickle
from types import MethodType


# checks if format is correct
def is_type(t, text, m=None):
    def f(event):
        value = text.GetValue().strip()
        pos = text.GetInsertionPoint()
        try:
            t(value)
            if m != None and t(value) > m:
                raise Exception
        except:
            text.ChangeValue(value[: pos - 1] + value[pos:])
            text.SetInsertionPoint(pos - 1)

    return f


class SkeletonManipulation(wx.Panel):
    def __init__(self, parent, canvas, skeleton_distances):
        wx.Panel.__init__(self, parent)
        # self.SetupScrolling()
        self.skeleton_distances = skeleton_distances
        self.parent = parent
        self.sizer = wx.GridBagSizer(2, 2)
        self.SetSizer(self.sizer)
        self.canvas = canvas
        self.mode = ""
        self.points = []

        self.info = wx.StaticText(self, style=wx.ALIGN_RIGHT)
        self.sizer.Add(
            self.info, pos=(0, 0), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=(1, 6)
        )
        self.mark_junction_button = self.add_button(
            "Mark Junciton", (1, 0), (1, 1), self.mark_junction
        )
        self.mark_any_button = self.add_button(
            "Mark Any Point", (2, 0), (1, 1), self.mark_any_point
        )
        self.mark_any_button = self.add_button("Cancel", (4, 0), (1, 1), self.cancel)

        self.remove_secondary_lines_button = self.add_button(
            "Rem. Se. Lines", (8, 0), (1, 3), self.remove_secondary_lines
        )
        self.remove_to_junction_button = self.add_button(
            "Rem. To Junciton", (9, 0), (1, 3), self.remove_to_junction
        )
        self.remove_line_button = self.add_button(
            "Rem. Lines Between", (10, 0), (1, 3), self.remove_line
        )

        self.connect_lines_button = self.add_button(
            "Connect Points", (12, 0), (1, 3), self.connect_points
        )
        self.straighten_line_button = self.add_button(
            "Straighten Line", (13, 0), (1, 3), self.straighten_line
        )
        self.straighten_line_button = self.add_button(
            "Undo remove", (15, 0), (1, 3), lambda x: self.canvas.undo()
        )

        


        self.straighten_line_button = self.add_button(
            "Show marked points",
            (8, 3),
            (1, 3),
            self.show_points_event,
            button_type=wx.ToggleButton,
        )
        self.straighten_line_button = self.add_button(
            "Add line to measue", (9, 3), (1, 3), self.add_line_to_measure
        )
        self.straighten_line_button = self.add_button(
            "Add points to measue", (10, 3), (1, 3), self.add_points_to_measure
        )

        
        self.connect_lines_button = self.add_button(
            "Load Skeleton", (12, 3), (1, 3), self.load_skeleton
        )
        self.straighten_line_button = self.add_button(
            "Save Skeleton", (13, 3), (1, 3), self.save_skeleton
        )

        self.show_points = False

        self.mark_any_button = self.add_button("Debug", (20, 0), (1, 2), self.debug)
        self.SetSizer(self.sizer)

        self.grid = wx.grid.Grid(self)

        self.grid.EnableEditing(False)
        self.grid.CreateGrid(5, 2)
        self.grid.SetColLabelValue(0, "Point")
        self.grid.SetColLabelValue(1, "Type")
        self.grid.SetRowLabelSize(20)
        self.grid.SetColSize(0, 100)
        self.grid.SetColSize(1, 30)
        self.sizer.Add(
            self.grid, pos=(1, 1), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=(6, 6)
        )
        self.grid.SetSelectionMode(1)
        self.grid.GetGridWindow().Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)
        self.grid.GetGridWindow().Bind(wx.EVT_LEFT_UP, self.on_click)

    def on_click(self, evnt):
        if self.show_points:
            self.canvas.skeleton_bundle.mark_points(
                list(map(lambda x: self.points[x][0], self.grid.GetSelectedRows()))
            )
            self.canvas.OnDraw()

    def connect_points(self, evnt):
        points = self.get_points(2)
        if points:
            self.canvas.connect_points(*points)

    def show_points_event(self, evnt):
        self.show_points = evnt.IsChecked()
        if evnt.IsChecked() and self.grid.GetSelectedRows():
            self.canvas.skeleton_bundle.mark_points(
                list(map(lambda x: self.points[x][0], self.grid.GetSelectedRows()))
            )
            self.canvas.OnDraw()
        else:
            self.canvas.skeleton_bundle.mark_points([])
            self.canvas.OnDraw()

    def straighten_line(self, evnt):
        points = list(self.get_points(2))
        if points:
            self.canvas.remove_line(*points)
            self.canvas.connect_points(*points)

    def add_line_to_measure(self, evnt):
        points = list(self.get_points(2))
        if points:
            self.skeleton_distances.add_line(*points)

    def add_points_to_measure(self, evnt):
        points = list(self.get_points())
        if points:
            for point in points:
                self.skeleton_distances.add_point(point)
            # self.canvas.mark_line(*points)

    def mark_junction(self, evnt):
        if self.mode != "mark point":
            self.mode = "mark junction"
            self.info.SetLabel("Curent mode: mark junction")
            self.canvas.set_mode("find_junction", self.add_point)

    def mark_any_point(self, evnt):
        if self.mode != "mark junction":
            self.mode = "mark point"
            self.info.SetLabel("Current mode: mark point")
            self.canvas.set_mode("find_any_point", self.add_point)

    def cancel(self, evnt):
        if self.mode != "mark junction":
            self.canvas.set_mode("find_junction", None)

        if self.mode != "mark point":
            self.canvas.set_mode("find_any_point", None)

        self.info.SetLabel("")
        self.mode = ""

    def get_points(self, number=0):
        if number:
            dic = {1: "First select a point"}
            if len(self.grid.GetSelectedRows()) == number:
                selected = map(lambda x: self.points[x][0], self.grid.GetSelectedRows())
            elif len(self.grid.GetSelectedRows()) == 0:
                selected = selected = map(lambda x: x[0], self.points[-number:])
            elif len(self.grid.GetSelectedRows()) != number:
                self.info.SetLabel(dic.get(number, d=f"First select {number} points"))
                return
        else:
            selected = map(lambda x: self.points[x][0], self.grid.GetSelectedRows())
        return selected

    def remove_secondary_lines(self, evnt):
        points = self.get_points(2)
        if points:
            self.canvas.remove_secondary_lines(*points)

    def remove_to_junction(self, evnt):
        points = self.get_points(1)
        if points:
            self.canvas.remove_to_junction(*points)

    def remove_line(self, evnt):
        points = self.get_points(2)
        if points:
            self.canvas.remove_line(*points)

    def add_point(self, point, typ):
        self.info.SetLabel("")
        if (point, typ) not in self.points:
            self.mode = ""
            self.points.append((point, typ))
            self.reload_grid()

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
        pickle.dump(self.canvas.skeleton_graph,file)

    def load_skeleton(self, event):
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

        file=open(path,'rb')
        self.canvas.reload.skeleton_graph(pickle.load(file))

    def debug(self, evnt):
        import pdb

        pdb.set_trace()

    def reload_grid(self):
        if self.grid.GetNumberRows():
            self.grid.DeleteRows(0, self.grid.GetNumberRows())
        self.grid.AppendRows(max((len(self.points), 5)))
        for i, (value, typ) in enumerate(self.points):
            self.grid.SetCellValue(i, 0, "{:.3f},{:.3f},{:.3f}".format(*value))
            self.grid.SetCellValue(i, 1, typ)

    def on_right_click(self, event):
        """"""
        x, y = self.grid.CalcUnscrolledPosition(event.GetX(), event.GetY())
        row, col = self.grid.XYToCell(x, y)
        # print(row, col)
        self.points.pop(row)
        self.reload_grid()

    def add_button(self, label, pos, span, funct, button_type=wx.Button):
        button = button_type(self, label=label)
        self.sizer.Add(button, pos=pos, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=span)
        if button_type == wx.Button:
            self.Bind(wx.EVT_BUTTON, funct, button)
        if button_type == wx.ToggleButton:
            self.Bind(wx.EVT_TOGGLEBUTTON, funct, button)

        return button


class GradientSetter(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent, canvas, reload_gradient):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent)
        self.SetupScrolling()
        self.sizer = wx.GridBagSizer(2, 0)
        self.values = []
        self.canvas = canvas
        self.reload_gradient = reload_gradient

        self.grid = wx.grid.Grid(self)

        grid_size = 10
        self.grid.EnableEditing(False)
        self.grid.CreateGrid(10, 3)
        self.grid.SetColLabelValue(0, "value")
        self.grid.SetColLabelValue(1, "Color in RGBA")
        self.grid.SetColLabelValue(2, "Color")
        self.grid.SetColSize(2, 70)
        self.grid.SetColSize(0, 50)
        self.grid.SetColSize(1, 120)
        self.grid.SetRowLabelSize(0)
        self.grid.GetGridWindow().Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)

        self.sizer.Add(self.grid, pos=(0, 0), span=(grid_size, 30))

        # self.sizer.Add(self.grid, pos=(0, 0), span=(5, 30), flag=wx.EXPAND)

        # print(self.sizer.FindItemAtPosition(wx.GBPosition(1,1)))
        # print(self.sizer.GetItemSpan(self.grid))

        self.sliders = {}
        for i, name in enumerate(["red", "green", "blue", "alpha"]):
            label = wx.StaticText(self, style=wx.ALIGN_LEFT, label=name + ":")
            self.sizer.Add(label, pos=(grid_size + i, 0), span=(1, 5), flag=wx.EXPAND)
            self.sliders[name] = wx.Slider(self, minValue=0, maxValue=255, name=(name))
            self.sizer.Add(
                self.sliders[name],
                pos=(grid_size + i, 6),
                span=(1, 21),
                flag=wx.EXPAND,
                border=1,
            )
            self.sliders[name].SetValue(255)
            self.Bind(wx.EVT_SCROLL, self.on_slide, self.sliders[name])

        label = wx.StaticText(self, style=wx.ALIGN_LEFT, label="value:")
        self.sizer.Add(label, pos=(grid_size + 4, 0), span=(1, 5), flag=wx.EXPAND)
        self.value = wx.TextCtrl(self, name="value")
        self.sizer.Add(
            self.value, pos=(grid_size + 4, 7), span=(1, 12), flag=wx.EXPAND, border=1
        )
        self.Bind(wx.EVT_TEXT, is_type(int, self.value), self.value)

        self.button = wx.Button(self, label="Add")
        self.sizer.Add(
            self.button, pos=(grid_size + 5, 0), span=(1, 12), flag=wx.EXPAND
        )
        self.Bind(wx.EVT_BUTTON, self.add_color, self.button)
        self.p = wx.Panel(self)
        self.sizer.Add(self.p, pos=(grid_size + 5, 12), span=(1, 15), flag=wx.EXPAND)
        self.rgba = (255, 255, 255, 255)
        self.color = wx.Colour(*self.rgba)
        self.p.SetBackgroundColour(self.color)

        self.load_button = wx.Button(self, label="Load")
        self.sizer.Add(
            self.load_button, pos=(grid_size + 7, 0), span=(0, 14), flag=wx.EXPAND
        )
        self.Bind(wx.EVT_BUTTON, self.load, self.load_button)

        self.save_button = wx.Button(self, label="Save")
        self.sizer.Add(
            self.save_button, pos=(grid_size + 7, 14), span=(0, 14), flag=wx.EXPAND
        )
        self.Bind(wx.EVT_BUTTON, self.save, self.save_button)

        self.SetSizerAndFit(self.sizer)

    def reload_grid(self):
        self.values.sort(key=lambda x: x[0])
        if self.grid.GetNumberRows():
            self.grid.DeleteRows(0, self.grid.GetNumberRows())
        self.grid.AppendRows(max((len(self.values), 10)))
        for i, (value, rgba) in enumerate(self.values):
            self.grid.SetCellValue(i, 0, str(value))
            self.grid.SetCellValue(i, 1, str(rgba))
            self.grid.SetCellBackgroundColour(i, 2, wx.Colour(rgba))

        GRADIENT_SIZE = 255
        gradient = [(0, 0, 0, 0)] * GRADIENT_SIZE
        t_values = copy.copy(self.values)
        if self.values and self.values[0][0] > 0:
            t_values = [(0, (0, 0, 0, 0))] + t_values
        print(self.canvas.max_concentration)
        t_values = [(min(1, a / self.canvas.max_concentration), b) for a, b in t_values]
        # print(t_values)
        if t_values and t_values[-1][0] < 1:
            t_values.append((1, t_values[-1][1]))
        for v1, v2 in zip(t_values[:-1], t_values[1:]):
            for i in range(ceil(v1[0] * GRADIENT_SIZE), floor(v2[0] * GRADIENT_SIZE)):
                weight = (i - ceil(v1[0] * GRADIENT_SIZE)) / (
                    floor(v2[0] * GRADIENT_SIZE) - ceil(v1[0] * GRADIENT_SIZE)
                )
                # print('weight:',weight,i,)
                approx_collor = lambda x: int(
                    weight * v2[1][x] + (1 - weight) * v1[1][x]
                )
                gradient[i] = (
                    approx_collor(0),
                    approx_collor(1),
                    approx_collor(2),
                    approx_collor(3),
                )
        # print(gradient)
        self.reload_gradient(np.array(gradient).astype(np.uint8))

    def add_color(self, event):
        # Todo remove duplicates
        if self.value.GetValue():
            value = float(self.value.GetValue())
            self.values.append((value, self.rgba))
            self.reload_grid()

    def on_right_click(self, event):
        """"""
        x, y = self.grid.CalcUnscrolledPosition(event.GetX(), event.GetY())
        row, col = self.grid.XYToCell(x, y)
        # print(row, col)
        self.values.pop(row)
        self.reload_grid()

    def on_slide(self, event):
        self.rgba = (
            self.sliders["red"].GetValue(),
            self.sliders["green"].GetValue(),
            self.sliders["blue"].GetValue(),
            self.sliders["alpha"].GetValue(),
        )
        self.color.Set(*self.rgba)
        self.p.SetBackgroundColour(self.color)
        self.Refresh()

    def load(self, event):
        openFileDialog = wx.FileDialog(
            self,
            "Open",
            "",
            "",
            "Txt (*.txt)|*.txt|Any type (*.*)|*.*",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        self.values = []
        file = open(openFileDialog.GetPath())
        openFileDialog.Destroy()
        for line in file:
            value, *color = map(int, line.split(" "))
            self.values.append((value, tuple(color)))
        self.reload_grid()

    def save(self, event):
        saveFileDialog = wx.FileDialog(
            self,
            "Save",
            "",
            "",
            "Txt (*.txt)|*.txt|Any type (*.*)|*.*",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind
        path = saveFileDialog.GetPath()
        saveFileDialog.Destroy()
        file = open(path, "w")
        file.writelines(
            ["{} {} {} {} {}\n".format(value, *color) for value, color in self.values]
        )


class ClipPlanes(wx.Panel):
    def __init__(self, parent, canvas):
        wx.Panel.__init__(self, parent)
        self.canvas = canvas
        self.parent = parent

        self.sizer = wx.GridBagSizer(9, 5)
        self.SetSizer(self.sizer)

        # self.slider_x = RangeSlider(0,1)
        self.sliders = {}
        self.add_sliders("X", 0)
        self.add_sliders("Y", 1)
        self.add_sliders("Z", 2)

        self.button = self.add_button("Clip", (9, 4), (0, 9), self.on_button)
        # self.button = wx.Button(self, label="Clip")
        # self.sizer.Add(self.button, pos=(9, 4), span=(0, 9), flag=wx.EXPAND)
        # self.SetSizer(self.sizer)

        # self.Bind(wx.EVT_BUTTON, self.on_button, self.button)

    def add_sliders(self, axis, pos):
        label = wx.StaticText(
            self, style=wx.ALIGN_CENTRE_HORIZONTAL, label=axis + " clipping sliders"
        )
        self.sliders["label_" + axis] = label
        self.sizer.Add(
            label, pos=(0 + 3 * pos, 0), span=(1, 17), flag=wx.EXPAND, border=3
        )

        slider_min = wx.Slider(self, minValue=0, maxValue=1000, name=(axis + "_min"))
        self.sliders["slider_min_" + axis] = slider_min
        self.sizer.Add(
            slider_min, pos=(1 + 3 * pos, 0), span=(1, 17), flag=wx.EXPAND, border=3
        )

        slider_max = wx.Slider(self, minValue=0, maxValue=1000, name=(axis + "_max"))
        self.sliders["slider_max_" + axis] = slider_max
        slider_max.SetValue(1000)
        self.sizer.Add(
            slider_max, pos=(2 + 3 * pos, 0), span=(1, 17), flag=wx.EXPAND, border=3
        )

        self.Bind(wx.EVT_SCROLL, self.on_move_min(slider_min, slider_max), slider_min)
        self.Bind(wx.EVT_SCROLL, self.on_move_max(slider_max, slider_min), slider_max)

    def on_button(self, event):
        self.canvas.set_clip(
            (
                1 - self.sliders["slider_min_X"].GetValue() / 1000,
                1 - self.sliders["slider_min_Y"].GetValue() / 1000,
                1 - self.sliders["slider_min_Z"].GetValue() / 1000,
            ),
            (
                self.sliders["slider_max_X"].GetValue() / 1000,
                self.sliders["slider_max_Y"].GetValue() / 1000,
                self.sliders["slider_max_Z"].GetValue() / 1000,
            ),
        )

    def on_move_min(self, slider, other):
        def f(event):
            if other.GetValue() <= slider.GetValue():
                if slider.GetValue() > 999:
                    slider.SetValue(999)
                    other.SetValue(1000)
                else:
                    other.SetValue(slider.GetValue() + 1)

        return f

    def on_move_max(self, slider, other):
        def f(event):
            if other.GetValue() >= slider.GetValue():
                if slider.GetValue() < 1:
                    slider.SetValue(1)
                    other.SetValue(0)
                else:
                    other.SetValue(slider.GetValue() - 1)

        return f


    def add_button(self, label, pos, span, funct, button_type=wx.Button):
        button = button_type(self, label=label)
        self.sizer.Add(button, pos=pos, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, span=span)
        if button_type == wx.Button:
            self.Bind(wx.EVT_BUTTON, funct, button)
        if button_type == wx.ToggleButton:
            self.Bind(wx.EVT_TOGGLEBUTTON, funct, button)

        return button