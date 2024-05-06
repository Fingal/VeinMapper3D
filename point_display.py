import wx
import wx.grid
import wx.lib.scrolledpanel
import copy
import numpy as np
from math import floor, ceil
from importlib import reload
import test_functions as tf
from types import MethodType
from stack_io import *
import pickle
import re
import context


class PointDisplay(wx.Frame):
    def __init__(self,parent):
        wx.Frame.__init__(self, parent, size=(400,800))
        self.context=context.Context()
        self.context.point_display=self
        self.parent = parent
        self.sizer = wx.GridBagSizer(3, 5)
        self.SetSizer(self.sizer)
        class GridPanel(wx.lib.scrolledpanel.ScrolledPanel):
            def __init__(self, parent,on_edit,show_marked):
                wx.Panel.__init__(self, parent,size=(200,100))
                self.sizer = wx.GridBagSizer(9, 5)
                self.SetSizer(self.sizer)
                self.show_points = False
                #self.SetupScrolling()
                self.grid = wx.grid.Grid(self)
                self.grid.CreateGrid(20, 2)
                self.grid.SetColLabelValue(0, "Label")
                self.grid.SetColLabelValue(1, "Positions")
                self.grid.SetColSize(0, 80)
                self.grid.SetColSize(1, 230)
                self.grid.SetRowLabelSize(40)
                self.grid.SetSelectionMode(1)
                self.grid.Bind(wx.grid.EVT_GRID_CELL_CHANGING, on_edit)
                self.grid.GetGridWindow().Bind(wx.EVT_LEFT_UP, show_marked)
                self.sizer.Add(
                    self.grid, pos=(0, 0), span=(30, 9), flag=wx.EXPAND, border=3
                )


        self.pages = wx.Notebook(self,size=(200,700))
        self.pages.parent=self

        self.point_panel = GridPanel(self.pages,self.on_edit('point'),parent.show_marked)
        self.lines_panel = GridPanel(self.pages,self.on_edit('line'),parent.show_marked)
        self.label_panel = GridPanel(self.pages,self.on_edit('label'),parent.show_marked)
        self.grids={}
        self.grids['point'] = self.point_panel.grid
        self.grids['line'] = self.lines_panel.grid
        self.grids['label'] = self.label_panel.grid

        self.pages.AddPage(
            self.point_panel,
            "Points",
        )
        self.pages.AddPage(
            self.lines_panel,
            "Lines",
        )

        self.pages.AddPage(
            self.label_panel,
            "Labels",
        )

        self.sizer.Add(self.pages, pos=(0, 0), flag=wx.EXPAND, span=(5, 25))
        
        self.button1 = wx.Button(self, label="Sort")
        self.button1.Bind(wx.EVT_BUTTON,self.sort)
        self.sizer.Add(self.button1, pos=(5, 0), span=(0, 12), flag=wx.EXPAND)

        self.button2 = wx.Button(self, label="Remove selected points")
        self.button2.Bind(wx.EVT_BUTTON,self.remove)
        self.sizer.Add(self.button2, pos=(5, 13), span=(0, 12), flag=wx.EXPAND)

        self.button3 = wx.Button(self, label="Add to workspace")
        self.button3.Bind(wx.EVT_BUTTON,self.add_to_workspace)
        self.sizer.Add(self.button3, pos=(6, 0), span=(0, 12), flag=wx.EXPAND)

        self.button4 = wx.Button(self, label="Add label")
        self.button4.Bind(wx.EVT_BUTTON,self.add_label)
        self.sizer.Add(self.button4, pos=(7, 0), span=(0, 12), flag=wx.EXPAND)
        

        self.button5 = wx.Button(self, label="Reload points")
        self.button5.Bind(wx.EVT_BUTTON,self.reload_points)
        self.sizer.Add(self.button5, pos=(6, 13), span=(0, 12), flag=wx.EXPAND)
        self.Show()
        

    # def reload_all(self):
    #     if self.reload_other:
    #         self.reload_other()
    #     self.reload_inner()
    #     self.canvas.load_labels(self.values['label'])
            

    
    def reload_inner(self):
        self.reload_grid('point')
        self.reload_grid('line')
        self.reload_grid('label')


    def sort(self,event):
        #self.values['point']=[(f'p_{1+i//5}_{i%5}',(i*3/7,i**2/4-0.2,i**0.1)) for i in range(100)]
        self.context.data_context.sort()
        self.context.data_context.reload_all()
        #import pdb; pdb.set_trace()

    def get_selected(self):
        if self.pages.GetSelection()==0:
            selected = [self.context.data_context.values['point'][i] for i in self.grids['point'].GetSelectedRows() if i<len(self.context.data_context.values['point'])]
            type='point'
        if self.pages.GetSelection()==1:
            selected = [self.context.data_context.values['line'][i] for i in self.grids['line'].GetSelectedRows() if i<len(self.context.data_context.values['line'])]
            type='line'
        if self.pages.GetSelection()==2:
            selected = [self.context.data_context.values['label'][i] for i in self.grids['label'].GetSelectedRows() if i<len(self.context.data_context.values['label'])]
            type='label'
        return selected,type
        

    # def remove(self,event):
    #     if self.pages.GetSelection()==0:
    #         selected,_ = self.get_selected()
    #         for p in selected:
    #             self.values['point'].remove(p)
    #             self.canvas.remove_point(p[1])
    #         self.parent.main.position_setter.remove_points([p for _,p in selected])
    #     if self.pages.GetSelection()==2:
    #         selected,_ = self.get_selected()
    #         for p in selected:
    #             self.values['label'].remove(p)
    #     self.context.data_context.reload_all()
    def remove(self,evnet):
        selected,typ=self.get_selected()
        self.context.data_context.remove(selected,typ)
        self.grids[typ].ClearSelection()
            
    

    def add_to_workspace(self,event):
        selected,_=self.get_selected()
        self.context.data_context.show_data(set(selected))

    def add_label(self,event):
        selected,_=self.get_selected()
        for s in selected:
            self.context.data_context.add_label(s)


    def reload_grid(self,type):
        grid=self.grids[type]
        # if grid.GetNumberRows():
        #     grid.DeleteRows(0, grid.GetNumberRows())
        new_size=max(len(self.context.data_context.values[type])+1, 20)
        if grid.GetNumberRows()==new_size:
            pass
        elif grid.GetNumberRows()<new_size:
            grid.AppendRows(new_size-grid.GetNumberRows())
        else:
            grid.DeleteRows(new_size, grid.GetNumberRows()-new_size)
        index=0
        for i,positions in enumerate(self.context.data_context.values[type]):
            #print(i,positions)
            try:
                label=self.context.data_context.labels[type][positions]
            except:
                print(type,positions,"ERROR")
            positions_string=""
            if type=='point' or type=='label':
                positions_string = "{:},{:},{:}".format(*positions)
            else:
                for position in positions:
                    positions_string = positions_string + "({:},{:},{:}) ".format(
                        *position
                    )
            grid.SetCellValue(i, 0, label)
            grid.SetCellValue(i, 1, positions_string)
            index=i
        for j in range(index+1,new_size):
            grid.SetCellValue(j, 0, '')
            grid.SetCellValue(j, 1, '')

        grid.SetSize(grid.Parent.GetSize())

        #self.show_marked('')
        

    def on_edit(self,type):
        # import pdb; pdb.set_trace()
        def g(evnt):
            self.context.data_context.reload_all()
            if evnt.GetCol() == 0 and evnt.GetRow() < len(self.context.data_context.values[type]):
                i=evnt.GetRow()
                points = self.context.data_context.values[type][i]
                self.context.data_context.set_label(type,points,evnt.GetString())
                self.context.data_context.reload_all()
            else:
                evnt.Veto()
        return g

    # def reload_points(self,evnt):
    #     loaded_points=[point for _,point in self.context.data_context.values['point']]
    #     self.context.data_context.hide_data(set(self.data_context.skeleton_graph.keys())-set(loaded_points))
    #     self.context.data_context.reload_all()
    def reload_points(self,evnt):
        self.reload_inner()
        self.context.data_context.reload_from_skeleton()




if __name__ == "__main__":
    class MyApp(wx.App):
        def OnInit(self):
            frame = PointDisplay(None)
            return True
    app = MyApp()
    app.MainLoop()
    

