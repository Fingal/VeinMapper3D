from skeleton import *
import wx
import pickle

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
    save={'graph':self.canvas.skeleton_graph,'values':self.values,'coord_data':{'scaling_ratio':self.canvas.scaling_ratio,'shape':self.canvas.points.shape,'unit_to_microns_coef':self.canvas.unit_to_microns_coef,'permuation':(0,2,1)}}
    pickle.dump(save,file)

def on_pressed_mouse(self, event):
    print(event.LeftDown())
    if event.Button(wx.MOUSE_BTN_LEFT):
        self.pos = event.GetPosition()    
        print(self.start_pos)
    event.Skip() 

def connect_points(self,evnt):
    points = self.get_points(2)
    if points:
        self.canvas.connect_points(*points)

def compute_distances_event(self,event):
    saveFileDialog = wx.FileDialog(
        self,
        "Save",
        "",
        "",
        "txt files *.txt",
        wx.FD_SAVE| wx.FD_OVERWRITE_PROMPT,
    )

    if saveFileDialog.ShowModal() == wx.ID_CANCEL:
        return     # the user changed their mind
    path=saveFileDialog.GetPath()
    saveFileDialog.Destroy()
    with open(path,'w') as file:
        for i,(label,typ,positions) in enumerate(self.values):
            #file.write(f"{typ} named {label} distances:\n")
            if typ=='point':
                file.write(f"distance between: {typ} {label} and surfacre is {self.canvas.height_distance(positions)}\n")
            if typ=="line":
                min_distance,max_distance=self.canvas.line_height_distance(*positions)
                file.write(f"closest distance between: {typ} {label} and surfacre is {min_distance}\n")
                file.write(f"furthest distance between: {typ} {label} and surfacre is {max_distance}\n")
            file.write("\n")
            for s_label,s_typ,s_positions in self.values[i+1:]:
                file.write(f"distance between: {typ} {label} and {s_typ} {s_label}:\n")
                if typ=='line':
                    if s_typ=='line':
                        distance=self.canvas.line_line_distance(positions,s_positions)
                        file.write(f"\tclosest distance{distance}\n")
                    if s_typ=='point':
                        distance=self.canvas.point_line_distance(s_positions,positions)
                        file.write(f"\tclosest distance{distance}\n")
                if typ=='point':
                    if s_typ=='line':
                        distance=self.canvas.point_line_distance(positions,s_positions)
                        file.write(f"\tclosest distance{distance}\n")
                    if s_typ=='point':
                        distance=self.canvas.point_distance_L2(s_positions,positions)
                        file.write(f"\tclosest distance{distance}\n")
                        distance=self.canvas.point_distance_skeleton(s_positions,positions)
                        if distance<0:
                            file.write(f"\tpoints not connected\n")
                        else:
                            file.write(f"\tpoints connected, skeleton distance{distance}\n")
            file.write("\n")








if __name__ == "__main__":
    from importlib import reload
    import test_functions as tf
    from types import MethodType
    reload(tf)

if False:
    self.reload_points=MethodType(tf.reload_points, self)
    self.remove_line=MethodType(tf.remove_line, self)

