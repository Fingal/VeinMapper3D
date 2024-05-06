import wx
from wx import glcanvas
from wx.lib import sized_controls

from OpenGL.GL import *
import OpenGL.GL.shaders
from pyrr import Matrix44, matrix44, Vector3, Vector4, vector3
import time, sys
from Cube import *
from stack_io import *
import numpy as np
import shaders as sh
import mockup
import time
import copy

from scipy import ndimage
from skimage.morphology import skeletonize_3d
from skimage.transform import downscale_local_mean
from skeleton import *
from context import Context

TEST = False




class SkeletonShaderBundle:
    def __init__(self, lines):
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(sh.vertex_skeleton, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(sh.fragment_skeleton, GL_FRAGMENT_SHADER),
        )
        
        self.smooth_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(sh.smooth_skeleton['vertex'], GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(sh.smooth_skeleton['tess_control'], GL_TESS_CONTROL_SHADER),
            OpenGL.GL.shaders.compileShader(sh.smooth_skeleton['tess_eval'], GL_TESS_EVALUATION_SHADER),
            OpenGL.GL.shaders.compileShader(sh.smooth_skeleton['fragment'], GL_FRAGMENT_SHADER),
        )

        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)

        self.lines = lines
        # arr = cube
        self.vao_lines = glGenVertexArrays(1)
        glBindVertexArray(self.vao_lines)
        self.vbo_lines, self.vbo_lines_color,self.vbo_lines_tangent = glGenBuffers(3)
        self.size = len(self.lines) * 2
        self.line_colors = [(0, 0, 0) for i in range(self.size)]
        self.line_tangents = [(0, 0, 0) for i in range(self.size)]
        self.init_lines()

        self.vao_points = glGenVertexArrays(1)
        glBindVertexArray(self.vao_points)
        self.vbo_points, self.vbo_points_color = glGenBuffers(2)
        self.marked_points = []
        # self.marked_points=[]

    def init_lines(self):
        self._init_bufers(
            self.vao_lines,
            self.vbo_lines,
            self.vbo_lines_color,
            np.array(self.lines, dtype=np.float32),
            np.array(self.line_colors, dtype=np.float32),
            self.size
        )
        self._init_tangent_buffer(self.size,np.array(self.line_tangents, dtype=np.float32))

    def _init_bufers(self, vao, vbo_points, vbo_colors, data_lines, data_color, size):
        """Initialised buffer data"""
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_points)
        glBufferData(
            GL_ARRAY_BUFFER, size * 3 * 4, data_lines.tobytes(), GL_STATIC_DRAW
        )
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glBufferData(
            GL_ARRAY_BUFFER, size * 3 * 4, data_color.tobytes(), GL_STATIC_DRAW
        )
        # glBufferData(GL_ARRAY_BUFFER, self.size * 3 * 4, colors.tobytes(), GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def _init_tangent_buffer(self,size,data):
        glBindVertexArray(self.vao_lines)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines_tangent)
        glBufferData(
            GL_ARRAY_BUFFER, size * 3 * 4, data.tobytes(), GL_STATIC_DRAW
        )
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))


        # vertex attribute pointers

    def set_lines(self, lines,tangents):
        """Lines points that are showed"""
        self.lines = lines
        glBindVertexArray(self.vao_lines)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        self.size = len(self.lines) * 2
        self.line_colors = [(0, 0, 0) for i in range(self.size)]
        self.line_tangents=tangents
        self.init_lines()

    def mark_points(self, points):
        """Sets points that are showed"""
        self.marked_points = points
        self.points_size = len(points)
        #print(self.marked_points)
        self._init_bufers(
            self.vao_points,    
            self.vbo_points,
            self.vbo_points_color,
            np.array(self.marked_points, dtype=np.float32),
            np.array([(1, 0, 1)] * self.points_size, dtype=np.float32),
            self.points_size
        )

    def mark_lines(self, lines):
        """Sets points that are marked"""
        self.marked_lines = lines
        # self.line_colors=[((1,0,0),(1,0,0)) if (x,y) in self.marked_lines or (y,x) in self.lines else ((0,0,0),(0,0,0)) for (x,y) in self.lines]
        self.line_colors = []
        #print(self.lines)
        for x, y in self.lines:
            if (x, y) in self.marked_lines or (y, x) in self.marked_lines:
                self.line_colors.append(((1, 0, 0), (1, 0, 0)))
            else:
                self.line_colors.append(((0, 0, 0), (0, 0, 0)))
        self.init_lines()

    def set_lines_color(self,lines,color):
        def change_color(line):
            index=self.lines.index(line)
            self.line_colors[index]=(color,color)
        not_found_times=0
        for (x,y) in lines:
            not_found=0
            try:
                change_color((x,y))
                continue
            except:
                not_found+=1
            try:
                change_color((y,x))
                continue
            except:
                not_found+=1
        #     if not_found==2:
        #         print('not found',(x,y))
        #         not_found_times+=1
        # print('not found lines', not_found_times)
        self.init_lines()

    def remove_lines(self, lines):
        for line in lines:
            a, b = line
            if (a, b) in self.lines:
                self.lines.remove((a, b))
            elif (b, a) in self.lines:
                self.lines.remove((b, a))
            else:
                print((a, b), "not found")


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent):
        self.size = (800, 5000)
        self.aspect_ratio = self.size[0] / self.size[1]
        self.context=Context()
        self.context.canvas=self
        # glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        self._context = glcanvas.GLContext(self)
        self.SetCurrent(self._context)
        self.init = False
        self.rotate = False
        self.rot = Matrix44.identity()
        self.rot_loc = None
        self.trans_loc = None
        self.trans_x, self.trans_y, self.trans_z = 0.0, 0.0, -2.0
        self.translate = Matrix44.identity()
        self.combined_matrix = Matrix44.identity()
        self.camera_position = Vector3([0.0, 0.0, 2.0])
        self.concentration_loaded = False
        self.moving_pressed = False
        self.is_lines_initialized = False
        self.skeleton_bundle = None
        self.real_distance = None
        self.line_width = 2
        self.point_size = 8
        self.labels=[]
        # self.vol = np.zeros((100,100,100,4), np.uint8)
        # self.vol[:10] += 10

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)
        self.counter = 0
        self.time = time.time()
        self.draw_text=True
        self._draw_text=False

        self.min_clip = (1.0, 1.0, 1.0)
        self.max_clip = (1.0, 1.0, 1.0)
        self.skeleton_graph={}

        self.dr_gradient = np.array(
            [(0, 255, 20, max(0, i // 10 - 3)) for i in range(255)]
        ).astype(np.uint8)
        self.pi_gradient = np.array(
            [(255, 255, 255, min(2, max(0, i // 10))) for i in range(255)]
        ).astype(np.uint8)
        self.if_gradient_initialised = False
        self.max_concentration = 1

        self.old_graph = set()
        self.modes = {}
        self._blur_cache=None
        self.dc=None
        self.scaling_ratio=None
        self.InitGL()

        if TEST:
            self.test_list = []
            self.init_concentration(
                "I:\\symulacje\\stacks_vascular_1\\stacks_vascular_1\\xp5_dr3\\dr3_1_dr.tif",
                "I:\\symulacje\\stacks_vascular_1\\stacks_vascular_1\\xp5_dr3\\dr3_1_pi.tif",
                downscale=False,
            )
            # self.init_concentration("D:\\symulacje\\stacks_vascular_1\\stacks_vascular_1\\xp5_dr3\\1-5_dr.tif","D:\\symulacje\\stacks_vascular_1\\stacks_vascular_1\\xp5_dr3\\1-5_pi.tif", downscale=True)
            # self.load_skeleton_from_file(
            #     "I:\\symulacje\\stacks_vascular_1\\stacks_vascular_1\\xp5_dr3\\skeleton.tif"
            # )

            # self.prune_nodes(90)
            # self.prune_nodes(2)
            # self.skeleton_to_lines()

    def _flatten_skeleton(self):
        new_skeleton={(k[0],k[1],100):{(i[0],i[1],100) for i in v} for k,v in self.skeleton_graph.items()}
        self.reload_skeleton_graph(new_skeleton)

    def mark_lines(self, points):
        lines = []
        for start, end in points:
            lines = lines + list((self._lines_coord_to_opengl(find_primary(self.skeleton_graph, start, end))))
        # print(lines)
        self.skeleton_bundle.mark_lines(lines)
        self.Refresh()

    def get_equal_points(self,start,end,distance):
        lines = find_primary(self.skeleton_graph, start, end)
        result = []
        temp_distance=distance
        for line in lines:
            length = self.point_distance_L2(*line)
            if length<temp_distance:
                temp_distance=temp_distance-length
            else:
                ratio = temp_distance/length
                result.append(tuple(a*(1-ratio)+b*ratio for a,b in zip(*line)))
                temp_distance = temp_distance+distance
                while temp_distance<length:
                    ratio = temp_distance/length
                    result.append(tuple(a*(1-ratio)+b*ratio for a,b in zip(*line)))
                    temp_distance = temp_distance+distance
                temp_distance=temp_distance-length
        return result
        
    
    def reload_skeleton_graph(self,graph):
        self.skeleton_graph=graph
        self.redraw_graph()
        
    def redraw_graph(self):
        self.context.graphic_context.set_graph_lines(*self.graph_to_lines())
        self.Refresh()

    def get_selected_lines(self, points):
        lines = []
        for start, end in points:
            lines = lines + (find_primary(self.skeleton_graph, start, end))
        return self._lines_coord_to_opengl(lines)
    
    def mark_stack_lines(self, points):
        lines = []
        for start, end in points:
            lines = lines + (find_primary(self.skeleton_graph, start, end))
        # print(lines)
        self.skeleton_bundle.mark_lines(self._lines_coord_to_opengl(lines))
        self.Refresh()

    def mark_points(self, points):
        self.skeleton_bundle.mark_points(points)
        self.Refresh()

    def add_point(self,point):
        if point not in self.skeleton_graph:
            self.skeleton_graph[point]=set()

    def add_junction(self,line,point):
        #print('shortest path',shortest_path(self.skeleton_graph, *line))
        nearest=sorted(shortest_path(self.skeleton_graph, *line),key=lambda x: self.point_distance_L2(point,x))
        #print('sorted',nearest)
        nearest=sorted(shortest_path(self.skeleton_graph, *line),key=lambda x: self.point_distance_L2(point,x))[:2]
        #print(self.point_distance_L2(point,nearest[0]))
        if self.point_distance_L2(point,nearest[0])<1:
            return nearest[0]
        else:
            self.skeleton_graph[point]=set(nearest)
            self.skeleton_graph[nearest[0]].add(point)
            self.skeleton_graph[nearest[1]].add(point)
            self.skeleton_graph[nearest[0]].remove(nearest[1])
            self.skeleton_graph[nearest[1]].remove(nearest[0])

            self.redraw_graph()
            #print(self.skeleton_graph[point])
            #print(self.skeleton_graph)
            return point

    def project_point_line(self,point,line):
        start,end=line
        for key in self.skeleton_graph[point]:
            self.skeleton_graph[key].remove(point)
        a = np.array(point) - np.array(start)
        b = np.array(end) - np.array(start)
        c = np.array(point) - np.array(end)
        proj_len = np.dot(a, b) / np.linalg.norm(b)
        if 2 < proj_len < np.linalg.norm(b)-2:
            coef=proj_len/np.linalg.norm(b)
            new_point=tuple(start*(1-coef)+end*coef for start,end in zip(*line))

            self.skeleton_graph[new_point]=self.skeleton_graph[point]
            self.skeleton_graph[new_point].add(start)
            self.skeleton_graph[new_point].add(end)

            del self.skeleton_graph[point]


            self.skeleton_graph[start].add(new_point)
            self.skeleton_graph[end].add(new_point)

            self.skeleton_graph[start].remove(end)
            self.skeleton_graph[end].remove(start)

            self.redraw_graph()
            self.Refresh()
            return new_point
        else:
            if 2 > proj_len:
                new_point = start
            elif proj_len > np.linalg.norm(b)-2:
                new_point=end
            self.skeleton_graph[new_point]=self.skeleton_graph[new_point]|self.skeleton_graph[point]
            del self.skeleton_graph[point]
            self.redraw_graph()
            self.Refresh()




    def mark_stack_points(self,points):
        points=list(map(self._stack_coord_to_opengl,points))
        self.mark_points(points)

    def _stack_coord_to_opengl(self,point):
        p=point
        scaling_ratio=(self.scaling_ratio[1],self.scaling_ratio[0],self.scaling_ratio[2])
        point = tuple(
            (p * scale) / shape - 0.5 * scale
            for p, scale, shape in zip(
                point, scaling_ratio, self.context.data_context.dr_stack_shape
            )
        )
        return (point[1]*self.unit_to_microns_coef,point[2]*self.unit_to_microns_coef,point[0]*self.unit_to_microns_coef)

    def _stack_coord_to_micron(self,point):
        p=point
        scaling_ratio=(self.scaling_ratio[1],self.scaling_ratio[0],self.scaling_ratio[2])
        point = tuple(
            (p * scale) / shape - 0.5 * scale
            for p, scale, shape in zip(
                point, scaling_ratio, self.context.data_context.dr_stack_shape
            )
        )
        return (point[1]*self.unit_to_microns_coef,point[2]*self.unit_to_microns_coef,point[0]*self.unit_to_microns_coef)

    def _micron_coord_to_stack(self,point):
        _p=tuple(i/self.unit_to_microns_coef for i in point)
        return self._opengl_coord_to_stack(_p)

    def _opengl_coord_to_stack(self,point):
        point=(point[2],point[0],point[1])
        scaling_ratio=(self.scaling_ratio[1],self.scaling_ratio[0],self.scaling_ratio[2])
        point = tuple(
            (p / scale) * shape + 0.5 * shape
            for p, scale, shape in zip(
                point, scaling_ratio, self.context.data_context.dr_stack_shape
            )
        )
        return point

    def _lines_coord_to_opengl(self,lines):
        return [tuple(map(self._stack_coord_to_opengl,line)) for line in lines]

    def _lines_coord_to_microns(self,lines):
        return [tuple(map(self._stack_coord_to_micron,line)) for line in lines]

    def remove_secondary_lines(self, start, end):
        """binding for rmeoving secondary lines"""
        to_remove = find_secondary(self.skeleton_graph, start, end)
        print("removing")
        self.remove_from_graph(self.skeleton_graph, to_remove)
        self.redraw_graph()
        print("done")
        self.Refresh()

    def remove_to_junction(self, point):
        """binding for rmeoving line up to junction"""
        to_remove = paths_to_junction(self.skeleton_graph, point)
        self.remove_from_graph(self.skeleton_graph, to_remove)
        self.redraw_graph()

    def remove_point(self,point):
        for neighbour in self.skeleton_graph[point]:
            self.skeleton_graph[neighbour].remove(point)
        
        del self.skeleton_graph[point]

    def remove_line(self, start, end):
        """binding for rmeoving line from start to end"""
        to_remove = find_primary(self.skeleton_graph, start, end)
        print("removing")
        self.remove_from_graph(self.skeleton_graph, to_remove)
        self.redraw_graph()
        print("done")
        self.Refresh()

    def connect_points(self, start, end):
        for node in (start,end):
            if node not in self.skeleton_graph:
                self.skeleton_graph[node]=set()
        if (
                start not in self.skeleton_graph[end]
                or end not in self.skeleton_graph[start]
            ):
                self.skeleton_graph[end].add(start)
                self.skeleton_graph[start].add(end)
        else:
            return False
        
        self.redraw_graph()

        self.Refresh()
        return True

    def undo(self):
        """undo last graph modification"""
        self.load_skeleton_graph(self.old_graph)

    def load_skeleton_graph(self,graph):
        self.skeleton_graph = graph
        self.redraw_graph()

    def _graph_to_lines(self):
        graph={self._stack_coord_to_opengl(key):set(map(self._stack_coord_to_opengl,value)) for key,value in self.skeleton_graph.items()}
        return graph_to_lines(graph)

    def graph_to_lines(self):
        def p_fun(t,a,b,m,i):
            p = (2*t**3 - 3*t**2+1) *a + (t**3 - 2*t**2+t)*m[i]+((-2)*t**3 + 3*t**2)*b+(t**3 - t**2)*m[i+1]
            return p
        graph={self._stack_coord_to_opengl(key):set(map(self._stack_coord_to_opengl,value)) for key,value in self.skeleton_graph.items()}
        paths=get_all_paths(graph)
        lines=[]
        tangents=[]
        for path in paths:
            m=[]
            points = [path[0]]+path+[path[-1]]
            for a,b in zip(map(np.array,points[:-1]),map(np.array,points[2:])):
                m.append(tuple((self.context.graphic_context.curve_tension)*(b-a)*(2)))
            for (a, b),(m1,m2) in zip(zip(path[:-1], path[1:]),zip(m[:-1], m[1:]),):
                lines.append((a,b))
                tangents.append((m1,m2))
        #print(lines,tangents)
        return lines,tangents



    def points_angles(self,_a,_b,_c):
        a=self._stack_coord_to_opengl(_a)
        b=self._stack_coord_to_opengl(_b)
        c=self._stack_coord_to_opengl(_c)
        def normalize(a):
            return a/np.linalg.norm(a)
        result = [np.arccos(np.dot(normalize(x),normalize(y)))/np.pi*180 for x,y in [((c-a),(b-a)),((a-b),(c-b)),((a-c),(b-c))] ]
        return result

    # def point_from_skeleton_distance_L2(self,point,skeleton):
    #     lines = graph_to_lines(skeleton)
    #     distance = self.point_line_distance(point,lines[0])
    #     for line in lines:
    #         distance=min(distance,self.point_line_distance(point,line))
    #     return distance
    

    def point_distance_L2(self, _start, _end):
        start=self._stack_coord_to_opengl(_start)
        end=self._stack_coord_to_opengl(_end)
        distance = (
            sum(((x - y)* self.unit_to_microns_coef) ** 2 for x, y in zip(start, end)) ** 0.5
        )
        return distance

    def point_distance_coords_L2(self, _start, _end):
        start=self._stack_coord_to_opengl(_start)
        end=self._stack_coord_to_opengl(_end)
        distances = [abs(x - y)* self.unit_to_microns_coef for x, y in zip(start, end)]
        return (*distances,self.point_distance_L2(_start,_end))
    

    def point_distance_skeleton(self, start, end):
        points = shortest_path(self.skeleton_graph, start, end)
        distance = [0,0,0,0]
        for a, b in zip(points[:-1], points[1:]):
            a=self._stack_coord_to_opengl(a)
            b=self._stack_coord_to_opengl(b)
            for i,(x, y) in enumerate(zip(a, b)):
                distance[i] += abs(x - y)
            distance[3] += sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        distance=list(map(lambda x: x*self.unit_to_microns_coef,distance))
        return distance

    def point_line_distance(self, point, line):
        point=self._stack_coord_to_opengl(point)
        line = self._lines_coord_to_opengl(find_primary(self.skeleton_graph, *line))
        distance = 9999
        for start, end in line:
            a = np.array(point) - np.array(start)
            b = np.array(end) - np.array(start)
            c = np.array(point) - np.array(end)
            proj_len = np.dot(a, b) / np.linalg.norm(b)
            if 0 < proj_len < np.linalg.norm(b):
                new_distance = (np.dot(a, a) - np.dot(a, b) ** 2 / np.dot(b, b)) ** 0.5
            else:
                new_distance = min(np.linalg.norm(a), np.linalg.norm(c))
            distance=min(distance,new_distance)
        return distance * self.unit_to_microns_coef

            
        

    def line_line_distance(self, first, second):
        first = self._lines_coord_to_opengl(find_primary(self.skeleton_graph, *first))
        second = self._lines_coord_to_opengl(find_primary(self.skeleton_graph, *second))
        distance = 999
        for a, b in first:
            for c, d in second:
                new_distance = closestDistanceBetweenLines(
                    np.array(a), np.array(b), np.array(c), np.array(d), clampAll=True
                )[-1]
                if distance > new_distance:
                    distance = new_distance
        return distance * self.unit_to_microns_coef

    def height_distance(self, coord):
        if type(coord) != type(None):
            distance = (
                (
                    abs(self.data_context.heightmap[int(coord[0]), int(coord[1])] - coord[2])
                    * self.scaling_ratio[2]
                )
                / self.shape[2]
                * self.unit_to_microns_coef
            )
            return distance
        return -1

    def line_height_distance(self, start, end):
        path = shortest_path(self.skeleton_graph, start, end)
        min_distance = 999
        max_distance = -1
        for point in path:
            distance = self.height_distance(point)
            min_distance = min(min_distance, distance)
            max_distance = max(max_distance, distance)
        return min_distance, max_distance

    def calculate_ray(self, x, y):
        """calculate ray based on relative screen position"""
        # print(x,y)
        start = Vector4([x, y, -1, 1])
        end = Vector4([x, y, 0, 1])
        self.camera_position
        view = matrix44.create_from_translation(-self.camera_position)
        projection = matrix44.create_perspective_projection_matrix(
            45.0, self.aspect_ratio, 0.1, 100.0
        )
        self.translate = matrix44.create_from_translation(Vector3([0, 0, 0]))
        vp = matrix44.multiply(view, projection)
        start = Vector4(
            np.array(start)
            @ np.array(matrix44.inverse(vp))
            @ np.array(matrix44.inverse(self.rot))
        )
        end = Vector4(
            np.array(end)
            @ np.array(matrix44.inverse(vp))
            @ np.array(matrix44.inverse(self.rot))
        )

        start = Vector3([start.x / start.w, start.y / start.w, start.z / start.w])
        end = Vector3([end.x / end.w, end.y / end.w, end.z / end.w])
        return start, end

    def remove_from_graph(self, graph, to_remove):
        self.old_graph = copy.deepcopy(graph)
        for start, end in to_remove:
            try:
                graph[start].remove(end)
                graph[end].remove(start)
            except:
                print(f"{(start,end)} doesn't exist")
        return graph

    def OnMouse(self, event):
        # if(time.time()-self.time<1):
        #     self.counter+=1
        # else:
        #     print(self.counter)
        #     self.time=time.time()
        #     self.counter=0
        if event.Button(wx.MOUSE_BTN_LEFT):
            pass
            # print('button')
        if event.ButtonDown(wx.MOUSE_BTN_LEFT):
            # calculating click ray
            x = event.GetLogicalPosition(self.dc).x / self.dc.GetSize().x * 2 - 1
            y = event.GetLogicalPosition(self.dc).y / self.dc.GetSize().y * 2 - 1
            y = -y
            start, end = self.calculate_ray(x, y)
            point_distance = lambda x: (start - Vector3(self._stack_coord_to_opengl(x))).length
            def line_distance(_point): 
                point=Vector3(self._stack_coord_to_opengl(_point))
                a = point-start
                b = end-start
                c = point-end
                proj_len = (a | b) / b.length
                new_distance = ((a | a) - proj_len**2) ** 0.5
                return new_distance
            a, b = self.get_clip_borders()
            in_clip_space = lambda x: all(np.array(a) <= np.array(x)) and all(
                np.array(x) <= np.array(b)
            )

            test_option = self.modes.get("test", 0)
            if self.modes.get("find_junction"):
                r = list(
                    filter(
                        lambda x: line_distance(x) < 0.004 and in_clip_space(x),
                        self.joints,
                    )
                )
                if len(r) > 1:
                    r.sort(key=point_distance)
                    if abs(point_distance(r[0]) - point_distance(r[1])) < 0.008:
                        rr = filter(
                            lambda x: abs(point_distance(r[0]) - point_distance(x))
                            < 0.008,
                            r,
                        )
                        result = min(rr, key=line_distance)
                    else:
                        result = r[0]
                elif len(r) == 1:
                    result = r[0]
                else:
                    return

                self.modes.get("find_junction")(result, "J")
                del self.modes["find_junction"]
            if self.modes.get("find_any_point"):
                # r=list(filter(lambda x: line_distance(x)<0.004 and len(self.skeleton_graph[x])<3, self.skeleton_graph.keys()))
                r = list(
                    filter(
                        lambda x: line_distance(x) < 0.02,
                        self.skeleton_graph.keys(),
                    )
                )
                print(r)
                if len(r) > 1:
                    r.sort(key=line_distance)
                    if abs(point_distance(r[0]) - point_distance(r[1])) < 0.001:
                        rr = filter(
                            lambda x: abs(point_distance(r[0]) - point_distance(x))
                            < 0.008,
                            r,
                        )
                        result = min(rr, key=point_distance)
                    else:
                        result=r[0]
                elif len(r) == 1:
                    result = r[0]
                elif len(r) == 0:
                    return

                self.modes.get("find_any_point")(result)
                del self.modes["find_any_point"]
            if test_option > 0:
                if test_option == 1:
                    y = time.time()
                    r = list(
                        filter(
                            lambda x: line_distance(x) < 0.004
                            and len(self.skeleton_graph[x]) < 3,
                            self.skeleton_graph.keys(),
                        )
                    )
                    if len(r) > 1:
                        r.sort(key=point_distance)
                        if abs(point_distance(r[0]) - point_distance(r[1])) < 0.008:
                            rr = filter(
                                lambda x: abs(point_distance(r[0]) - point_distance(x))
                                < 0.008,
                                r,
                            )
                            result = min(rr, key=line_distance)
                    elif len(r) == 1:
                        result = r[0]
                    elif len(r) == 0:
                        return
                    v = time.time() - y
                    print(f"all nodes filter taken:{v} seconds")
                    print(f"recieved nodes {list(r)}")
                    to_remove = paths_to_junction(self.skeleton_graph, result)
                    remove_from_graph(self.skeleton_graph, to_remove)
                    self.redraw_graph()
                    print("done")
                self.Refresh()
            else:
                self.time = time.time()
                self.mouse_pos = event.GetLogicalPosition(self.dc)
                self.moving_pressed = True
            # print('button down')
        if event.ButtonUp(wx.MOUSE_BTN_LEFT) or event.Leaving():
            self.mouse_pos = event.GetLogicalPosition(self.dc)
            self.moving_pressed = False
            self.counter = 0
            self.Refresh()
            # print('button up')
        if event.Dragging() and self.moving_pressed:
            new_y = event.GetLogicalPosition(self.dc).y
            new_x = event.GetLogicalPosition(self.dc).x
            self.rot = (
                Matrix44.from_y_rotation((self.mouse_pos.x - new_x) / 100)
                * Matrix44.from_x_rotation((self.mouse_pos.y - new_y) / 100)
                * self.rot
            )
            self.mouse_pos = event.GetLogicalPosition(self.dc)
            self.counter += 1

        if (event.GetWheelRotation()) != 0:
            # print(event.GetWheelRotation())
            self.camera_position.z = max(0.01,self.camera_position.z - 0.001 * event.GetWheelRotation())
            self.context.graphic_context.set_zoom_scale_field(1/self.camera_position.z)
            self.Refresh()
            # print('draging')
        if self.counter >= 1:
            self.counter = 0
            self.Refresh()
        # import pdb; pdb.set_trace()

    def OnResize(self, event):
        size = self.GetClientSize()
        glViewport(0, 0, size.width, size.height)

        self.aspect_ratio = size.width / size.height

    def OnPaint(self, event):
        self.dc = wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.draw_clear()
        else:
            if not self.concentration_loaded and self.scaling_ratio==None:
                self.draw_clear()
            elif self._draw_text and self.draw_text:
                self.dc = wx.PaintDC(self)
                self._draw_labels()
                self._draw_text=False
            elif self.scaling_ratio!=None and not self.concentration_loaded:
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.OnSkeletonDraw()
                self.SwapBuffers()
                self.dc = wx.PaintDC(self)
                self._draw_labels()
                self._draw_text=True
            else:
                self.OnDraw()
                self._draw_text=True

    

    def _draw_labels(self):
        for label,point in self.context.data_context.get_labels():
            if label:
                pos = Vector4([*self._stack_coord_to_opengl(point), 1])
                view = matrix44.create_from_translation(-self.camera_position)
                projection = matrix44.create_perspective_projection_matrix(
                    45.0, self.aspect_ratio, 0.1, 100.0
                )
                vp = matrix44.multiply(view, projection)
                pos=pos@self.rot@vp
                pos = Vector3([pos[0] / pos[3], pos[1] / pos[3], pos[2] / pos[3]])
                x,y=pos.x,pos.y
                x=int((x+1)/2*self.dc.GetSize().x)
                y=int((-y+1)/2*self.dc.GetSize().y)
                #print(x,y)
                self.dc.DrawText(label,x,y)


    def draw_clear(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.SwapBuffers()

    def _init_3d_texture(self, loc, _data, enum, name, num, downscale=False):
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glActiveTexture(enum)
        glBindTexture(GL_TEXTURE_3D, loc)

        if downscale:
            data = downscale_local_mean(_data, (2, 2, 2)).astype(np.uint8)
        else:
            data = _data

        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP)

        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_ALPHA,
            data.shape[2],
            data.shape[1],
            data.shape[0],
            0,
            GL_ALPHA,
            GL_UNSIGNED_BYTE,
            data.tobytes(),
        )

        # Init the texture units
        glActiveTexture(enum)
        glBindTexture(GL_TEXTURE_3D, loc)
        # Setup VolumeData on Texture Unit 2 (we are using 1 for TransferFunction)
        glUniform1i(glGetUniformLocation(self.concentration_shader, name), num)
        print(f"loaded {name}")



    def _load_concentration_from_file(self, directory,calculate_scale=False):
        img = Image.open(directory)

        points = np.maximum(0, image_to_array(img)[:, :, :])  # [:,:,:100]
        if calculate_scale:
            try:
                scale = get_dimentions(img)
            except:
                dlg = CustomDialog(None, title='Set scales')
                result = dlg.ShowModal()
                print(dlg.get_scale())
                scale=dlg.get_scale()
                dlg.Destroy()
            self.voxel_dimentions=scale
            self._point_scale=tuple((s/min(scale) for s in scale))
            relative_scale = tuple(
                scale[i] / max(points.shape) * points.shape[i] for i in range(3)
            )
            relative_scale = tuple(x / max(relative_scale) for x in relative_scale)
            if self.real_distance == None:
                temp = [x * z / y for x, y, z in zip(scale, relative_scale, points.shape)]
                assert abs(max(temp) - min(temp)) < 0.0001, "Stack size cant be determined"
                self.unit_to_microns_coef = sum(temp) / 3
            relative_scale=(relative_scale[1],relative_scale[0],relative_scale[2])
            return (points, relative_scale)
        else:
            return (points, None)

    def load_concentration(self, dr_directory, pi_directory):
        img = Image.open(dr_directory)
        self.context.data_context.dr_stack, self.scaling_ratio = self._load_concentration_from_file(
            dr_directory,calculate_scale=True
        )
        self.context.data_context.dr_stack_shape=self.context.data_context.dr_stack.shape
        self.max_concentration = np.amax(self.context.data_context.dr_stack)
        self.concentration = (self.context.data_context.dr_stack / np.max(self.context.data_context.dr_stack) * 255).astype(np.uint8)
        self.context.data_context.pi_stack, _ = self._load_concentration_from_file(pi_directory)
        self.pi_concentration = (self.context.data_context.pi_stack / np.max(self.context.data_context.pi_stack) * 255).astype(
            np.uint8
        )

        self.concentration_loaded = True
        self.is_skeleton_loaded = 0

    def init_concentration(self, dr_directory, pi_directory, downscale=False):
        self.directories = (dr_directory, pi_directory)
        self.load_concentration(dr_directory, pi_directory)
        # TODO uncoment later
        self.init_gradient()
        self.downscale = downscale
        # loading texture
        # Create Texture
        self.dr_data, self.pi_data, self.scel_data = glGenTextures(3)
        if downscale:
            self._init_3d_texture(
                self.dr_data,
                self.concentration,
                GL_TEXTURE1,
                "dr_volume",
                1,
                downscale=True,
            )
            self._init_3d_texture(
                self.pi_data,
                self.pi_concentration,
                GL_TEXTURE2,
                "pi_volume",
                2,
                downscale=True,
            )
            self.shape = (
                self.concentration.shape[0] // 2,
                self.concentration.shape[1] // 2,
                self.concentration.shape[2] // 2,
            )
        else:
            self._init_3d_texture(
                self.dr_data, self.concentration, GL_TEXTURE1, "dr_volume", 1
            )
            self._init_3d_texture(
                self.pi_data, self.pi_concentration, GL_TEXTURE2, "pi_volume", 2
            )
            self.shape = self.concentration.shape
        self.pi_concentration = None
        self.concentration = None
        self.Refresh()
        return self.context.data_context.dr_stack,self.context.data_context.pi_stack


    def _2d_texture(self, loc, data, enum, name, num):
        glUseProgram(self.concentration_shader)
        glActiveTexture(enum)
        glBindTexture(GL_TEXTURE_2D, loc)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            data.shape[0],
            1,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data.tobytes(),
        )

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # Init the texture units
        glActiveTexture(enum)
        glBindTexture(GL_TEXTURE_2D, loc)
        # Setup VolumeData on Texture Unit 2 (we are using 1 for TransferFunction)
        glUniform1i(glGetUniformLocation(self.concentration_shader, name), num)

    # todo
    def init_gradient(self):
        self.dr_gradient_loc, self.pi_gradient_loc = glGenTextures(2)
        self._2d_texture(
            self.dr_gradient_loc, self.dr_gradient, GL_TEXTURE4, "dr_gradient", 4
        )
        self._2d_texture(
            self.pi_gradient_loc, self.pi_gradient, GL_TEXTURE5, "pi_gradient", 5
        )
        self.if_gradient_initialised = True
        # gradient = np.array([(200)]*128)#+[(255,0,20,2*i) for i in range(128)])
        # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def InitGL(self):
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.concentration_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(sh.vertex_concentration, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(
                sh.fragment_concentration, GL_FRAGMENT_SHADER
            ),
        )

        glUseProgram(self.concentration_shader)
        glEnable(GL_DEPTH_TEST)

        view = matrix44.create_from_translation(-self.camera_position)
        projection = matrix44.create_perspective_projection_matrix(
            45.0, self.aspect_ratio, 0.1, 100.0
        )

        vp = matrix44.multiply(view, projection)
        vp_loc = glGetUniformLocation(self.concentration_shader, "vp")
        glUniformMatrix4fv(vp_loc, 1, GL_FALSE, vp)

        self.skeleton_loaded_loc = glGetUniformLocation(
            self.concentration_shader, "skeleton_loaded"
        )


        self.vao_cube = glGenVertexArrays(1)
        glBindVertexArray(self.vao_cube)
        vbo_cube = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
        glBufferData(GL_ARRAY_BUFFER, len(cube) * 3 * 4, cube.tobytes(), GL_STATIC_DRAW)
        # vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        self.skeleton_bundle = SkeletonShaderBundle([])
        self.init = True

    def reset_position(self,scale=2.0):
        self.rot = Matrix44.from_x_rotation(-np.pi / 2)
        self.camera_position.z=scale
        self.context.graphic_context.set_zoom_scale_field(1/scale)
        self.Refresh()

    def set_clip(self, min_tup, max_tup):
        # print(min_tup,max_tup)
        self.min_clip = min_tup
        self.max_clip = max_tup
        self.Refresh()


    def init_uniforms(self,shader):
        
        vp_loc = glGetUniformLocation(shader, "vp")

        eye_pos_loc = glGetUniformLocation(shader, "eye_pos")

        rot_loc = glGetUniformLocation(shader, "rotate")
        trans_loc = glGetUniformLocation(shader, "translate")


        min_clip_loc = glGetUniformLocation(shader, "min_clip")
        max_clip_loc = glGetUniformLocation(shader, "max_clip")


        glUniform3f(
            eye_pos_loc,
            self.camera_position.x,
            self.camera_position.y,
            self.camera_position.z,
        )
        

        glUniform3f(min_clip_loc, *self.min_clip)
        glUniform3f(max_clip_loc, *self.max_clip)

        view = matrix44.create_from_translation(-self.camera_position)
        projection = matrix44.create_perspective_projection_matrix(
            45.0, self.aspect_ratio, 0.1, 100.0
        )

        # glClearColor(1.0, 1.0, 1.0, 1.0)

        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.translate = matrix44.create_from_translation(Vector3([0, 0, 0]))

        self.combined_matrix = matrix44.multiply(self.rot, self.translate)

        vp = matrix44.multiply(view, projection)

        glUniformMatrix4fv(vp_loc, 1, GL_FALSE, vp)

        if self.rotate:
            ct = time.perf_counter()
            self.rot = Matrix44.from_y_rotation(ct)
            glUniformMatrix4fv(rot_loc, 1, GL_FALSE, self.rot)
            glUniformMatrix4fv(trans_loc, 1, GL_FALSE, self.translate)
            self.Refresh()
        else:
            glUniformMatrix4fv(rot_loc, 1, GL_FALSE, self.rot)
            glUniformMatrix4fv(trans_loc, 1, GL_FALSE, self.translate)


        
        
        shape_loc = glGetUniformLocation(shader, "volume_dims")
        scale_loc = glGetUniformLocation(shader, "cube_scaling")

        skeleton_loaded_loc = glGetUniformLocation(
            shader, "skeleton_loaded"
        )
        glUniform3f(scale_loc, *self.scaling_ratio)
        shape = (
                self.context.data_context.dr_stack_shape[0] // 2,
                self.context.data_context.dr_stack_shape[1] // 2,
                self.context.data_context.dr_stack_shape[2] // 2,
            )
        glUniform3i(shape_loc, *shape)
        glUniform1i(skeleton_loaded_loc, 0)


            
    def OnDraw(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.OnSkeletonDraw()

        glUseProgram(self.concentration_shader)
        self.init_uniforms(self.concentration_shader)

        glEnable(GL_CLIP_PLANE0)
        glBindVertexArray(self.vao_cube)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        self.SwapBuffers()
        self._draw_text=True
        self.Refresh()

    def OnSkeletonDraw(self):
        glUseProgram(self.skeleton_bundle.smooth_shader)
        glEnable(GL_CLIP_PLANE0)

        def init_skeleton_uniform(shader):
            self.init_uniforms(shader)
            min_clip_loc = glGetUniformLocation(shader, "min_clip")
            max_clip_loc = glGetUniformLocation(shader, "max_clip")
            a, b = self.get_clip_borders()
            glUniform3f(min_clip_loc, a[0], a[1], a[2])
            glUniform3f(max_clip_loc, b[0], b[1], b[2])

        init_skeleton_uniform(self.skeleton_bundle.smooth_shader)
        glBindVertexArray(self.skeleton_bundle.vao_lines)
        glLineWidth(self.line_width)
        glPatchParameteri(GL_PATCH_VERTICES, 2 )
        glDrawArrays(GL_PATCHES, 0, self.skeleton_bundle.size)
        if self.skeleton_bundle.marked_points:
            glUseProgram(self.skeleton_bundle.shader)
            init_skeleton_uniform(self.skeleton_bundle.shader)
            glPointSize(self.point_size)
            glBindVertexArray(self.skeleton_bundle.vao_points)
            glDrawArrays(GL_POINTS, 0, self.skeleton_bundle.points_size)
        glDisable(GL_CLIP_PLANE0)

    def get_clip_borders(self):
        a = list(map(lambda x: -(x - 0.5), self.min_clip))
        b = list(map(lambda x: (x - 0.5), self.max_clip))
        return (
            (a[0], a[1] * self.scaling_ratio[2], a[2]),
            (b[0], b[1] * self.scaling_ratio[2], b[2]),
        )

    def _reload_gradient(self, gradient, enum):
        glUseProgram(self.concentration_shader)
        # gradient = np.array([(200)]*128)#+[(255,0,20,2*i) for i in range(128)])
        # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glActiveTexture(enum)
        # self.dr_gradient = np.array([(0,255,20,i) for i in range(256)]).astype(np.uint8)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            gradient.shape[0],
            1,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            gradient.tobytes(),
        )
        # Init the texture units
        glActiveTexture(enum)
        self.Refresh()

    def reload_dr_gradinet(self, gradient):
        self.dr_gradient = gradient
        if self.if_gradient_initialised:
            self._reload_gradient(gradient, GL_TEXTURE4)

    def reload_pi_gradinet(self, gradient):
        self.pi_gradient = gradient
        if self.if_gradient_initialised:
            self._reload_gradient(gradient, GL_TEXTURE5)

    def set_mode(self, mode, func):
        self.modes = {}
        self.modes[mode] = func



class CustomDialog(sized_controls.SizedDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)
        pane = self.GetContentsPane()

        static_line = wx.StaticLine(pane, style=wx.LI_HORIZONTAL)
        static_line.SetSizerProps(border=(('all', 0)), expand=True)

        pane_btns = sized_controls.SizedPanel(pane)
        pane_btns.SetSizerType('vertical')
        pane_btns.SetSizerProps(align='center')

        button_ok = wx.StaticText(pane_btns,label='pixel width')
        self.X = wx.TextCtrl(pane_btns)
        button_ok = wx.StaticText(pane_btns,label='pixel height')
        self.Y = wx.TextCtrl(pane_btns)
        button_ok = wx.StaticText(pane_btns,label='voxel depth')
        self.Z = wx.TextCtrl(pane_btns)
        self.X.SetValue('0')
        self.Y.SetValue('0')
        self.Z.SetValue('0')
        button_ok = wx.Button(pane_btns, wx.ID_OK, label='OK')
        button_ok.Bind(wx.EVT_BUTTON, self.on_button)

        self.Fit()

    def on_button(self, event):
        if self.IsModal():
            self.EndModal(event.EventObject.Id) 

    def get_scale(self):
        return tuple(float(i.GetValue()) for i in (self.X,self.Y,self.Z))