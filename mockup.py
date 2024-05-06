import wx
from wx import glcanvas
from OpenGL.GL import *
import OpenGL.GL.shaders
from pyrr import Matrix44, matrix44, Vector3
import time, sys
from Cube import *
from stack_io import *
import numpy as np
import shaders_mockup


class OpenGLCanvasMockup(glcanvas.GLCanvas):
    def __init__(self, parent):
        self.size = (1120, 5000)
        self.aspect_ratio = self.size[0] / self.size[1]
        # glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        self.SetMaxSize((5000, 5000))
        self.Parent
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.init = False
        self.rotate = False
        self.rot_y = Matrix44.identity()
        self.mesh = None
        self.show_triangle = False
        self.show_quad = False
        self.show_cube = False
        self.rot_loc = None
        self.trans_loc = None
        self.trans_x, self.trans_y, self.trans_z = 0.0, 0.0, -2.0
        self.translate = Matrix44.identity()
        self.bg_color = False
        self.wireframe = False
        self.combined_matrix = Matrix44.identity()
        self.camera_position = Vector3([0.0, 0.0, 2.0])
        # self.vol = np.zeros((100,100,100,4), np.uint8)
        # self.vol[:10] += 10

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)

        self.mouse_pos = (0, 0)
        self.moving_pressed = False

    def OnMouse(self, event):
        if event.Button(wx.MOUSE_BTN_LEFT):
            pass
            # print('button')
        if event.ButtonDown(wx.MOUSE_BTN_LEFT):
            self.mouse_pos = event.GetLogicalPosition(self.dc)
            self.moving_pressed = True
            # print('button down')
        if event.ButtonUp(wx.MOUSE_BTN_LEFT) or event.Leaving():
            self.mouse_pos = event.GetLogicalPosition(self.dc)
            self.moving_pressed = False
            # print('button up')
        if event.Dragging() and self.moving_pressed:
            new_y = event.GetLogicalPosition(self.dc).y
            new_x = event.GetLogicalPosition(self.dc).x
            self.rot_y = (
                Matrix44.from_y_rotation((self.mouse_pos.x - new_x) / 100)
                * Matrix44.from_x_rotation((self.mouse_pos.y - new_y) / 100)
                * self.rot_y
            )
            self.mouse_pos = event.GetLogicalPosition(self.dc)
            self.Refresh()

        if (event.GetWheelRotation()) != 0:
            # print(event.GetWheelRotation())
            self.camera_position = (
                self.camera_position + Vector3([0, 0, 0.001]) * event.GetWheelRotation()
            )
            self.Refresh()
            # print('draging')
        # import pdb; pdb.set_trace()

    def OnResize(self, event):
        size = self.GetClientSize()
        glViewport(0, 0, size.width, size.height)

        self.aspect_ratio = size.width / size.height

    def OnPaint(self, event):
        self.dc = wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

    def InitGL(self):
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.concentration_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(
                shaders_mockup.vertex_concentration, GL_VERTEX_SHADER
            ),
            OpenGL.GL.shaders.compileShader(
                shaders_mockup.fragment_concentration, GL_FRAGMENT_SHADER
            ),
        )

        glUseProgram(self.concentration_shader)
        glEnable(GL_DEPTH_TEST)

        view = matrix44.create_from_translation(-self.camera_position)
        projection = matrix44.create_perspective_projection_matrix(
            45.0, self.aspect_ratio, 0.1, 100.0
        )

        vp = matrix44.multiply(view, projection)
        self.vp_loc = glGetUniformLocation(self.concentration_shader, "vp")
        glUniformMatrix4fv(self.vp_loc, 1, GL_FALSE, vp)

        self.eye_pos_loc = glGetUniformLocation(self.concentration_shader, "eye_pos")

        self.rot_loc = glGetUniformLocation(self.concentration_shader, "rotate")
        self.trans_loc = glGetUniformLocation(self.concentration_shader, "translate")

        self.vao_cube = glGenVertexArrays(1)
        glBindVertexArray(self.vao_cube)
        vbo_cube = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
        glBufferData(GL_ARRAY_BUFFER, len(cube) * 3 * 4, cube.tobytes(), GL_STATIC_DRAW)
        # vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def OnDraw(self):

        glUniform3f(
            self.eye_pos_loc,
            self.camera_position.x,
            self.camera_position.y,
            self.camera_position.z,
        )

        view = matrix44.create_from_translation(-self.camera_position)
        projection = matrix44.create_perspective_projection_matrix(
            45.0, self.aspect_ratio, 0.1, 100.0
        )

        glClearColor(1.0, 1.0, 1.0, 1.0)

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.translate = matrix44.create_from_translation(Vector3([0, 0, 0]))

        self.combined_matrix = matrix44.multiply(self.rot_y, self.translate)

        vp = matrix44.multiply(view, projection)

        glUniformMatrix4fv(self.vp_loc, 1, GL_FALSE, vp)

        if self.rotate:
            ct = time.perf_counter()
            self.rot_y = Matrix44.from_y_rotation(ct)
            glUniformMatrix4fv(self.rot_loc, 1, GL_FALSE, self.rot_y)
            glUniformMatrix4fv(self.trans_loc, 1, GL_FALSE, self.translate)
            self.Refresh()
        else:
            glUniformMatrix4fv(self.rot_loc, 1, GL_FALSE, self.rot_y)
            glUniformMatrix4fv(self.trans_loc, 1, GL_FALSE, self.translate)

        glBindVertexArray(self.vao_cube)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        self.SwapBuffers()
