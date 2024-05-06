import numpy as np
from OpenGL import GL
import OpenGL.GL.shaders
import shaders as sh
from PyQt5.QtWidgets import QOpenGLWidget, QApplication

vertex_skeleton = """
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;
in layout(location = 2) vec3 tangents;

out VS_OUT {
    vec3 color;
    vec3 tangent;
} vs_out;


void main(){
    gl_Position = vec4(positions, 1.0);
	vs_out.color = vec3(colors);
	vs_out.tangent = vec3(tangents);

}
"""
tess_control_skeleton = """
# version 420


layout (vertices = 4) out;

in VS_OUT { 
    vec3 color; 
    vec3 tangent;
} tcs_in[];   /* new */
out TCS_OUT { 
    vec3 color; 
    vec3 tangent;
} tcs_out[];  /* new */

void
main()
{
    gl_TessLevelOuter[0] = 1;
    gl_TessLevelOuter[1] = 64;
	
	gl_out[gl_InvocationID].gl_Position = vec4(gl_in[gl_InvocationID].gl_Position);
    tcs_out[gl_InvocationID].color = tcs_in[gl_InvocationID].color; /* forward the data */
    tcs_out[gl_InvocationID].tangent = tcs_in[gl_InvocationID].tangent; /* forward the data */
}
"""
tess_eval_skeleton = """
# version 420

layout (isolines, equal_spacing, ccw) in;


in TCS_OUT { 
    vec3 color; 
    vec3 tangent;
} tes_in[];  /* new */
out TES_OUT { vec3 color; } tes_out;  /* new */
///////////////////////////////////////////////////
// function to calculate the hermite function value
vec3 hermite(float u, vec3 p0, vec3 p1, vec3 t0, vec3 t1)
{
	float F1 = 2.*u*u*u - 3.*u*u + 1.;
	float F2 = -2.*u*u*u + 3*u*u;
	float F3 = u*u*u - 2.*u*u + u;
	float F4 = u*u*u - u*u;
    

	vec3 p = F1*p0 + F2*p1 + F3*t0 + F4*t1;
	return p;
} 

void
main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    vec3 vTan0=tes_in[0].tangent;
    vec3 vTan1=tes_in[1].tangent;

	vec3 vPos0 = vec3( gl_in[0].gl_Position.xyz );
	vec3 vPos1 = vec3( gl_in[1].gl_Position.xyz );
	vec3 v3pos = hermite( u, vPos0, vPos1, vTan0, vTan1 );
	vec4 pos = vec4( v3pos, 1.);
	gl_Position = pos;
    tes_out.color = (1-u)*tes_in[0].color+u*tes_in[1].color;
}
"""
fragment_skeleton = """
# version 420

in TES_OUT { vec3 color; } tes_out;

out vec4 f_color;
void main() {

	f_color = vec4(tes_out.color, 1);
}
"""
def points_to_strip(elements):
    result = []
    for x in zip(elements[:-1],elements[1:]):
        result.append(x)
    #result.append((elements[-1],elements[0]))
    return result


class OpenGLWidget(QOpenGLWidget):

    def initializeGL(self):
        C=0.7
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_skeleton, GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(tess_control_skeleton, GL.GL_TESS_CONTROL_SHADER),
            OpenGL.GL.shaders.compileShader(tess_eval_skeleton, GL.GL_TESS_EVALUATION_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_skeleton, GL.GL_FRAGMENT_SHADER),
        )
        path=[(0.0,0.5,0.0),(-0.5,-0.4,0.0)]
        print(points_to_strip([1,2,3,4]))
        vertices = np.array(points_to_strip(path), dtype=np.float32)
        colors = np.array([0,0,1,0,1,0]*3,dtype=np.float32)
        
        m=[]
        points = [path[0]]+path+[path[-1]]
        for a,b in zip(map(np.array,points[:-1]),map(np.array,points[2:])):
            m.append((1-C)*(b-a)*(2))
        tangents=np.array(points_to_strip(m),dtype=np.float32)

        bufferId,col_buffer,tangent_buffer = GL.glGenBuffers(3)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufferId)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices.tobytes(), GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FLOAT, 12, GL.ctypes.c_void_p(0))

        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors.tobytes(), GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, GL.ctypes.c_void_p(0))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, tangent_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, tangents.nbytes, tangents.tobytes(), GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, GL.ctypes.c_void_p(0))

    def paintGL(self):
        GL.glUseProgram(self.shader)
        GL.glPatchParameteri( GL.GL_PATCH_VERTICES, 2 )
        GL.glDrawArrays( GL.GL_PATCHES, 0, 2 )


app = QApplication([])
widget = OpenGLWidget()
widget.show()
app.exec_()