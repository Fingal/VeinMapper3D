vertex_concentration = """
# version 300 es
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;

out vec3 newColor;
uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
uniform vec3 eye_pos;
uniform vec3 cube_scaling;

out vec3 vray_dir;
flat out vec3 transformed_eye;
out vec3 pos;

void main(){
    gl_Position = vp * translate * rotate * vec4(positions, 1.0);
    newColor = colors;
    transformed_eye = ((transpose(rotate)*vec4(eye_pos,1)).xyz);
	vray_dir = normalize(positions - transformed_eye);
    pos = positions;
}
"""

fragment_concentration = """
# version 300 es

precision highp int;
precision highp float;
uniform ivec3 volume_dims;
uniform highp sampler3D volume;
uniform highp sampler3D sceleton;

in vec3 newColor;
in vec3 vray_dir;
in vec3 pos;
flat in vec3 transformed_eye;


// WebGL doesn't support 1D textures, so we use a 2D texture for the transfer function
//uniform ivec3 volume_dims;

out vec4 color;


void main(void) {
    color=vec4(vray_dir+0.5,1);
}
"""


vertex_sceleton = """
# version 300 es
in layout(location = 0) vec3 positions;

uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
//uniform vec3 eye_pos;

flat out vec3 vray_dir;
flat out vec3 transformed_eye;
out vec3 pos;

void main(){
    gl_Position = vp * translate * rotate * vec4(positions, 1.0);
    newColor = colors;
    pos = positions;
}
"""
fragment_sceleton = (
    """
    #version 330
    out vec4 f_color;
    void main() {

        f_color = vec4(0, 0, 0, 1);
    }
""",
)
