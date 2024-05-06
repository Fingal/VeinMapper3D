vertex_concentration = """
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;

out vec3 newColor;
uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
uniform vec3 eye_pos;
uniform vec3 cube_scaling;

uniform vec3 min_clip;
uniform vec3 max_clip;

flat out vec3 vray_dir;
//flat out vec3 transformed_eye;
out vec3 pos;
out vec3 dir;
out vec3 vPos;

void main(){
	vec3 p = (positions - max(sign(positions)*(vec3(1.0) - max_clip),vec3(0)) - min(sign(positions)*(vec3(1.0) - min_clip),vec3(0)));
    gl_Position = vp * rotate * translate *  vec4(cube_scaling.xzy*p, 1.0);
	//dir=(vp * rotate * translate *  vec4(cube_scaling.xzy*p, 0.0)).xyz;
	dir=(vp * rotate * translate *  vec4(cube_scaling.xzy*vec3(0,0,1), 0.0)).xyz;
    newColor = colors;
    //transformed_eye = ((transpose(rotate)*vec4(eye_pos,1)).xyz)/cube_scaling.xzy;
	vray_dir = vec3(1,0,0);
    pos = p;
	vPos = (vp * rotate * translate *  vec4(cube_scaling.xzy*p, 1.0)).xyz;
	vPos =(1/cube_scaling.xzy)*( inverse(vp * rotate * translate) *  vec4(vec3(p.xy,-1), 1.0)).xyz;
}
"""

fragment_concentration = """
# version 420

precision highp int;
precision highp float;
uniform ivec3 volume_dims;
uniform highp sampler3D dr_volume;
uniform highp sampler3D pi_volume;
uniform highp sampler3D skeleton;
uniform sampler2D dr_gradient;
uniform sampler2D pi_gradient;
uniform int skeleton_loaded;
uniform vec3 cube_scaling;

uniform vec3 min_clip;
uniform vec3 max_clip;

in vec3 newColor;
flat in vec3 vray_dir;
in vec3 pos;
in vec3 dir;
in vec3 vPos;
//flat in vec3 transformed_eye;


// WebGL doesn't support 1D textures, so we use a 2D texture for the transfer function
//uniform ivec3 volume_dims;

out vec4 color;

vec2 intersect_box(vec3 orig, vec3 dir,vec3 box_dim) {
	//const vec3 box_min = vec3(-0.5) - ;
	//const vec3 box_max = vec3(0.5);
	vec3 box_min = box_dim/2 - min_clip;
	vec3 box_max = -box_dim/2 + max_clip;
	vec3 inv_dir = 1.0 / dir;
	vec3 tmin_tmp = (box_min - orig) * inv_dir;
	vec3 tmax_tmp = (box_max - orig) * inv_dir;
	vec3 tmin = min(tmin_tmp, tmax_tmp);
	vec3 tmax = max(tmin_tmp, tmax_tmp);
	float t0 = max(tmin.x, max(tmin.y, tmin.z));
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	return vec2(t0, t1);
}

void main(void) {
	color = vec4(1,1,1,0);
    //ivec3 volume_dims = ivec3(100,100,100);
	// Step 1: Normalize the view ray
	vec3 transformed_eye=vPos;
	vec3 ray_dir = normalize(pos-transformed_eye);
	//vec3 ray_dir = normalize(dir);
	//vec3 ray_dir = vec3(0,0,1);
    bool skeleton_caught;

	// Step 2: Intersect the ray with the volume bounds to find the interval
	// along the ray overlapped by the volume.
	vec2 t_hit = intersect_box(transformed_eye, ray_dir,vec3(1,1,1));
	if (t_hit.x > t_hit.y) {
		discard;
	}
    t_hit = t_hit;
	// We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
	t_hit.x = max(t_hit.x, 0.0);

	// Step 3: Compute the step size to march through the volume grid
	vec3 dt_vec = 1.0 / (vec3(volume_dims) * abs(ray_dir));
	float dt = min(dt_vec.x, min(dt_vec.y, dt_vec.z));

	// Step 4: Starting from the entry point, march the ray through the volume
	// and sample it
	vec3 p = transformed_eye + t_hit.x * ray_dir;
	for (float t = t_hit.x; t < t_hit.y; t += dt) {
		// Step 4.1: Sample the volume, and color it by the transfer function.
		// Note that here we don't use the opacity from the transfer function,
		// and just use the sample value as the opacity
		float dr_val = texture(dr_volume, p.yxz+0.5).a;
		float pi_val = texture(pi_volume, p.yxz+0.5).a;
		if (skeleton_loaded > 0){
			float scel = texture(skeleton, p.yxz+0.5).a;
			
			if (scel>0.0){
				vec4 val_color = vec4(0,0,0, scel);
				color = mix(color,vec4(0,0,0,1),0.5);
			}
			else if (color.a < 0.95){
				//vec4 val_color = vec4(0,1.0-dr_val,0, dr_val);
				vec4 dr_color = texture(dr_gradient, vec2(dr_val,0.5));
				vec4 pi_color = texture(pi_gradient, vec2(pi_val,0.5)); 
				float al_sum=pi_color.a+dr_color.a;
				if (al_sum>0){
					vec4 val_color = vec4(mix(dr_color.rgb,pi_color.rgb,pi_color.a/(al_sum)),al_sum);
					color.rgb += (1.0 - color.a) * val_color.a * val_color.rgb;
					color.a += (1.0 - color.a) * val_color.a;
				}
			}
		}
		else if (color.a < 0.95){
			//vec4 val_color = vec4(0,1.0-dr_val,0, dr_val);
			vec4 dr_color = texture(dr_gradient, vec2(dr_val,0.5));
			vec4 pi_color = texture(pi_gradient, vec2(pi_val,0.5));
			float al_sum=max(pi_color.a,dr_color.a);
			if (al_sum>0){
				vec4 val_color = vec4(mix(dr_color.rgb,pi_color.rgb,pi_color.a/(al_sum)),al_sum);
				//color.rgb += (1.0 - color.a) * val_color.a * val_color.rgb;
				color.rgb -= (1.0 - color.a) * val_color.a * (vec3(1) - val_color.rgb)*1;
				color.a += (1.0 - color.a) * val_color.a;
			}
		}

		// Step 4.2: Accumulate the color and opacity using the front-to-back
		// compositing equation

		// Optimization: break out of the loop when the color is near opaque
		p += ray_dir * dt;	
	}
    vec3 c = transformed_eye + t_hit.x * ray_dir;
    //color=vec4(transformed_eye,1);
    //color=vec4(texture(dr_volume,vec3(0.11,pos.y+0.5,pos.x/.0+0.05)).a,0,0,1);
}
"""


vertex_skeleton = """
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;

out vec3 color;

uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
uniform vec3 min_clip;
uniform vec3 max_clip;


//uniform vec3 eye_pos;


void main(){
    gl_Position = vp * rotate * translate * vec4(positions, 1.0);
	vec3 v = min((max_clip)-positions,positions-(min_clip));
	gl_ClipDistance[0]=min(min(v.x,v.y),v.z);
	color = vec3(colors);
    //gl_Position = vp * vec4(positions, 1.0);

}
"""

fragment_skeleton = """
# version 420
in vec3 color;
out vec4 f_color;
void main() {

	f_color = vec4(color, 1);
}
"""

#round points shaders
vertex_sphere = """
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;

out vec3 vcolor;

uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
uniform vec3 min_clip;
uniform vec3 max_clip;


//uniform vec3 eye_pos;


void main(){
    gl_Position = vp * rotate * translate * vec4(positions, 1.0);
	vec3 v = min((max_clip)-positions,positions-(min_clip));
	gl_ClipDistance[0]=min(min(v.x,v.y),v.z);
	vcolor = vec3(colors);
    //gl_Position = vp * vec4(positions, 1.0);

}
"""
geometry_sphere = """
# version 420
layout (points) in;
layout (triangle_strip, max_vertices = 90) out;

uniform float aspect_ratio;
uniform float scale;
uniform float point_size;

in vec3 vcolor[];

out vec3 color;

vec4 circle(float t){
	return vec4(cos(t*3.14*2),sin(t*3.14*2),0,0);
}

void emit_triangle(vec4 a, vec4 b){
	vec4 position = gl_in[0].gl_Position;
	gl_Position = position;
    EmitVertex(); 
    gl_Position = position + a/vec4(aspect_ratio,1,1,1)*1; 
    EmitVertex(); 
    gl_Position = position + b/vec4(aspect_ratio,1,1,1)*1; 
    EmitVertex();
    EndPrimitive();
}
void emit_vertex(vec4 a){
    gl_Position = gl_in[0].gl_Position + a/vec4(aspect_ratio,1,1,1)*1; 
    EmitVertex(); 
}

void main(){
	color=vcolor[0];
	//float points = 27;
	float distance=point_size/100.;
	int points = 30;
	if (points%2==1){
		emit_vertex(circle(0.)*distance);
	}
	for (int i=0;i<int(points/2);i++){
		if (points%2==1){
			emit_vertex(circle((-i-1)/float(points))*distance);
			emit_vertex(circle((i+1)/float(points))*distance);
		}
		else{
			emit_vertex(circle((i)/float(points))*distance);
			emit_vertex(circle((-i-1)/float(points))*distance);
		}
	}
    EndPrimitive();
	//for (float i=0;i<points;i+=1){
	//	emit_triangle(circle(-i/points)*distance,circle(-(i+1)/points)*distance);
	//}
	//emit_triangle(-circle(0.0)*distance,-circle(0.4)*distance);
	//emit_triangle(circle(0.0)*distance,circle(0.4)*distance);
	//emit_triangle(vec4(distance, -distance, 0.0, 0.0),vec4(distance, distance, 0.0, 0.0));
	//emit_triangle(vec4(distance, distance, 0.0, 0.0),vec4(distance, -distance, 0.0, 0.0));
	//emit_triangle(vec4(distance, -distance, 0.0, 0.0),vec4(-distance, -distance, 0.0, 0.0));
	//emit_triangle(vec4(-distance, -distance, 0.0, 0.0),vec4(-distance, distance, 0.0, 0.0));
	//emit_triangle(vec4(-distance, distance, 0.0, 0.0),vec4(distance, distance, 0.0, 0.0));
}
"""

fragment_sphere = """
# version 420
in vec3 color;
out vec4 f_color;
void main() {

	f_color = vec4(color, 1);
}
"""


smooth_skeleton = {

"vertex":"""
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;
in layout(location = 2) vec3 tangents;


uniform vec3 min_clip;
uniform vec3 max_clip;


out VS_OUT {
    vec3 color;
    vec3 tangent;
} vs_out;


void main(){
    gl_Position = vec4(positions, 1.0);
	vs_out.color = vec3(colors);
	vs_out.tangent = vec3(tangents);
	vec3 v = min((max_clip)-positions,positions-(min_clip));
	gl_ClipDistance[0]=min(min(v.x,v.y),v.z);

}
""",
'tess_control':"""
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
""",
'tess_eval':"""
# version 420

layout (isolines, equal_spacing, ccw) in;

uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;

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
    gl_Position = vp * translate * rotate * vec4( v3pos, 1.);
    tes_out.color = (1-u)*tes_in[0].color+u*tes_in[1].color;
}
""",
'fragment':"""
# version 420

in TES_OUT { vec3 color; } tes_out;

out vec4 f_color;
void main() {

	f_color = vec4(tes_out.color, 1);
}
"""


}
arrow={
"vertex": """
# version 420
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 dirs;

out vec4 d_point;
out vec4 d_point_t1;
out vec4 d_point_t2;

uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
uniform vec3 min_clip;
uniform vec3 max_clip;


//uniform vec3 eye_pos;


void main(){
    gl_Position = vp * rotate * translate * vec4(positions, 1.0);
	vec3 v = min((max_clip)-positions,positions-(min_clip));
	gl_ClipDistance[0]=min(min(v.x,v.y),v.z);
	vec4 dd = vp * rotate * translate * vec4(positions+0.02*normalize(dirs), 1.0);
	d_point = dd/dd.w;

	vec3 vt = cross(dirs,vec3(0,1,0));
	dd = vp * rotate * translate * vec4(positions+0.015*normalize(dirs)+0.002*normalize(vt), 1.0);
	d_point_t1 = dd/dd.w;
	dd = vp * rotate * translate * vec4(positions+0.015*normalize(dirs)+0.002*normalize(-vt), 1.0);
	d_point_t2 = dd/dd.w;


    //gl_Position = vp * vec4(positions, 1.0);

}
""",
#geometry
"geometry": """
# version 420
layout (points) in;
layout (line_strip, max_vertices = 6) out;

uniform float aspect_ratio;
uniform float scale;
uniform float point_size;

in vec4 d_point[];
in vec4 d_point_t1[];
in vec4 d_point_t2[];



void main(){
	vec4 position = gl_in[0].gl_Position;
	gl_Position = position;
    EmitVertex(); 
	position = d_point[0];
	gl_Position = position;
    EmitVertex(); 
    EndPrimitive();
	gl_Position = d_point[0];
    EmitVertex(); 
	gl_Position = d_point_t1[0];
    EmitVertex(); 
    EndPrimitive();
	gl_Position = d_point[0];
    EmitVertex(); 
	gl_Position = d_point_t2[0];
    EmitVertex(); 
    EndPrimitive();
}
""",
#fragment
"fragment" : """
# version 420
out vec4 f_color;
void main() {

	f_color = vec4(1,0,0, 1);
}
"""
}
