# def conduct_filling():
#     starting_point = c.canvas._stack_coord_to_micron((0.5,0.5,0.5))
#     vec_dir = tuple(np.array(c.canvas._stack_coord_to_micron((1.,1.,1.)))-np.array(c.canvas._stack_coord_to_micron((0.,0.,0.))))

#     prediction_lines=[]
#     for v in gs.lines:
#         prediction_lines.append(({(gs.points[k],c.canvas._micron_coord_to_stack(gs.points[k])) for k in v.skeleton.keys()},v.age))


#     return generate_line.fill_array(c.data_context.dr_stack_shape,starting_point,vec_dir,prediction_lines,**gs.coeffs)

#x=conduct_filling()
# normal_growth()
# c.simulation_controller.color_step(steps=40, time_step=30)


    

# def f(a):
#     c.filled_data=conduct_filling()
# button = c.skeleton_distances.add_button("new button", (20, 0), (1, 10), f)



# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy

# from pycuda.compiler import SourceModule
# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)

# multiply_them = mod.get_function("multiply_them")

# def random_vector():
#     result = Vector3((np.random.uniform(),np.random.uniform(),np.random.uniform()))
#     result.normalize()
#     return result

# def random_from_center(center,scale=1):
#     result = Vector3(center)+2*scale*(random_vector()-0.5)
#     return tuple(result)

# def growth_tables(steps=800,starting_point=None):
#     global gs
#     global time_step
#     global cone_center
#     if starting_point is None:
#         global points
#         starting_point = points[0]
#     results=[[starting_point] for point in range(31)]
#     stop = -1
#     big_enough = set()
#     for i in range(steps):
#         for age in range(31):
#             growth_coeffs={k:(min(30,age)*gs.growth_coeffs_old[k]+max(0,30-age)*v)/30 for k,v in gs.growth_coeffs_young.items()}
#             new_points=generate_line.grow_points_cone([results[age][-1]],gs.center,time_step,**gs.cone_coeffs,**growth_coeffs)
#             if (Vector3(cone_center)-Vector3(new_points[0])).length>200:
#                 big_enough.add(age)
#             results[age].append(new_points[0])
#         if len(big_enough)>=30:
#             return results
#     return results

#points = [random_from_center(cone_center,10) for i in range(5)]
#plt.plot([i*time_step for i in range(len(results[0]))],np.array([[(Vector3(point)-Vector3(cone_center)).length for point in result] for result in results]).T)
#from cycler import cycler
# def plot_color_gradient(results):
#     global cone_center
#     from cycler import cycler
#     colors=cycler(color=[((31-i)/31,0,0) for i in range(31)])
#     fig,ax = plt.subplots(1)
#     ax.set_prop_cycle(colors)
#     ax.plot(np.array([[(Vector3(point)-Vector3(cone_center)).length for point in result[:-1]] for result in results]).T,np.array([[(Vector3(a)-Vector3(b)).length for a,b in zip(result,result[1:])] for result in results]).T)
#     return fig,ax
#fig,ax=plot_color_gradient
#c.graphic_context.mark_temporary_points([c.canvas._micron_coord_to_stack(point) for point in points])
#c.graphic_context.mark_temporary_points([c.canvas._micron_coord_to_stack(point) for result in results for point in result])
#results = growth_tables()

# cone_center = generate_line.project_to_cone(gs.center,gs.center,**gs.cone_coeffs)

# points = [random_from_center(cone_center,40) for i in range(5)]
# time_step=1/8
# results = growth_tables()