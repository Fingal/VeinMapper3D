from glob import glob
from math import dist

from numpy.lib.function_base import angle
import global_simulation
import context
import shape_calculation

def _angle_diff(a,b):
        r = abs(a-b)
        if r>180:
            r = 360-r
        return r


def _relative_angle_diff(a,b):
        r = (a-b)
        if r>180:
            r = r-360
        if r<-180:
            r=360+r
        return r
class SurfacePoint:
    def __init__(self,distance,angle,stage,position,label=None):
        self.context: context.Context=context.Context()
        self.distance=distance
        self.angle=angle
        self.stage=stage
        self.position=position
        self.primordium_label = label

    def __str__(self) -> str:
        if self.context.data_context.settings.PRINT_MODE==context.PrintMode.NOTHING:
            return ""
        else:
            return f"{self.stage:.2f} {self.primordium_label}"

    def __repr__(self) -> str:
        return str(self)

class SurfacePoints:
    def __init__(self,global_simulation : global_simulation.GlobalSimulation,time_passed = 0,existing_points_number=0,angle_offest=None,initial_angle=None,extra_offset=0,skip_primordiums = 0 ) -> None:
        self.gs : global_simulation.GlobalSimulation = global_simulation
        if angle_offest==None:
            self.offset,self.angle=shape_calculation.calculate_offset()
        else:
            self.angle=angle_offest
        if initial_angle==None:
            self.initial_angle = shape_calculation.get_initial_angle()
        else:
            self.initial_angle=initial_angle
        self.angle_error = 25
        self.time_passed = time_passed
        self.side_influence=0.2
        self.existing_points=[]
        self.extra_offset = extra_offset
        self.current_angle = self.initial_angle
        self.current_primordium_label = global_simulation.primordium_label
        self.skip_primordiums=skip_primordiums

        self._extra_offset_max = 8/12
        self._extra_offset_step = 8/(12*40)

        for i in range(existing_points_number):
            self.create_new_point()

    def create_new_point(self):
        distance = self.gs.growth_coeff_development.calculate_distance_to_center(self.gs.simulation_time) 
        self.current_angle = (360+self.current_angle-self.angle)%360
        #import pdb; pdb.set_trace()
        self.existing_points.append(SurfacePoint(distance,self.current_angle,-(self.gs.MATURED_AGE)*self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12.-self.extra_offset,
                                    shape_calculation.translate_from_cone_coordinate
                                    (self.gs.cone_coeffs,self.gs.center,distance,self.current_angle),
                                    label=self.current_primordium_label))
        self.current_primordium_label-=1
        

    

    def step(self,dt):
        #print("surface_step")
        new_extra_offset =min(self._extra_offset_max,self.extra_offset+self._extra_offset_step * dt)
        self.time_passed +=dt+new_extra_offset-self.extra_offset
        self.extra_offset = new_extra_offset
        for p in self.existing_points:
            p.stage+=dt
        new_p =self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12
        if self.time_passed>new_p:
            if self.skip_primordiums>0:
                self.skip_primordiums-=1
                self.time_passed-=new_p
            #print("new_point")
            else:
                self.create_new_point()
                self.time_passed-=new_p
            
    def find_closest(self,point):
        r,a=shape_calculation.translate_to_cone_coordinate(self.gs.cone_coeffs,self.gs.center,point)
        consumed : SurfacePoint = None
        # if self.existing_points:
        #     p = min(self.existing_points, key = lambda x: _angle_diff(a,x.angle))
        #     return tuple(p.position),_angle_diff(a,p.angle)
        # moved to sendig oldest
        for point in self.existing_points:
            if _angle_diff(a,point.angle)<self.angle_error:
                return tuple(point.position),_angle_diff(a,point.angle)

    def check_if_connected(self,point):
        r,a=shape_calculation.translate_to_cone_coordinate(self.gs.cone_coeffs,self.gs.center,point)
        consumed : SurfacePoint = None
        if len(self.existing_points)==0:
            return None
        # p = min(self.existing_points, key = lambda x: _angle_diff(a,x.angle))
        # if _angle_diff(a,p.angle)<self.angle_error:
        #     consumed=p
        #     #adjusting step
        #     self.current_angle+=_relative_angle_diff(a,p.angle)*self.side_influence
        
        for point in self.existing_points:
            if _angle_diff(a,point.angle)<self.angle_error:
                consumed=point
                self.current_angle+=_relative_angle_diff(a,point.angle)*self.side_influence
                break
            
        if consumed is not None:
            self.existing_points.remove(consumed)
            return consumed
        else:
            return None
    

class DoubleSurfacePoint(SurfacePoints):
    def __init__(self,global_simulation : global_simulation.GlobalSimulation,time_passed = 0,existing_points_number=0,angle_offest=None,initial_angle=None,extra_offset=0,modification_after_first = 1) -> None:
        SurfacePoints.__init__(self,global_simulation,time_passed,existing_points_number,angle_offest,initial_angle,extra_offset=extra_offset)
    
    def create_new_point(self):

        print("New SURFACE POINT")
        print("New SURFACE POINT")
        print(self.time_passed)
        print(self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12)
        print(self.current_primordium_label)
        print(-1*self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12.-self.extra_offset)
        print(self.extra_offset)
        print(self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12.)
        print(self._extra_offset_max)
        print("New SURFACE POINT")

        distance = self.gs.growth_coeff_development.calculate_distance_to_center(self.gs.simulation_time) 
        self.current_angle = (360+self.current_angle-self.angle)%360
        self.existing_points.append(SurfacePoint(distance,self.current_angle,-1*self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12.-self.extra_offset,
                                    shape_calculation.translate_from_cone_coordinate
                                    (self.gs.cone_coeffs,self.gs.center,distance,self.current_angle),
                                    label=self.current_primordium_label))
        self.existing_points.append(SurfacePoint(distance,(self.current_angle+180)%360,-1*self.gs.growth_coeff_development.calculate_surface_frequency(self.gs.simulation_time)/12.-self.extra_offset,
                                    shape_calculation.translate_from_cone_coordinate
                                    (self.gs.cone_coeffs,self.gs.center,distance,(self.current_angle+180)%360),
                                    label=self.current_primordium_label))
        # if self.current_primordium_label==-1:
        #     self.extra_offset-=1.6
        #     self.time_passed=-1.6
        self.current_primordium_label-=1