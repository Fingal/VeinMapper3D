import imp
from simulation_controller import SimulationController
import context
from simulation_initialization import *

c=Context()

def b_mod(x):
    return [{'A': 0.2614290317628093, 'B': 7.062656265881261+x, 'C': 35.85128301663676},
{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25)+x, 'C': 46.3787741621418},
#{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
{'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625)+x, 'C': 56.38910353160223}]
def init_c(screenshots=240):
    c.simulation_controller=SimulationController()
    c.simulation_controller.screenshots_number=screenshots
    c.simulation_controller.do_save_images=True
    c.simulation_controller.run_calculation_thread()

def coeffs_mod(name,mod_value):
    result = {'pos_range': 20.6139871413999,
                'age_coef': 0.03963136053954481,
                'inhibition_coef': 2.7873971411850995,
                'attraction_coef': 0.0,
                'neg_range': 59.22591211160096,
                'straight_coef': 3.0624121912173043,
                'inertia': 0.5,
                #'inertia': 1.,
                'peak_coef': 0.99,
                'age_cut_off_coef': 0.8961433833351786}
    result[name]+=mod_value
    return result
#c=context.Context()
# def run_experiments():
#     init_small_no_growth()
#     init_c(300)
#     yield None
#     init_small_no_growth_frequency()
#     init_c(300)
#     yield None

from itertools import product

def run_lucas():
    c.data_context.settings.SAVE_IMAGES=True
    c.data_context.settings.SAVE_VECTOR_IMAGES=False
    init_c(200)
    init_lucas()


def run_bijugate():
    c.data_context.settings.SAVE_IMAGES=True
    c.data_context.settings.SAVE_VECTOR_IMAGES=False
    init_c(200)
    init_bijugate_idealized()


def run_fibonacci():
    #init_c(300)
    c.data_context.settings.SAVE_IMAGES=True
    c.data_context.settings.SAVE_VECTOR_IMAGES=False
    init_c(400)
    init_small_idealized(medium_n5_time=45,suffix="")

def run_experiments():
    #init_c(300)
    c.data_context.settings.SAVE_IMAGES=True
    c.data_context.settings.SAVE_VECTOR_IMAGES=False
    init_c(400)
    init_small_idealized(medium_n5_time=45,suffix="_final")
    yield None
    # init_lucas()
    # yield None
    #init_c(220)
    #init_bijugate_idealized(suffix="n+3_time")
    #yield None
    # #c.data_context.global_simulation.growth_coeff_development.surface_frequency = [13-1.5,10-1.5,8-1.5]
    # #c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=b_mod(2.5)
    # init_c(300)
    # c.data_context.settings.SAVE_IMAGES=True
    # yield None
    #"b+2.5 p-1.5"
    #"b-1.5 p+1.5"
    # init_small_idealized(medium_n5_time=45,suffix=f"p-1.5 b+2.5")
    # c.data_context.global_simulation.growth_coeff_development.surface_frequency = [13-1.5,10-1.5,8-1.5]
    # c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=b_mod(2.5)
    # init_c(300)
    # c.data_context.settings.SAVE_IMAGES=True
    # yield None

    # init_small_idealized(medium_n5_time=45,suffix=f"p+1.5 b-1.5_400")
    # c.data_context.global_simulation.growth_coeff_development.surface_frequency = [13+1.5,10+1.5,8+1.5]
    # c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=b_mod(-1.5)
    # init_c(400)
    # c.data_context.settings.SAVE_IMAGES=True
    # yield None

    # experiments = 0
    # b_mod_values=[-3.0,-2.5,-2.0,-1.5,0.0,1.5,2.0,2.5,3.0]
    # b_names=['-3.0', '-2.5', '-2.0', '-1.5', '', '+1.5', '+2.0', '+2.5', '+3.0']
    # for values,name in zip(b_mod_values,b_names):
    #     if experiments%3!=2 or experiments<6:
    #         experiments+=1
    #         print(experiments)
    #         continue

    #     init_small_idealized(medium_n5_time=45,suffix=f"b{name}")
    #     c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=b_mod(values)
    #     init_c(300)
    #     print(c.data_context.global_simulation.growth_coeff_development.coeffs_list)
    #     c.data_context.settings.SAVE_IMAGES=True
    #     experiments+=1
    #     yield None
    # inhibition_coeffs=[0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2]
    # names=["0.7", "1.2", "1.7", "2.2", "2.7", "3.2", "3.7", "4.2"]
    # experiments = 0
    # for coeff,name in zip(inhibition_coeffs,names):
    #     coeffs=coeffs_mod('inhibition_coef',coeff-2.7)
    #     init_small_idealized(medium_n5_time=45,suffix=f"inhibition={name}_v",coeffs=coeffs)
    #     init_c(300)
    #     print(c.data_context.global_simulation.growth_coeff_development.coeffs_list)
    #     #c.data_context.settings.SAVE_IMAGES=Galse
        
    #     experiments+=1
    #     yield None
    # experiments = 0
    # for j in range(0,3):
    #     for i in range (0,3):
    #         experiments+=1
    #         if not (experiments >7):
    #             print(experiments)
    #             continue

    #         plastochron_mod=[-1.5,0,1.5]
    #         plastochron_name=["-1.5","","+1.5"]
    #         b_mod_value=[-1.5,0,1.5]
    #         b_name=["-1.5","","+1.5"]
    #         init_small_idealized(medium_n5_time=45,suffix=f"p{plastochron_name[i]} b{b_name[j]}_2")
    #         c.data_context.global_simulation.growth_coeff_development.surface_frequency = [13+plastochron_mod[i],10+plastochron_mod[i],8+plastochron_mod[i]]
    #         c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=b_mod(b_mod_value[j])
    #         init_c(300)
    #         print(c.data_context.global_simulation.growth_coeff_development.coeffs_list)
    #         c.data_context.settings.SAVE_IMAGES=True
    #         yield None
#     init_small_no_growth(suffix="2")
#     c.data_context.global_simulation.growth_coeff_development.n8_ages=[2]
#     c.data_context.settings.SAVE_IMAGES=True
#     yield None
#     init_c(440)
#     init_small_no_growth(suffix="_plastochron_change2")
#     c.data_context.global_simulation.growth_coeff_development.surface_frequency = [12,10,8]
#     c.data_context.global_simulation.growth_coeff_development.n8_ages=[2]
#     c.data_context.settings.SAVE_IMAGES=True
#     yield None
#     init_c(440)
#     init_small_no_growth(suffix="_growth_change2")
#     c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=grow_coeffs=[{'A': 0.2614290317628093, 'B': 7.062656265881261, 'C': 35.85128301663676},
# {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
# #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
# {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
#     c.data_context.global_simulation.growth_coeff_development.n8_ages=[2]
#     c.data_context.settings.SAVE_IMAGES=True
#     yield None
#     init_c(440)
#     init_small_no_growth(suffix="_change_both2")
#     c.data_context.global_simulation.growth_coeff_development.growth_coeffs_list=grow_coeffs=[{'A': 0.2614290317628093, 'B': 7.062656265881261, 'C': 35.85128301663676},
# {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
# #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
# {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
#     c.data_context.global_simulation.growth_coeff_development.surface_frequency = [12,10,8]
#     c.data_context.global_simulation.growth_coeff_development.n8_ages=[2]
#     n8_ages=[1]
#     yield None

    # init_small_no_growth(suffix="_ring",coeffs={'pos_range': 20.6139871413999,
    #     'age_coef': 0.03963136053954481,
    #     'inhibition_coef': 2.9873971411850995,
    #     'attraction_coef': 0.0,
    #     'neg_range': 59.22591211160096,
    #     'straight_coef': 3.0624121912173043,
    #     'inertia': 0.5,
    #     #'inertia': 1.,
    #     'peak_coef': 0.99,
    #     'age_cut_off_coef': 0.8961433833351786})
    # init_single_test(coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             #'age_coef': 0.0,
    #             'inhibition_coef': 3.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 84.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786},suffix="2")
    #c.data_context.global_simulation.surface_points.angle_error=360
    #c.data_context.settings.UPDATE_COEFFS_WITH_TIME=True
    #_1 has gs.surface_points.skip_primordiums=4.8  c.data_context.global_simulation.growth_coeff_development.n5_ages=[4,4]
    #_2 has gs.surface_points.skip_primordiums=1.8 c.data_context.global_simulation.growth_coeff_development.n5_ages=[3,3]

    yield None
    yield None
    # def generate_coeffs(a,b):
    #     return {'pos_range': 20.6139871413999,
    #     'age_coef': 0.03963136053954481,
    #     'inhibition_coef': 2.7873971411850995+b,
    #     'attraction_coef': 0.0,
    #     'neg_range': 59.22591211160096 + a,
    #     'straight_coef': 3.0624121912173043,
    #     'inertia': 0.5,
    #     #'inertia': 1.,
    #     'peak_coef': 0.99,
    #     'age_cut_off_coef': 0.8961433833351786}
    # # coeffss = [generate_coeffs(a,b) for (a,b) in [(0, 0),
    # #                                               (0, 0.1),
    # #                                               (0, -0.1),
    # #                                               (0, -0.3),
    # #                                               (0, 0.3),
    # #                                               (1, 0),
    # #                                               (1, 0.1),
    # #                                               (-1, 0),
    # #                                               (-1, -0.1),
    # #                                               (-2, 0),
    # #                                               (-2, -0.3),
    # #                                               (2, 0.3)]]
    # coeffss = [generate_coeffs(a,b) for (a,b) in [(0, 0),
    #                                               (0, 0.1),
    #                                               (0, -0.1),
    #                                               (0, -0.3),
    #                                               (0, 0.3),
    #                                               (1, 0),
    #                                               (1, 0.1),
    #                                               (-1, 0),
    #                                               (-1, -0.1),
    #                                               (-2, 0),
    #                                               (-2, -0.3),
    #                                               (2, 0.3),
    #                                               (0, 0.4),
    #                                               (0, 0.5),
    #                                               (0, 0.6),
    #                                               (0, 0.7),
    #                                               (0, 0.8),]]
    # print(coeffss)

    # for i,coeffs in enumerate(coeffss):
    #     if i!=13:
    #         continue
    #     init_bijugate_idealized(suffix=f"_{i}_no_text",coeffs=coeffs)
    #     init_c(440)
    #     yield None
    # init_small_idealized(medium_n5_time=45)
    # init_c(440)
    # yield None

# def run_experiments():
#     for i in range(6,9):
#         age = 1-i/4
#         init_small_idealized(suffix=f"matured_age={age:.2f}")
#         c.data_context.global_simulation.MATURED_AGE=age
#         init_c()
#         yield None
    # init_small_idealized(suffix="inhibition_coef_17 b-15",coeffs=coeffs_mod('inhibition_coef',-1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(-1.5)
    # init_c()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_17 b",coeffs=coeffs_mod('inhibition_coef',-1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(0)
    # init_c()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_17 b+15",coeffs=coeffs_mod('inhibition_coef',-1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_37 b-15",coeffs=coeffs_mod('inhibition_coef',1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(-1.5)
    # init_c()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_37 b",coeffs=coeffs_mod('inhibition_coef',1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(0)
    # init_c()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_37 b+15",coeffs=coeffs_mod('inhibition_coef',1.0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(1.5)
    # init_c()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_22 b-15",coeffs=coeffs_mod('inhibition_coef',-0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(-1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_22 b+15",coeffs=coeffs_mod('inhibition_coef',-0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_32 b-15",coeffs=coeffs_mod('inhibition_coef',0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(-1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_32 b+15",coeffs=coeffs_mod('inhibition_coef',0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_27 b-15",coeffs=coeffs_mod('inhibition_coef',0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(-1.5)
    # init_c()
    # yield None

    # init_small_idealized(suffix="inhibition_coef_27 b+15",coeffs=coeffs_mod('inhibition_coef',0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(1.5)
    # init_c()
    # yield None
    
    # init_small_idealized(suffix="inhibition_coef_22 b",coeffs=coeffs_mod('inhibition_coef',-0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(0)
    # init_c()
    # yield None
    
    # init_small_idealized(suffix="inhibition_coef_27 b",coeffs=coeffs_mod('inhibition_coef',0))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(0)
    # init_c()
    # yield None
    
    # init_small_idealized(suffix="inhibition_coef_32 b",coeffs=coeffs_mod('inhibition_coef',0.5))
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=b_mod(0)
    # init_c()
    # init_small_idealized(suffix="inhibition_coef_22",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.2873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_27",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_69",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 69.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_49",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 49.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="meristem_growth b+1")
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 8.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 7.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="meristem_growth b-1")
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 6.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 5.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_49 b+1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 49.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 8.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 7.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_49 b-1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 49.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 6.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 5.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_69 b+1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 69.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 8.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 7.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="neg_range_69 b-1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.7873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 69.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 6.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 5.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_32 b+1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 3.2873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 8.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 7.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_32 b-1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 3.2873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 6.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 5.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_22 b+1",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.2873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 8.062656265881261, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 7.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625), 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    # init_small_idealized(suffix="inhibition_coef_22 b-15",coeffs={'pos_range': 20.6139871413999,
    #             'age_coef': 0.03963136053954481,
    #             'inhibition_coef': 2.2873971411850995,
    #             'attraction_coef': 0.0,
    #             'neg_range': 59.22591211160096,
    #             'straight_coef': 3.0624121912173043,
    #             'inertia': 0.5,
    #             #'inertia': 1.,
    #             'peak_coef': 0.99,
    #             'age_cut_off_coef': 0.8961433833351786})
    # c.data_context.global_simulation.growth_coeff_development.coeffs_list=[{'A': 0.2614290317628093, 'B': 7.062656265881261-2, 'C': 35.85128301663676},
    #     {'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25)-1, 'C': 46.3787741621418},
    #     #{'A': 0.17592879728901245*(1.625/1.25)**2, 'B': 6.563675767507394*(1.625/1.25), 'C': 46.3787741621418},
    #     {'A': 0.11900277137012134*(1.625)**2, 'B': 5.910352308336133*(1.625)-1, 'C': 56.38910353160223}]
    # c.simulation_controller=SimulationController()
    # c.simulation_controller.screenshots_number=220
    # c.simulation_controller.do_save_images=True
    # c.simulation_controller.run_calculation_thread()
    # yield None
    
