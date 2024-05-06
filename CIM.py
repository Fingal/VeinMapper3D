import re
from math import ceil
import pickle, copy, time
from sys import prefix
from itertools import product
from random import sample
import typing
from skeleton import *
from stack_io import *
from calculation_config import *
import context, generate_line
import random

"""local simulation used for finding parameters to  global simulation """

class CIM:

    def __init__(self):
        self.context = context.Context()

    def _generate_lines(self, _labels, inertia=2, pos_range=120, neg_range=50, inhibition_coef=1, attraction_coef=1, age_coef=0.1, age_cut_off_coef=4, straight_coef=10, peak_coef=0.5, draw=False,mode="max"):
        line_length = lambda a, b: sum(((x - y) ** 2 for x, y in zip(a, b))) * 0.5
        result = []
        other_pos = []
        difference = []
        if self.context.data_context.values['center'] is not None:
            center = self.context.canvas._stack_coord_to_micron(self.context.data_context.values['center'])
        else:
            center = (0, 0, 0)
            size = 0
            for i in range(-2, 4):
                point = self.context.data_context.get_point_from_label(str(i))
                if point is not None:
                    center = tuple((a + b for a, b in zip(center, point)))
                    size += 1

            center = tuple((a / size for a in center))
            center = self.context.canvas._stack_coord_to_micron(center)
        for labels in _labels:
            lines = []
            try:
                for l in labels[1:]:
                    point = self.context.data_context.get_points_from_label(l)[0]
                    graph = find_compact_component(self.context.canvas.skeleton_graph, point)
                    lines.append((self.context.canvas._lines_coord_to_microns(graph_to_lines(graph)), int(l)))

            except IndexError:
                continue

            def _calculation(sufix):
                # s or st no idea now suffix sometiems is at the beggining or at the end
                _start = self.context.data_context.get_points_from_label(f"{labels[0]}s{sufix}")
                if not _start:
                    _start = self.context.data_context.get_points_from_label(f"{labels[0]}st{sufix}")
                    if not _start:
                        return
                _start = _start[0]
                start = self.context.canvas._stack_coord_to_micron(_start)
                _target = []
                for label in [f"{labels[0]}e", f"{labels[0]}e{sufix}", (f"{labels[0]}")]:
                    _target = self.context.data_context.get_points_from_label(label)
                    if _target:
                        _target = _target[0]
                        break
                if not _target:
                    return
                target = self.context.canvas._stack_coord_to_micron(_target)
                t = time.time()
                original_line = []
                graph = find_compact_component(self.context.canvas.skeleton_graph, _target)
                original_line = self.context.canvas._lines_coord_to_microns(graph_to_lines(graph))
                _result, _other_pos = generate_line.generate_model_line(start, target, original_line, lines, inertia, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef, age_cut_off_coef, straight_coef, peak_coef, center)
                t = time.time()
                cutoff = -1
                if len(_result) >= 1000:
                    for i in range(100, 970):
                        distance = lambda start, end: sum(((x - y) ** 2 for x, y in zip(start, end))) ** 0.5
                        try:
                            if distance(_result[i], result[(i + 20)]) < 5:
                                cutoff = i
                                break
                        except IndexError:
                            pass

                if cutoff > -1:
                    _result = _result[:cutoff]
                    print(f"cutting in {cutoff}")

                original = self.context.canvas._lines_coord_to_microns(find_primary(self.context.canvas.skeleton_graph, _start, _target))
                length_cost = 0
                original_length = generate_line.line_segment_length(original_line)
                predicted_length = generate_line.line_segment_length(list(zip(_result, _result[1:])))
                if original_length < predicted_length:
                    length_cost = (predicted_length / original_length - 1) ** 2 * 20
                else:
                    l = sum(((x - y) ** 2 for x, y in zip(start, target))) * 0.5
                    _c = 50
                    if original_length - l < 10:
                        _c = 10
                    if original_length - l > 4:
                        length_cost = (original_length - predicted_length) / (original_length - l) * _c

                end_distance = sum(((x - y) ** 2 for x, y in zip(_result[(-1)], target)))
                distances = generate_line.calculate_distance(_result, original)

                ### check with avg_d
                max_d = max([d ** 2 for d in distances])
                try:
                    avg_d = (sum([d ** 2 for d in distances]) / len(distances)) ** 0.5
                except:
                    avg_d = sum([d ** 2 for d in distances]) ** 0.5
                if mode=="avg":
                    diff = avg_d
                else:
                    diff = max_d
                #diff = avg_d + end_distance + length_cost
                difference.append(diff)
                if draw:
                    _result = [self.context.canvas._micron_coord_to_stack(i) for i in _result]
                    _other_pos = [self.context.canvas._micron_coord_to_stack(i) for i in _other_pos]
                    result.append(_result)
                    other_pos.extend(_other_pos)
                    self.context.graphic_context.mark_temporary_paths(result)

            _calculation('')
            _calculation('r')
            _calculation('l')
            _calculation('R')
            _calculation('L')

        if draw:
            self.context.graphic_context.mark_temporary_paths(result)
            self.context.graphic_context.mark_temporary_points(other_pos)
        d_value = (sum([d ** 2 for d in difference]) / len(difference)) ** 0.5
        return (result, d_value)
    def generate_line_segment(self, labels, inertia=2, pos_range=120, neg_range=50, inhibition_coef=1, attraction_coef=1, age_coef=0.1, age_cut_off_coef=4, straight_coef=10, peak_coef=0.5, draw=False):
        line_length = lambda a, b: sum(((x - y) ** 2 for x, y in zip(a, b))) * 0.5
        result = []
        other_pos = []
        difference = []
        if self.context.data_context.values['center'] is not None:
            center = self.context.canvas._stack_coord_to_micron(self.context.data_context.values['center'])
        else:
            center = (0, 0, 0)
            size = 0
            for i in range(-2, 4):
                point = self.context.data_context.get_point_from_label(str(i))
                if point is not None:
                    center = tuple((a + b for a, b in zip(center, point)))
                    size += 1

            center = tuple((a / size for a in center))
            center = self.context.canvas._stack_coord_to_micron(center)
        lines = []
        try:
            for l in labels[1:]:
                point = self.context.data_context.get_points_from_label(l)[0]
                graph = find_compact_component(self.context.canvas.skeleton_graph, point)
                lines.append((self.context.canvas._lines_coord_to_microns(graph_to_lines(graph)), int(l)))

        except IndexError:
            pass

        def _calculation(sufix):
            # s or st no idea now suffix sometiems is at the beggining or at the end
            _start = self.context.data_context.get_points_from_label(f"{labels[0]}s{sufix}")
            if not _start:
                _start = self.context.data_context.get_points_from_label(f"{labels[0]}st{sufix}")
                if not _start:
                    return
            _start = _start[0]
            start = self.context.canvas._stack_coord_to_micron(_start)
            _target = []
            for label in [f"{labels[0]}e", f"{labels[0]}e{sufix}", (f"{labels[0]}")]:
                _target = self.context.data_context.get_points_from_label(label)
                if _target:
                    _target = _target[0]
                    break
            if not _target:
                return
            target = self.context.canvas._stack_coord_to_micron(_target)
            t = time.time()
            original_line = []
            graph = find_compact_component(self.context.canvas.skeleton_graph, _target)
            original_line = self.context.canvas._lines_coord_to_microns(graph_to_lines(graph))
            _result, _other_pos = generate_line.generate_model_line(start, target, original_line, lines, inertia, pos_range, neg_range, inhibition_coef, attraction_coef, age_coef, age_cut_off_coef, straight_coef, peak_coef, center)
            t = time.time()
            cutoff = -1
            if len(_result) >= 1000:
                for i in range(100, 970):
                    distance = lambda start, end: sum(((x - y) ** 2 for x, y in zip(start, end))) ** 0.5
                    try:
                        if distance(_result[i], result[(i + 20)]) < 5:
                            cutoff = i
                            break
                    except IndexError:
                        pass

            if cutoff > -1:
                _result = _result[:cutoff]
                print(f"cutting in {cutoff}")

            original = self.context.canvas._lines_coord_to_microns(find_primary(self.context.canvas.skeleton_graph, _start, _target))
            result.append((labels,sufix,_result,original))
        _calculation('')
        _calculation('r')
        _calculation('l')
        _calculation('R')
        _calculation('L')

        return (result)

    def space_exploration(self, nums=[5, 20, 20, 20], out_filename='exploration_results\\dump_old_4', min_value=None, max_value=None):
        files = []
        file_numbers = []
        labels_list = []
        for x in self.calculation_config:
            #_file_numbers = x.file_numbers
            _file_numbers = x.filenames
            file_numbers.extend(_file_numbers)
            # number = x.number
            # for path in [f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{number}_dr{n}_mark.ske" for n in _file_numbers]:
            #     with open(path, 'rb') as (file):
            #         files.append(pickle.load(file))
            for path in [f"{x.path}{n}" for n in _file_numbers]:
                with open(path, 'rb') as (file):
                    files.append(pickle.load(file))

            labels_list.extend(x.labels_list)

        def perform_calc(labels, v):
            result = []
            i = 0
            for number, file, labels in zip(file_numbers, files, labels_list):
                if not number == 50:
                    if number == 48 or number == 31 or number == 43:
                        continue
                    self.context.data_context._load_data_no_graphics(file)
                    _, _d = (self._generate_lines)(labels, **v)
                    result.append(_d)

            return (sum([d ** 2 for d in result]) / len(result)) ** 0.5

        results = []
        i = 0
        names = ['pos_range', 'attraction_coef', 'neg_range', 'inhibition_coef',"inertia","age_coef"]
        print(nums)
        all_posibilities = reduce(lambda x, y: x * y, nums)
        start_time = time.time()
        for values in self._generate_values(nums, names=names, min_value=min_value, max_value=max_value):
            d_value = perform_calc(labels_list, values)
            results.append((values, d_value))
            i += 1
            if i % 50 == 0:
                print(f"done {i / all_posibilities * 100:.3f}%")
            if i % 200 == 0:
                print(f"time passed {time.time() - start_time}")

        results.sort(key=(lambda x: x[1]))
        print(out_filename)
        with open(out_filename, 'wb') as (file):
            pickle.dump(results[:20000], file)

    def explore_given(self,coeffs, out_filename='exploration_results\\dump_old_4',all_items = None):
        files = []
        file_numbers = []
        labels_list = []
        # for x in self.calculation_config:
        #     _file_numbers = x.file_numbers
        #     file_numbers.extend(_file_numbers)
        #     number = x.number
        #     for path in [f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{number}_dr{n}_mark.ske" for n in _file_numbers]:
        #         with open(path, 'rb') as (file):
        #             files.append(pickle.load(file))

        #     labels_list.extend(x.labels_list)
        
        for x in self.calculation_config:
            #_file_numbers = x.file_numbers
            _file_numbers = x.filenames
            file_numbers.extend(_file_numbers)
            # number = x.number
            # for path in [f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{number}_dr{n}_mark.ske" for n in _file_numbers]:
            #     with open(path, 'rb') as (file):
            #         files.append(pickle.load(file))
            for path in [f"{x.path}{n}" for n in _file_numbers]:
                with open(path, 'rb') as (file):
                    files.append(pickle.load(file))

            labels_list.extend(x.labels_list)

        def perform_calc(labels, v):
            result = []
            i = 0
            for number, file, labels in zip(file_numbers, files, labels_list):
                if not number == 50:
                    if number == 48 or number == 31 or number == 43:
                        continue
                    self.context.data_context._load_data_no_graphics(file)
                    _, _d = (self._generate_lines)(labels, **v)
                    result.append(_d)

            return (sum([d ** 2 for d in result]) / len(result)) ** 0.5

        results = []
        i = 0
        names = ['pos_range', 'attraction_coef', 'neg_range', 'inhibition_coef']
        start_time = time.time()
        if all_items is None:
            all_items = len(coeffs)
        for values in coeffs:
            d_value = perform_calc(labels_list, values)
            results.append((values, d_value))
            i += 1
            if i % 50 == 0:
                print(f"done {i / all_items * 100:.3f}%")
            if i % 200 == 0:
                print(f"time passed {time.time() - start_time}")

        results.sort(key=(lambda x: x[1]))
        print(out_filename)
        with open(out_filename, 'wb') as (file):
            pickle.dump(results, file)
            
    def explore_given_path(self,coeffs, out_filename='exploration_results\\dump_old_4',all_items = None):
        files = []
        file_numbers = []
        labels_list = []
        for x in self.calculation_config:
            file_numbers.extend(x.filenames)
            for path in [f"{x.path}{filename}" for filename in x.filenames]:
                with open(path, 'rb') as (file):
                    files.append(pickle.load(file))

            labels_list.extend(x.labels_list)
            
        

        def perform_calc(labels, v):
            result = []
            i = 0
            for number, file, labels in zip(file_numbers, files, labels_list):
                if not number == 50:
                    if number == 48 or number == 31 or number == 43:
                        continue
                    self.context.data_context._load_data_no_graphics(file)
                    _, _d = (self._generate_lines)(labels, **v)
                    result.append(_d)

            return (sum([d ** 2 for d in result]) / len(result)) ** 0.5

        results = []
        i = 0
        names = ['pos_range', 'attraction_coef', 'neg_range', 'inhibition_coef']
        start_time = time.time()
        if all_items is None:
            all_items = len(coeffs)
        for values in coeffs:
            d_value = perform_calc(labels_list, values)
            results.append((values, d_value))
            i += 1
            if i % 50 == 0:
                print(f"done {i / all_items * 100:.3f}%")
            if i % 200 == 0:
                print(f"time passed {time.time() - start_time}")

        results.sort(key=(lambda x: x[1]))
        print(out_filename)
        with open(out_filename, 'wb') as (file):
            pickle.dump(results, file)

    def _generate_values(self, nums=[
 20] * 4, names = ['pos_range', 'attraction_coef', 'neg_range', 'inhibition_coef'], min_value=None, max_value=None):
        result = {'pos_range':39.47522510201009,  'age_coef':0.07000000000000005,  'inhibition_coef':3.2636629653532916,  'attraction_coef':0.6403846635475494,  'neg_range':34.87783365037785,  'straight_coef':2.5560733141578544,  'inertia':1.5,  'peak_coef':0.4991447183847797,  'age_cut_off_coef':1.3757699971976245}
        min_step = {'pos_range':2, 
         'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.1,  'attraction_coef':0.1,  'neg_range':1,  'straight_coef':0.1,  'inertia':0.2,  'peak_coef':0.02,  'age_cut_off_coef':0.05}
        if min_value == None:
            min_value = {'pos_range':20,'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.01,  'attraction_coef':0.01,  'neg_range':10,  'straight_coef':0.3,  'inertia':0.5,  'peak_coef':0.01,  'age_cut_off_coef':-10}
        if max_value == None:
            max_value = {'pos_range':140,'age_coef':10.0,  'inhibition_coef':5.0,  'attraction_coef':5.0,  'neg_range':100,  'straight_coef':20,  'inertia':500,  'peak_coef':0.99,  'age_cut_off_coef':20}
        iters = [np.linspace(min_value[name], max_value[name], steps) for steps, name in zip(nums, names)]
        for values in product(*iters):
            _result = copy.copy(result)
            for name, value in zip(names, values):
                _result[name] = value

            yield _result

    def _test(self, steps=1000, values={'strongest':5,  'attraction_coef':3,  'neg_range':30,  'straight_coef':5,  'inertia':30}, filename='results_2.txt', draw=False,max_jumps=5):
        files = []
        file_numbers = []
        labels_list = []
        for x in self.calculation_config:
            _file_numbers = x.file_numbers
            file_numbers.extend(_file_numbers)
            number = x.number
            for path in [f"C:\\Users\\Andrzej\\Desktop\\2send_1\\reconstructions_ske\\ske_marked\\new\\only_joined\\xp{number}_dr{n}_mark.ske" for n in _file_numbers]:
                with open(path, 'rb') as (file):
                    files.append(pickle.load(file))

            labels_list.extend(x.labels_list)

        def perform_calc(labels, v):
            result = []
            i = 0
            for number, file, labels in zip(file_numbers, files, labels_list):
                if not number == 50:
                    if number == 48 or number == 31 or number == 43:
                        continue
                    self.context.data_context._load_data_no_graphics(file)
                    _, _d = (self._generate_lines)(labels, **v)
                    result.append(_d)

            return (sum([d ** 2 for d in result]) / len(result)) ** 0.5

        d_value = perform_calc(labels_list, values)
        min_step = {'pos_range':2, 
         'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.1,  'attraction_coef':0.1,  'neg_range':1,  'straight_coef':0.1,  'inertia':0.2,  'peak_coef':0.02,  'age_cut_off_coef':0.05}
        #min_value = {'pos_range':2,  'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.01,  'attraction_coef':0.01,  'neg_range':1,  'straight_coef':0.3,  'inertia':0.5,  'peak_coef':0.01,  'age_cut_off_coef':-10}
        #max_value = {'pos_range':300,'age_coef':10.0,  'inhibition_coef':20.0,  'attraction_coef':20.0,  'neg_range':100,  'straight_coef':20,  'inertia':500,  'peak_coef':0.99,  'age_cut_off_coef':20}
        min_value = {'pos_range':2,  'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.01,  'attraction_coef':0.01,  'neg_range':0,  'straight_coef':0.3,  'inertia':0.5,  'peak_coef':0.01,  'age_cut_off_coef':-10}
        max_value = {'pos_range':300,'age_coef':10.0,  'inhibition_coef':20.0,  'attraction_coef':20.0,  'neg_range':120,  'straight_coef':20,  'inertia':500,  'peak_coef':0.99,  'age_cut_off_coef':20}

        def next_steps(values, min_step=min_step, pick=4):
            other = sample(range(len(values.keys())), len(values.keys()) - pick)
            result = [{key:max(min_value[key], min(max_value[key], val + t[i] * max(val * 0.05, min_step.get(key, 0)))) for i, (key, val) in enumerate(values.items())} for t in product([-1, 0, 1], repeat=(len(values.keys()))) if len([v for v in other if t[v] != 0]) == 0]
            duplicates = []
            for i, item in enumerate(result):
                for j, second in enumerate(result[i + 1:]):
                    same = True
                    for k, v in item.items():
                        if second[k] != v:
                            same = False
                            break

                    if same:
                        duplicates.append(j)

            result = [i for j, i in enumerate(result) if j not in duplicates]
            return result

        possible_values = next_steps(values)
        jumps = 0
        MAX_TRIES = 5
        file = open(filename, mode='a')
        file.write('new test\n')
        file.write(f"starts with {values}\n\n")
        file.close()
        visited = []
        t = time.time()
        for j in range(max_jumps):
            tries = 0
            for _i in range(steps):
                print('time', time.time() - t)
                temp_d = []
                print(len(possible_values))
                for _v in possible_values:
                    _d = perform_calc(labels_list, _v)
                    temp_d.append(_d)

                best_d = min(temp_d)
                best_index = temp_d.index(best_d)
                best = possible_values[best_index]
                if best_d / d_value > 0.9995:
                    tries += 1
                    print(f"nothing better found for {tries} time(s)")
                else:
                    print(f"step {_i} best so far {min(temp_d)}")
                    tries = 0
                    jumps = 0
                    d_value = best_d
                    possible_values = next_steps(best)

                if tries>MAX_TRIES or jumps>0:
                    jumps+=1
                    break

            if tries > MAX_TRIES or jumps > 0:
                print(f"jump, try number {j}")
                if jumps == 1:
                    file = open(filename, mode='a')
                    file.write(f"{best}\n")
                    file.write(f"best so far {best_d}\n")
                    file.close()
                    possible_values = []
                    for key in best.keys():
                        for x in (-1, 1):
                            result = dict(best)
                            val = result[key]
                            result[key] = max(min_value[key], min(max_value[key], val + x * max(val * 0.05, min_step.get(key, 0))))
                            possible_values.append(result)

                if jumps > 1:
                    possible_values=next_steps(best,min_step={k:v*(1+(jumps-1)*2) for k,v in min_step.items()},pick=len(best))
                    possible_values=sample(possible_values,1200)
                    possible_values=list(filter(lambda x: x not in visited, possible_values))
                    visited.extend(possible_values)
                    _possible_values=[]
                    _min_step={k:v*(10+(jumps)*4) for k,v in min_step.items()}
                    for key in (best.keys()):
                        for x in [-1,1]:
                            result=dict(best)
                            val=result[key]
                            result[key]=max(min_value[key],min(max_value[key],val+x*max(val*0.05,_min_step.get(key,0))))
                            _possible_values.append(result)
                    _possible_values=list(filter(lambda x: x not in visited, _possible_values))
                    possible_values.extend(_possible_values)
                best_d = best_d * 1.1

        if draw:
            (self.context.data_context._generate_lines)(labels_list[(-1)], **best, **{'draw': True})
            self.context.canvas.redraw_graph()
            self.context.data_context.reload_all()
        file = open(filename, mode='a')
        file.write('finished\n\n')
        file.close()
        return (best,d_value)    

    def optimize(self, steps=1000, values={'strongest':5,  'attraction_coef':3,  'neg_range':30,  'straight_coef':5,  'inertia':30}, filename='results_2.txt', draw=False,max_jumps=5):
        files = []
        file_numbers = []
        labels_list = []

        for x in self.calculation_config:
            file_numbers.extend(x.filenames)
            for path in [f"{x.path}{filename}" for filename in x.filenames]:
                with open(path, 'rb') as (file):
                    files.append(pickle.load(file))

            labels_list.extend(x.labels_list)

        def perform_calc(labels, v):
            result = []
            i = 0
            for number, file, labels in zip(file_numbers, files, labels_list):
                if not number == 50:
                    if number == 48 or number == 31 or number == 43:
                        continue
                    self.context.data_context._load_data_no_graphics(file)
                    _, _d = self._generate_lines(labels, **v)
                    result.append(_d)

            return (sum([d ** 2 for d in result]) / len(result)) ** 0.5

        d_value = perform_calc(labels_list, values)
        min_step = {'pos_range':2, 
         'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.1,  'attraction_coef':0.1,  'neg_range':1,  'straight_coef':0.1,  'inertia':0.2,  'peak_coef':0.02,  'age_cut_off_coef':0.05}
        #min_value = {'pos_range':2,  'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.01,  'attraction_coef':0.01,  'neg_range':1,  'straight_coef':0.3,  'inertia':0.5,  'peak_coef':0.01,  'age_cut_off_coef':-10}
        #max_value = {'pos_range':300,'age_coef':10.0,  'inhibition_coef':20.0,  'attraction_coef':20.0,  'neg_range':100,  'straight_coef':20,  'inertia':500,  'peak_coef':0.99,  'age_cut_off_coef':20}
        min_value = {'pos_range':2,  'strongest':0.1,  'age_coef':0.01,  'inhibition_coef':0.01,  'attraction_coef':0.01,  'neg_range':0,  'straight_coef':0.3,  'inertia':0.5,  'peak_coef':0.01,  'age_cut_off_coef':-10}
        max_value = {'pos_range':300,'age_coef':10.0,  'inhibition_coef':20.0,  'attraction_coef':20.0,  'neg_range':120,  'straight_coef':20,  'inertia':500,  'peak_coef':0.99,  'age_cut_off_coef':20}

        def next_steps(values, min_step=min_step, pick=4,mul=1):
            other = sample(range(len(values.keys())), len(values.keys()) - pick)
            result = [{key:max(min_value[key], min(max_value[key], val + mul*t[i] * max(val * 0.05, min_step.get(key, 0)))) for i, (key, val) in enumerate(values.items())} for t in product([-1, 0, 1], repeat=(len(values.keys()))) if len([v for v in other if t[v] != 0]) == 0]
            duplicates = []
            for i, item in enumerate(result):
                for j, second in enumerate(result[i + 1:]):
                    same = True
                    for k, v in item.items():
                        if second[k] != v:
                            same = False
                            break

                    if same:
                        duplicates.append(j)

            result = [i for j, i in enumerate(result) if j not in duplicates]
            return result

        possible_values = next_steps(values)
        jumps = 0
        MAX_TRIES = 4
        file = open(filename, mode='a')
        file.write('new test\n')
        file.write(f"starts with {values}\n\n")
        file.close()
        visited = []
        t = time.time()
        best = values
        for j in range(max_jumps+1):
            tries = 0
            for _i in range(steps):
                print('time', time.time() - t)
                temp_d = []
                print(len(possible_values))
                for _v in possible_values:
                    _d = perform_calc(labels_list, _v)
                    temp_d.append(_d)

                best_d = min(temp_d)
                best_index = temp_d.index(best_d)
                best = possible_values[best_index]
                if best_d / d_value > 0.9995:
                    tries += 1
                    print(f"nothing better found for {tries} time(s)")
                    possible_values = next_steps(best,mul=1/(tries**1.5))
                else:
                    print(f"step {_i} best so far {min(temp_d)}")
                    tries = 0
                    jumps = 0
                    d_value = best_d
                    possible_values = next_steps(best)

                if tries>MAX_TRIES or jumps>0:
                    jumps+=1
                    break
            if jumps>=max_jumps:
                break
            if tries > MAX_TRIES or jumps > 0:
                print(f"jump, try number {j}")
                if jumps == 1:
                    file = open(filename, mode='a')
                    file.write(f"{best}\n")
                    file.write(f"best so far {best_d}\n")
                    file.close()
                    possible_values = []
                    for key in best.keys():
                        for x in (-1, 1):
                            result = dict(best)
                            val = result[key]
                            result[key] = max(min_value[key], min(max_value[key], val + x * max(val * 0.05, min_step.get(key, 0))))
                            possible_values.append(result)

                if jumps > 1:
                    possible_values=next_steps(best,min_step={k:v*(1+(jumps-1)*2) for k,v in min_step.items()},pick=len(best))
                    possible_values=sample(possible_values,1200)
                    possible_values=list(filter(lambda x: x not in visited, possible_values))
                    visited.extend(possible_values)
                    _possible_values=[]
                    _min_step={k:v*(10+(jumps)*4) for k,v in min_step.items()}
                    for key in (best.keys()):
                        for x in [-1,1]:
                            result=dict(best)
                            val=result[key]
                            result[key]=max(min_value[key],min(max_value[key],val+x*max(val*0.05,_min_step.get(key,0))))
                            _possible_values.append(result)
                    _possible_values=list(filter(lambda x: x not in visited, _possible_values))
                    possible_values.extend(_possible_values)
                best_d = best_d * 1.1

        if draw:
            (self.context.data_context._generate_lines)(labels_list[(-1)], **best, **{'draw': True})
            self.context.canvas.redraw_graph()
            self.context.data_context.reload_all()
        file = open(filename, mode='a')
        file.write(f"best result\n{best}")
        file.write(f"best value{d_value}")
        file.write('finished\n\n')
        file.close()
        return (best,d_value)


    def generate_single_line(self,filepath,labels,config):
        c = context.Context()
        with open(filepath,"rb") as file:
            data=pickle.load(file)
            try:
                c.data_context.load_data(data)
            except:
                c.data_context._load_data_no_graphics(pickle.load(file))
            return self.generate_line_segment(labels,**config)

if __name__ == '__main__':
    import canvas_mokup as canvas, sys
    c = context.Context()
    c.canvas = canvas.OpenGLCanvas(None)

    def run_space_exploration():
        print(sys.argv)
        nums = eval(sys.argv[1])
        out_filename = sys.argv[2]
        config = sys.argv[3]
        c.data_context.CIM.calculation_config = config_dict[config]
        min_value = eval(sys.argv[4])
        max_value = eval(sys.argv[5])
        print(max_value)
        c.data_context.CIM.space_exploration(nums, out_filename, min_value, max_value)
        print('finished')

    def run_exploration_from_file():
        print(sys.argv)
        config = sys.argv[1]
        filepath = sys.argv[2]
        out_filename = sys.argv[3]
        c.data_context.CIM.calculation_config = config_dict[config]
        start_index = eval(sys.argv[4])
        end_index = eval(sys.argv[5])
        coeffs = []
        with open(filepath,'rb') as file:
            coeffs = pickle.load(file)[start_index:end_index]
        c.data_context.CIM.explore_given(coeffs, out_filename)
        print('finished')

    def run_random_exploration_from_file():
        print(sys.argv)
        config = sys.argv[1]
        filepath = sys.argv[2]
        out_filename = sys.argv[3]
        c.data_context.CIM.calculation_config = config_dict[config]
        start_index = eval(sys.argv[4])
        end_index = eval(sys.argv[5])
        try:
            step = eval(sys.argv[6])
        except:
            step = 1
        #amount = eval(sys.argv[4])
        coeffs = []
        with open(filepath,'rb') as file:
            #coeffs = list(map(lambda x: x[0],random.choices(pickle.load(file),k=amount)))
            coeffs = pickle.load(file)[start_index:end_index:step]
        c.data_context.CIM.explore_given_path(coeffs, out_filename)
        print('finished')

    


    def run_optimalization():
        config = sys.argv[1]
        in_filename = sys.argv[2]
        out_filename = sys.argv[3]
        start = int(sys.argv[4])
        end = int(sys.argv[5])
        try:
            step = eval(sys.argv[6])
        except:
            step = 1
        c.data_context.CIM.calculation_config = config_dict[config]
        with open(in_filename, 'rb') as (file):
            walk_results = pickle.load(file)
        results = []
        for i in range(start, end,step):
            values = walk_results[i]
            results.append(c.data_context.CIM.optimize(steps=1000, values=(values), filename=(out_filename + '.txt'),max_jumps=1))
            print("result!!")
            print("result!!")
            print("\npoint",i,results[-1])
            pickle.dump(results, open(out_filename, 'wb'))

    def run_optimalization_on_given():
        config = sys.argv[1]
        start = eval(sys.argv[2])
        out_filename = sys.argv[3]
        c.data_context.CIM.calculation_config = config_dict[config]
        c.data_context.CIM._test(steps=50,max_jumps=2, values=start, filename=(out_filename + '.txt'))
        print("result!!")


    


    print('started')
    #run_space_exploration()
    run_optimalization()
    #run_exploration_from_file()

    #last
    #run_optimalization()
    # if int(sys.argv[4])!=0:
    #     print(sys.argv[4])
    #     run_random_exploration_from_file()
    # else:
    #     run_optimalization()