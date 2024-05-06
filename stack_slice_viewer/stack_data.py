from PIL import Image
import numpy as np

from stack_io import *
import context

class StackData():
    def __init__(self):
        self.real_distance=None
        self.context=context.Context()
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
        self.context.data_context.pi_stack, _ = self._load_concentration_from_file(pi_directory)

        self.concentration_loaded = True
        self.is_skeleton_loaded = 0