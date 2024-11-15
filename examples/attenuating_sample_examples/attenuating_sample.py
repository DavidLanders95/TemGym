from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import numpy as np
from temgymbasic.utils import calculate_phi_0
from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import sys

n_rays = 10000
wavelength = 0.01
k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)

size = 64
det_shape = (size, size)
pixel_size = 1e-6
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

lens_dist = 0.5
sample_dist = 0.3
focal = 0.2
prop_dist = 1

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        semi_angle=0.001,
        radius=0.0001,
        wo=0.0001,
        amplitude=1.0,
    ),
    # comp.Lens(
    #     z=lens_dist,
    #     f=focal,
    #     z2=sample_dist,
    # ),
    comp.AttenuatingSample(
        z=lens_dist + sample_dist,
        x_width=0.1,
        y_width=0.1,
        thickness=0.01,
        attenuation_coefficient=10,
        centre_yx=(0.1/2, 0),
    ),
    comp.AccumulatingDetector(
        z=prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=1,
        interference='gauss'
    ),
)

model = Model(components, backend='gpu')

# Run Model Once
# rays = tuple(model.run_iter(num_rays=n_rays, random=False))
# image = model.detector.get_image(rays[-1])

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(np.abs(image))
# plt.show()
# Run Model Again With GUI
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
