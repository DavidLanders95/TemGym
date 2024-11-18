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

size = 128
det_shape = (size, size)
pixel_size = 10e-6
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

sample_dist = 0.5
condenser_dist = 0.25
objective_dist = sample_dist + 1e-2
total_dist = 1

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=200e3,
        semi_angle=0.001,
        radius=10e-6,
        wo=3e-6,
        amplitude=1.0,
    ),
    comp.Lens(
        z=condenser_dist,
        z1=-condenser_dist,
        z2=sample_dist - condenser_dist,
        name='Condenser Lens',
    ),
    comp.AttenuatingSample(
        z=sample_dist,
        x_width=0.1,
        y_width=0.1,
        thickness=50e-9,
        attenuation_coefficient=1e7,
        centre_yx=(0.1/2, 0),
    ),
    comp.Lens(
        z=objective_dist,
        z1=sample_dist - objective_dist,
        z2=total_dist - objective_dist,
        name='Objective Lens',
    ),
    comp.AccumulatingDetector(
        z=total_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=1,
        interference='gauss'
    ),
)

model = Model(components, backend='gpu')
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
