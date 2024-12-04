from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import numpy as np
from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import sys

n_rays = 1
wavelength = 0.01
k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)

size = 100
det_shape = (size, size)
pixel_size = 1e-6
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

condenser_dist = 0.25
sample_dist = 0.5
objective_dist = sample_dist + 0.05
projector_dist = 0.6
total_dist = 1

scale = 1e-4
x_width = 1 * scale
y_width = 1 * scale

grid_size = 100
half_width_x = x_width / 2
half_width_y = y_width / 2
thickness = 50e-7

x = np.linspace(-half_width_x, half_width_x, grid_size)
y = np.linspace(-half_width_y, half_width_y, grid_size)
z = np.linspace(0, thickness, 10)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Parameters for the lattice of gratings
line_width = 0.01 * scale  # Thickness of each grating line
grating_period = 0.05 * scale  # Spacing between gratings

# Create the lattice pattern
grating_X = (np.mod(X + grating_period / 2, grating_period) < line_width)
grating_Y = (np.mod(Y + grating_period / 2, grating_period) < line_width)
attenuation = np.where(grating_X | grating_Y, 1e11, 0)

rotation = np.array([0, 0, 0])
components = (
    comp.GaussBeam(
        z=0.0,
        voltage=200e3,
        # semi_angle=0.001,
        radius=10 * scale,
        wo=1e-10,
        amplitude=1.0,
    ),
    # comp.Lens(
    #     z=condenser_dist,
    #     z1=-condenser_dist,
    #     z2=sample_dist - condenser_dist,
    #     name='Condenser Lens',
    # ),
    comp.AttenuatingSample(
        z=sample_dist,
        x_width=x_width,
        y_width=y_width,
        thickness=thickness,
        rotation=rotation,
        attenuation=attenuation,
        centre_yx=[0, 0],
    ),
    # comp.Lens(
    #     z=objective_dist,
    #     z1=sample_dist - objective_dist,
    #     z2=total_dist - objective_dist,
    #     name='Objective Lens',
    # ),
    # comp.ProjectorLensSystem(
    #     first=comp.Lens(z=projector_dist, name='PL1', z1=-1, z2=1),
    #     second=comp.Lens(z=projector_dist + 1e-3, name='PL2', z1=-1, z2=1),
    #     name='Projector Lens System',
    # ),
    comp.AccumulatingDetector(
        z=total_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=1,
        interference='ray'
    )
)

sample_detector_model = comp.Detector(z=1.0,
                                      pixel_size=pixel_size,
                                      shape=det_shape,
                                      interference='ray')

model = Model(components, backend='gpu')
# all_rays = tuple(model.run_iter(num_rays=n_rays, random=False))
# rays_before_sample = model.run_to_z(z=sample_dist, num_rays=n_rays)
# rays_after_sample = model.run_to_z(z=sample_dist+thickness*2, num_rays=n_rays)

# print(np.average(rays_before_sample.amplitude))
# print(np.average(rays_after_sample.amplitude))

# all_rays = tuple(model.run_iter(num_rays=n_rays, random=False))
# rays_at_start = all_rays[0]
# rays_at_end = all_rays[-1]


# image_before_sample = sample_detector_model.get_image(rays_before_sample)
# image_after_sample = sample_detector_model.get_image(rays_after_sample)
# image_at_end = model.detector.get_image(rays_at_end)

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# axs[0].imshow(np.abs(image_before_sample), extent=sample_detector_model.extent)
# # axs[0].plot(rays_before_sample.x_central, rays_before_sample.y_central, 'r.', markersize=1)
# axs[0].set_xlabel('x (m)')
# axs[0].set_title('Image at Sample')

# axs[1].imshow(np.abs(image_after_sample), extent=sample_detector_model.extent)
# # axs[1].plot(rays_after_sample.x_central, rays_after_sample.y_central, 'r.', markersize=1)
# axs[1].set_xlabel('x (m)')
# axs[1].set_title('Image After Sample')

# axs[2].imshow(np.abs(image_at_end), extent=model.detector.extent)
# axs[2].set_title('Image at End')

# plt.show()

AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
