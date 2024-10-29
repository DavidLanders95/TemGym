
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import matplotlib.pyplot as plt

import numpy as np
from temgymbasic.utils import calculate_phi_0

n_rays = 1
wavelength = 0.01
k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)

size = 512
det_shape = (size, size)
pixel_size = 0.005
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

wo = 0.4
wos = np.full(n_rays, wo)

div = wavelength / (np.pi * wo)

dPx = wo
dPy = wo
dHx = div
dHy = div

z_r = (np.pi * wo ** 2) / wavelengths

theta_x = 0
theta_y = 0

deg_yx = np.deg2rad((theta_y, theta_x))
tilt_yx = np.tan(deg_yx)

x0 = 0
y0 = 0

prop_dist = 25

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wo,
        tilt_yx=tilt_yx
    ),
    comp.Detector(
        z=prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)
model = Model(components)

z_values = np.linspace(0.0001, prop_dist, 1000)
X, Z = np.meshgrid(x_det, z_values)
evaluated_field = np.zeros((len(x_det), len(z_values)), dtype=np.complex128)

for idx_z, z in enumerate(z_values):
    rays = model.run_to_z(z=z, num_rays=n_rays)
    gbd_output_field = model.detector.get_image(rays)
    evaluated_field[:, idx_z] = gbd_output_field[:, det_shape[0] // 2]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.abs(evaluated_field), cmap='gray', extent=[z_values[0], z_values[-1],
                                                         x_det[0], x_det[-1]], aspect='auto')
ax2.imshow(np.angle(evaluated_field), cmap='RdBu', extent=[z_values[0], z_values[-1],
                                                           x_det[0], x_det[-1]], aspect='auto')
plt.show()
