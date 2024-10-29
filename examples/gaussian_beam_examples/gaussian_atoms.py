
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import matplotlib.pyplot as plt

import numpy as np
from temgymbasic.utils import calculate_phi_0, calculate_wavelength
import ase
import abtem

# Configure potential
phi_0 = 100e3

wavelength = calculate_wavelength(phi_0) * 1e10
n_rays = 1
print(wavelength)

k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)

size = 1024
det_shape = (size, size)
pixel_size = 0.2
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

wo = 5.0
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

prop_dist = 1000

num_atoms_x = 1
num_atoms_y = 1
atom_spacing = dsize/2
x_start = dsize/2
y_start = dsize/2

z_start = 2000
z_atoms = prop_dist

# Create an empty Atoms object
atoms = ase.Atoms('Si0', cell=[x_start + num_atoms_x * atom_spacing,
                               y_start + num_atoms_y * atom_spacing,
                               z_start])

# Adding atoms in a row
for i in range(num_atoms_x):
    for j in range(num_atoms_y):
        x_position = x_start + i * atom_spacing
        y_position = y_start + j * atom_spacing
        atoms += ase.Atoms('Si1', positions=[(x_position, y_position, z_atoms)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
abtem.show_atoms(atoms, ax=ax1, title="Beam view", numbering=True, merge=False)
abtem.show_atoms(atoms, ax=ax2, plane="xz", title="Side view", legend=True)

potential = abtem.Potential(atoms, sampling=pixel_size, projection="infinite", slice_thickness=prop_dist)
phase_shift = np.asarray(potential.build().compute().transmission_function(phi_0).compute().array[1])
npix = phase_shift.shape[0]
extent = potential.build().extent

plt.figure()
plt.imshow(np.angle(phase_shift), extent=[0, extent[0], 0, extent[1]], origin='lower')

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wo,
        tilt_yx=tilt_yx,
    ),
    comp.Detector(
        z=z_start,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)
model = Model(components, backend='gpu')

steps = 1000
z_values = np.linspace(0.000001, prop_dist*2, steps)
X, Z = np.meshgrid(x_det, z_values)
evaluated_field = np.zeros((len(x_det), len(z_values)), dtype=np.complex128)

for idx_z, z in enumerate(z_values):
    rays = model.run_to_z(z=z, num_rays=n_rays)
    gbd_output_field = model.detector.get_image(rays)
    evaluated_field[:, idx_z] = gbd_output_field[:, det_shape[0] // 2]

final_field = gbd_output_field

wo = pixel_size / 2

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wo,
        tilt_yx=tilt_yx,
    ),
    comp.Detector(
        z=prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)
model = Model(components, backend='gpu')


for idx_z, z in enumerate(z_values[0:steps//2]):
    rays = model.run_to_z(z=z, num_rays=n_rays)
    gbd_output_field = model.detector.get_image(rays)
    evaluated_field[:, idx_z + steps//2] += gbd_output_field[:, det_shape[0] // 2]

final_field += gbd_output_field

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.abs(evaluated_field), cmap='gray', extent=[z_values[0], z_values[-1],
                                                         x_det[0], x_det[-1]], aspect='auto')
ax2.imshow(np.angle(evaluated_field), cmap='RdBu', extent=[z_values[0], z_values[-1],
                                                           x_det[0], x_det[-1]], aspect='auto')
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.log(np.abs(final_field)), cmap='gray', aspect='auto')
ax2.imshow(np.angle(final_field), aspect='auto')
plt.show()

# max_phase_shift_pixel = np.unravel_index(np.argmax(np.angle(phase_shift)), phase_shift.shape)
# print(f"The pixel with the largest phase shift is at: {max_phase_shift_pixel}")

# pixel_center_x = extent[0] * (max_phase_shift_pixel[1] + 0.5) / npix
# pixel_center_y = extent[1] * (max_phase_shift_pixel[0] + 0.5) / npix

# r_atoms = np.zeros(
#     (5, 1 * 5),
#     dtype=np.float64,
# )  # x, theta_x, y, theta_y, 1
