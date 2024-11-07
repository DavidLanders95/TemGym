
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import matplotlib.pyplot as plt

import numpy as np
from temgymbasic.utils import calculate_phi_0, calculate_wavelength
import ase
import abtem

def FresnelPropagator(E0, ps, lambda0, z):
    """
    Parameters:
        E0 : 2D array
            The initial complex field in the x-y source plane.
        ps : float
            Pixel size in the object plane (same units as wavelength).
        lambda0 : float
            Wavelength of the light (in the same units as ps).
        z : float
            Propagation distance (in the same units as ps).

    Returns:
        Ef : 2D array
            The complex field after propagating a distance z.
    """
    n, m = E0.shape

    fx = np.fft.fftfreq(n, ps)
    fy = np.fft.fftfreq(m, ps)
    Fx, Fy = np.meshgrid(fx, fy)
    
    H = np.exp(-1j * (2 * np.pi / lambda0) * z) * np.exp(-1j * np.pi * lambda0 * z * (Fx**2 + Fy**2))
    E0fft = np.fft.fft2(E0)
    G = H * E0fft
    Ef = np.fft.ifft2(G)
    
    return Ef

def zero_phase(u, idx_x):
    u_centre = u[idx_x]
    phase_difference =  0 - np.angle(u_centre)
    u = u * np.exp(1j * phase_difference)
    return u

# Configure potential
phi_0 = 100e3

wavelength = calculate_wavelength(phi_0) * 1e10
n_rays = 1

k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)

size = 128
det_shape = (size, size)
pixel_size = 0.5
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

wo = 2
wos = np.full(n_rays, wo)

amplitude = np.ones(n_rays)
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

prop_dist = 500

num_atoms_x = 1
num_atoms_y = 1
atom_spacing = dsize/2
x_start = dsize/2
y_start = dsize/2

z_start = 5000
z_atoms = prop_dist

# Create an empty Atoms object
atoms = ase.Atoms('Si0', cell=[x_start + num_atoms_x * atom_spacing,
                               y_start + num_atoms_y * atom_spacing,
                               z_start])

# Adding atoms in a row
for i in range(num_atoms_x):
    for j in range(num_atoms_y):
        x_position = x_start + i * atom_spacing + 4
        y_position = y_start + j * atom_spacing + 4
        atoms += ase.Atoms('Si1', positions=[(x_position, y_position, z_atoms)])

atom_positions = np.array([[4, 4]])

potential = abtem.Potential(atoms, sampling=pixel_size, projection="infinite",
                            slice_thickness=prop_dist)
phase_shift = np.asarray(potential.build().compute().transmission_function(phi_0).compute().array[1])

npix = phase_shift.shape[0]
extent = [0 - dsize / 2, 0 + dsize / 2, 0 - dsize / 2, 0 + dsize/2]

# Map atom positions to pixel indices
def world_to_pixel(atom_positions, extent, shape):
    x_world = atom_positions[:, 0]
    y_world = atom_positions[:, 1]
    nx, ny = shape[1], shape[0]

    pixel_coords_x = np.round((x_world / pixel_size) + (nx // 2)).astype(int)
    pixel_coords_y = np.round((y_world / pixel_size) + (ny // 2)).astype(int)

    return pixel_coords_y, pixel_coords_x

# Map atom positions to pixel indices
y_indices, x_indices = world_to_pixel(atom_positions, extent, phase_shift.shape)

# Initialize mask
mask = np.zeros_like(phase_shift, dtype=np.int8)

# Create masks around each atom
yy, xx = np.indices(phase_shift.shape)
r = wo * 4 # Radius of the circular shape in pixels

for yi, xi in zip(y_indices, x_indices):
    distance = (xx - xi) ** 2 + (yy - yi) ** 2
    mask[distance <= r ** 2] = 1
    indices = np.argwhere(mask == 1)
    y_indices, x_indices = indices[:, 0], indices[:, 1]

    # Convert pixel indices to world coordinates
    x_min, x_max, y_min, y_max = extent
    nx, ny = mask.shape[1], mask.shape[0]

    x_world = x_indices * pixel_size - (nx // 2) * pixel_size + pixel_size / 2
    y_world = y_indices * pixel_size - (ny // 2) * pixel_size + pixel_size / 2

    mask_coords = np.column_stack((x_world, y_world))
    mask_indices = np.column_stack((x_indices, y_indices))

plt.figure()
plt.imshow(mask, extent=extent, cmap='gray')

field = phase_shift * mask

components = (
    comp.GaussBeam(
        z=z_atoms-1e-11,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wos,
        amplitude=amplitude,
        tilt_yx=tilt_yx,
        random_subset=n_rays,
        offset_yx=(4, 4)
    ),
    comp.Detector(
        z=z_atoms,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)
model = Model(components, backend='gpu')
rays = tuple(model.run_iter(num_rays=n_rays))
gbd_input_field = model.detector.get_image(rays[-1])

scattered_field = gbd_input_field * phase_shift
detector_field = FresnelPropagator(scattered_field, pixel_size, wavelength, prop_dist)

components = (
    comp.GaussBeam(
        z=z_atoms-1e-11,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wos,
        amplitude=amplitude,
        tilt_yx=tilt_yx,
        random_subset=n_rays,
        offset_yx=(4, 4)
    ),
    comp.DiffractingPlane(
        z=z_atoms,
        field=field,
        atomic_mask=mask,
        atomic_coordinates=mask_coords,
        atomic_indices=mask_indices,
        pixel_size=pixel_size,
    ),
    comp.Detector(
        z=z_atoms+prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)
model = Model(components, backend='gpu')
rays = tuple(model.run_iter(num_rays=n_rays))
gbd_output_field = model.detector.get_image(rays[-1])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(np.abs(gbd_input_field), cmap='gray')
axs[0, 0].set_title('Input Field')

axs[0, 1].imshow(np.angle(phase_shift) * mask, extent=extent, cmap='gray')
axs[0, 1].scatter(x_world, -y_world, marker=',', color='red', alpha=0.1)
axs[0, 1].set_title('Phase Shift with Mask')

axs[1, 0].imshow(np.abs(detector_field), cmap='gray')
axs[1, 0].set_title('Multislice Method')

axs[1, 1].imshow(np.abs(gbd_output_field), cmap='gray')
axs[1, 1].set_title('GBD Method')
# Plot cross sections
central_pixel = size // 2

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Normalize the fields between 0 and 1
detector_field_magnitude = np.abs(detector_field[central_pixel, :])
gbd_output_field_magnitude = np.abs(gbd_output_field[central_pixel, :])

detector_field_magnitude /= np.max(detector_field_magnitude)
gbd_output_field_magnitude /= np.max(gbd_output_field_magnitude)

axs[0].set_xlim(-size/2, size/2)
axs[0].plot(detector_field_magnitude, label='Multislice Method')
axs[0].plot(gbd_output_field_magnitude, label='GBD Method')
axs[0].set_title('Cross Section Magnitude')
axs[0].legend()

detector_field_phase = zero_phase(detector_field[central_pixel, :], central_pixel)
gbd_output_field_phase = zero_phase(gbd_output_field[central_pixel, :], central_pixel)

axs[1].plot(np.angle(detector_field_phase), label='Multislice Method')
axs[1].plot(np.angle(gbd_output_field_phase), label='GBD Method')
axs[1].set_title('Cross Section Phase')
axs[1].legend()

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
