
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import matplotlib.pyplot as plt

import numpy as np
from temgymbasic.utils import calculate_phi_0, calculate_wavelength
import ase
import abtem
from matplotlib.patches import Circle
from ase.cluster import Decahedron

def GaussianPropagator(E0, ps, lambda0, z):
    rayleigh_range = np.pi * (ps/2)**2 / lambda0
    w_z = beam_waist * np.sqrt(1 + (z / rayleigh_range)**2)  # Beam waist at z
    gaussian_amplitude = np.exp(-((x**2 + y**2) / w_z**2))
    gaussian_phase = np.exp(-1j * k * (x**2 + y**2) / (2 * R))
    H = np.exp(-1j * (2 * np.pi / lambda0) * z) * gaussian_amplitude * gaussian_phase



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
    phase_difference = 0 - np.angle(u_centre)
    u = u * np.exp(1j * phase_difference)
    return u


# Configure potential
phi = 100e3

wavelength = calculate_wavelength(phi) * 1e10
n_rays = 1

k = 2 * np.pi / wavelength

wavelengths = np.full(n_rays, wavelength)
pixel_size = 0.1

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

prop_dist = 1e-11

z_start = 5000
z_atoms = prop_dist

slice_thickness = 100
# sampling = pixel_size
# unit_cell = ase.build.bulk("Au", cubic=True)
# atoms = unit_cell * (10, 10, 10)
# potential = abtem.Potential(
#     atoms,
#     slice_thickness=slice_thickness,
#     sampling=sampling,
#     # projection="infinite",
# )

cluster = Decahedron("Cu", 9, 2, 0)
cluster.rotate("x", -30)
substrate = ase.build.bulk("C", cubic=True)

# repeat diamond structure
substrate *= (12, 12, 10)

# displace atoms with a standard deviation of 50 % of the bond length
bondlength = 1.54  # Bond length
substrate.positions[:] += np.random.randn(len(substrate), 3) * 0.5 * bondlength

# wrap the atoms displaced outside the cell back into the cell
substrate.wrap()

translated_cluster = cluster.copy()

translated_cluster.cell = substrate.cell
translated_cluster.center()
translated_cluster.translate((0, 0, -25))

atoms = substrate + translated_cluster

atoms.center(axis=2, vacuum=2)


potential = abtem.Potential(
    atoms,
    gpts=128,
    slice_thickness=slice_thickness,
)

pixel_size = potential.sampling[0]
phase_shift = np.asarray(potential.build().compute().transmission_function(phi).compute().array)
# phase_shift = np.sum(phase_shift, axis=0)
npix = phase_shift.shape[1]
det_shape = (npix, npix)

print(phase_shift.shape)
# extent = [0 - dsize / 2, 0 + dsize / 2, 0 - dsize / 2, 0 + dsize/2]

field = phase_shift

components = (
    comp.GaussBeam(
        z=z_atoms-1e-11,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wos,
        amplitude=amplitude,
        tilt_yx=tilt_yx,
        random_subset=n_rays,
        offset_yx=(0, 0)
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

gaussian_wave = abtem.Waves(
    array=gbd_input_field.copy(), energy=phi, sampling=pixel_size
)

exit_wave = gaussian_wave.multislice(potential)
exit_wave.compute()
detector_field = exit_wave.complex_images().array

# scattered_field = gbd_input_field * phase_shift
# detector_field = FresnelPropagator(scattered_field, pixel_size, wavelength, prop_dist)

components = (
    comp.GaussBeam(
        z=z_atoms-1e-11,
        voltage=calculate_phi_0(wavelength),
        radius=x0,
        wo=wos,
        amplitude=amplitude,
        tilt_yx=tilt_yx,
        random_subset=n_rays,
        offset_yx=(0, 0)
    ),
    comp.DiffractingPlanes(
        z=z_atoms,
        field=field,
        z_step=slice_thickness,
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

fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# First row
axs[0, 0].imshow(np.abs(gbd_input_field), cmap='gray')
axs[0, 0].set_title('Input Field')

axs[0, 1].imshow(np.angle(np.sum(phase_shift, axis=0)), cmap='gray')
# circle = Circle((0, 0), radius=4, edgecolor='red', facecolor='none')
# axs[0, 1].add_patch(circle)
axs[0, 1].set_title('Potential Phase')

# Second row
axs[1, 0].imshow(np.abs(detector_field), cmap='gray')
axs[1, 0].set_title('Multislice Method Amplitude')

axs[1, 1].imshow(np.abs(gbd_output_field), cmap='gray')
axs[1, 1].set_title('GBD Method Amplitude')

# Third row showing phase images
axs[2, 0].imshow(np.angle(detector_field), cmap='gray')
axs[2, 0].set_title('Multislice Method Phase')

axs[2, 1].imshow(np.angle(gbd_output_field), cmap='gray')
axs[2, 1].set_title('GBD Method Phase')

plt.tight_layout()

# # Plot cross sections
central_pixel = det_shape[0] // 2

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Normalize the fields between 0 and 1
detector_field_amplitude = np.abs(detector_field[central_pixel, :])
gbd_output_field_amplitude = np.abs(gbd_output_field[central_pixel, :])

detector_field_amplitude /= np.max(detector_field_amplitude)
gbd_output_field_amplitude /= np.max(gbd_output_field_amplitude)

# axs[0].set_xlim(-size, size)
axs[0].plot(detector_field_amplitude, label='Multislice Method')
axs[0].plot(gbd_output_field_amplitude, label='GBD Method')
axs[0].set_title('Cross Section Amplitude')
axs[0].legend()

detector_field_phase = zero_phase(detector_field[central_pixel, :], central_pixel)
gbd_output_field_phase = zero_phase(gbd_output_field[central_pixel, :], central_pixel)

axs[1].plot(np.angle(detector_field_phase), label='Multislice Method')
axs[1].plot(np.angle(gbd_output_field_phase), label='GBD Method')
axs[1].set_title('Cross Section Phase')
axs[1].legend()

plt.tight_layout()
plt.show()
