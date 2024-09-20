import numpy as np
from diffractio.scalar_sources_XY import Scalar_source_XY
import matplotlib.pyplot as plt

def zero_phase(u, idx_x, idx_y):
    u_centre = u[idx_x, idx_y]
    phase_difference =  0 - np.angle(u_centre)
    u = u * np.exp(1j * phase_difference)
    
    return u

def abssqr(x):
    return np.real(x*np.conj(x))

def FT(x):
    return np.fft.fftshift(np.fft.fft2(x))

def iFT(x):
    return np.fft.ifft2(np.fft.ifftshift(x))
    
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

n_rays = 1
wavelength = 0.01
k = 2 * np.pi / wavelength
wo = 0.1

size = 2048
det_shape = (size, size)
pixel_size = 0.01
dsize = det_shape[0] * pixel_size

x_det = np.linspace(-dsize / 2, dsize / 2, size)

theta_x = -5
theta_y = -3

deg_yx = np.deg2rad((theta_y, theta_x))
tilt_yx = np.tan(deg_yx)

# Calculate theta and phi
tan_theta_x = np.tan(deg_yx[1])
tan_theta_y = np.tan(deg_yx[0])

theta = np.arctan(np.sqrt(tan_theta_x**2 + tan_theta_y**2))
phi = np.arctan2(tan_theta_y, tan_theta_x)

x0 = 0.9
y0 = 0

prop_dist = 25

fresnel_input_field = Scalar_source_XY(x=x_det, y=x_det, wavelength=wavelength)
fresnel_input_field.gauss_beam(A=1, r0=(x0, 0), z0=0, w0=(wo, wo), theta=theta, phi=phi)
fresnel_output_field = FresnelPropagator(fresnel_input_field.u, pixel_size, wavelength, prop_dist)
fresnel_output_field = zero_phase(fresnel_output_field, size//2, size//2)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.abs(fresnel_output_field),cmap='gray')
ax1.axvline(size // 2, color='white', alpha=0.3)
ax1.axhline(size // 2, color='white', alpha=0.3)
ax2.imshow(np.angle(fresnel_output_field ),cmap='RdBu')
ax2.axvline(size // 2, color='k', alpha=0.3)
ax2.axhline(size // 2, color='k', alpha=0.3)
fig.suptitle("Fresnel")



fig, (ax1, ax2) = plt.subplots(1, 2)
s = np.s_[size // 2, :]
ax1.plot(np.abs(fresnel_output_field[s]), label="Fresnel")
#ax1.plot(np.abs(gbd_output_field[s]), label="GBD") - Put Gaussian Beam cross section here
ax1.legend()
ax2.plot(np.angle(fresnel_output_field[s]), label="Fresnel")
#ax2.plot(np.angle(gbd_output_field[s]), label="GBD") - Put Gaussian Beam cross section here
ax2.legend()

plt.show()