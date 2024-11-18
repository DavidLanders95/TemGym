import abc
from typing import (
    Generator, Tuple, Optional, Type,
    TYPE_CHECKING,
)
from dataclasses import dataclass, astuple


import numpy as np
from numpy.typing import NDArray  # Assuming np is an alias for numpy
from scipy.constants import c, e, m_e
from scipy.interpolate import RegularGridInterpolator

from . import (
    UsageError,
    InvalidModelError,
    PositiveFloat,
    NonNegativeFloat,
    Radians,
    Degrees,
    BackendT
)
from .aber import dopd_dx, dopd_dy, opd, aber_x_aber_y
from .gbd import (
    differential_matrix,
    calculate_Qinv,
    calculate_Qpinv,
    propagate_misaligned_gaussian,
    propagate_misaligned_gaussian_region
)
from .rays import Rays, GaussianRays
from .utils import (
    get_array_module,
    P2R, R2P,
    circular_beam,
    gauss_beam_rayset,
    point_beam,
    calculate_direction_cosines,
    calculate_wavelength,
    get_array_from_device,
    initial_r_rayset
)

if TYPE_CHECKING:
    from .gui import ComponentGUIWrapper

# Defining epsilon constant from page 18 of principles of electron optics 2017, Volume 1.
EPSILON = abs(e)/(2*m_e*c**2)


class Component(abc.ABC):
    def __init__(self, z: float, name: Optional[str] = None):
        if name is None:
            name = type(self).__name__

        self._name = name
        self._z = z

    def _validate_component(self):
        pass

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, new_z: float):
        raise UsageError("Do not set z on a component directly, use Model methods")

    def _set_z(self, new_z: float):
        self._z = new_z

    @property
    def entrance_z(self) -> float:
        return self.z

    @property
    def exit_z(self) -> float:
        return self.z

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}: {self._name} @ z = {self.z}'

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        ...
        raise NotImplementedError

    @staticmethod
    def gui_wrapper() -> Optional[Type['ComponentGUIWrapper']]:
        return None


@dataclass
class LensAberrations:
    spherical: float
    coma: float
    astigmatism: float
    field_curvature: float
    distortion: float

    def __iter__(self):
        return iter(astuple(self))

    def BFCDE(self):
        return (
            self.spherical,
            self.coma,
            self.astigmatism,
            self.field_curvature,
            self.distortion,
        )

    def nonzero(self):
        return any(s != 0 for s in self)


class Lens(Component):
    def __init__(self, z: float,
                 f: Optional[float] = None,
                 m: Optional[float] = None,
                 z1: Optional[Tuple[float]] = None,
                 z2: Optional[Tuple[float]] = None,
                 aber_coeffs: Optional[LensAberrations] = None,
                 name: Optional[str] = None):
        super().__init__(z=z, name=name)

        self.aber_coeffs = aber_coeffs
        self._z1, self._z2, self._f, self._m = self._calculate_lens_paremeters(z1, z2, f, m)

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m: float):
        self._m = m

    @property
    def z1(self) -> float:
        return self._z1

    @z1.setter
    def z1(self, z1: float):
        self._z1 = z1

    @property
    def z2(self) -> float:
        return self._z2

    @z2.setter
    def z2(self, z2: float):
        self._z2 = z2

    @property
    def ffp(self) -> float:
        return self.z - abs(self._f)

    def _calculate_lens_paremeters(self, z1, z2, f, m, xp=np):

        if (f is not None and m is not None) and (z1 is None and z2 is None):
            # m <1e-10 means that the object is very far away, the lens focuses the beam to a point.
            if np.abs(m) <= 1e-10:
                z1 = -1e10
                z2 = f
            # m >1e-10 means that the image is formed very far away, the lens collimates the beam.
            elif np.abs(m) > 1e10:
                z1 = -f
                z2 = 1e10
            else:
                z1 = f * (1/m - 1)
                z2 = f * (1 - m)
        elif (z1 is not None and z2 is not None) and (f is None and m is None):
            if np.abs(z1) < 1e-10:
                z2 = 1e10
                f = -1e10
                m = 1e10
            if np.abs(z2) < 1e-10:
                z1 = 1e10
                f = 1 / z2
                m = 0.0
            else:
                f = (1 / z2 - 1 / z1) ** -1
                m = z2 / z1
        elif (f is not None and z1 is not None) and (z2 is None and m is None):
            if np.abs(z1) < 1e-10:
                z2 = 1e10
                m = 1e10
            elif np.abs(f) < 1e-10:
                z2 = 1e10
                m = 1e10
            else:
                z2 = (1 / f + 1 / z1) ** -1
                m = z2 / z1
        else:
            raise InvalidModelError("Lens must have defined: f and m, or z1 and z2, or f and z1")

        return z1, z2, f, m

    @staticmethod
    def lens_matrix(f, xp=np):
        '''
        Lens ray transfer matrix

        Parameters
        ----------
        f : float
            Focal length of lens

        Returns
        -------
        ndarray
            Output Ray Transfer Matrix
        '''
        return xp.array(
            [[1,      0, 0,      0, 0],
             [-1 / f, 1, 0,      0, 0],
             [0,      0, 1,      0, 0],
             [0,      0, -1 / f, 1, 0],
             [0,      0, 0,      0, 1]]
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        xp = rays.xp

        # Rays at lens plane
        u1 = rays.x_central
        v1 = rays.y_central

        du1 = rays.dx_central
        dv1 = rays.dy_central

        rays.data = xp.matmul(self.lens_matrix(xp.float64(self._f), xp=xp), rays.data)
        rays.path_length -= (rays.x ** 2 + rays.y ** 2) / (2 * xp.float64(self._f))

        if self.aber_coeffs is not None and self.aber_coeffs.nonzero():

            coeffs = self.aber_coeffs

            M = self._m
            z2 = self._z2

            # Rays at object plane
            x1 = u1 + du1 * self._z1
            y1 = v1 + dv1 * self._z1

            psi_a = np.arctan2(v1, u1)
            psi_o = np.arctan2(y1, x1)
            psi = psi_a - psi_o

            # Calculate the aberration in x and y
            # (Approximate R' as the reference sphere radius at image side)
            eps_x, eps_y = aber_x_aber_y(u1, v1, x1, y1, coeffs, z2, M, xp=xp)

            x2 = u1 + rays.dx_central * z2
            y2 = v1 + rays.dy_central * z2

            x2_aber = x2 + eps_x
            y2_aber = y2 + eps_y

            nx, ny, nz = calculate_direction_cosines(x2, y2, z2, u1, v1, 0.0, xp=xp)
            nx_aber, ny_aber, nz_aber = calculate_direction_cosines(x2_aber, y2_aber, z2,
                                                                    u1, v1, 0.0, xp=xp)
            W = opd(u1, v1, x1, y1, psi, coeffs, z2, M, xp=xp)

            dx_slope = nx_aber / nz_aber - nx / nz
            dy_slope = ny_aber / nz_aber - ny / nz

            if isinstance(rays, GaussianRays):
                rays.dx += xp.repeat(dx_slope, 5)
                rays.dy += xp.repeat(dy_slope, 5)
                rays.path_length += xp.repeat(W, 5)
            else:
                rays.dx += dx_slope
                rays.dy += dy_slope
                rays.path_length += W

        # Just straightforward matrix multiplication
        yield rays.new_with(
            data=rays.data,
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import LensGUI
        return LensGUI


class PerfectLens(Lens):
    def __init__(self, z: float,
                 f: float,
                 m: Optional[float] = None,
                 z1: Optional[Tuple[float]] = None,
                 z2: Optional[Tuple[float]] = None,
                 name: Optional[str] = None):
        super().__init__(z=z, f=f, m=m, z1=z1, z2=z2, name=name)
        # self._f = f
        # self._m = m

        # self._z1, self._z2, self._m = self.initialise_m_and_image_planes(z1, z2, m, f)

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def z1(self) -> float:
        return self._z1

    @z1.setter
    def z1(self, z1: float):
        self._z1 = z1

    @property
    def z2(self) -> float:
        return self._z2

    @z2.setter
    def z2(self, z2: float):
        self._z2 = z2

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m: float):
        self._m = m

    @property
    def ffp(self) -> float:
        return self.z - abs(self._f)

    def update_m_and_image_planes(self, z1, z2, m):
        f = self._f
        self._z, self._f1, self._z2, self._m = self.initialise_m_and_image_planes(z1, z2, m, f)

    def get_exit_pupil_coords(self, rays, xp=np):

        f = self._f
        m = self._m
        z1 = self._z1
        z2 = self._z2
        NA1 = self.NA1
        NA2 = self.NA2

        # Convert slope into direction cosines
        L1 = rays.dx / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
        M1 = rays.dy / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
        N1 = xp.sqrt(1 - L1 ** 2 - M1 ** 2)

        u1 = rays.x
        v1 = rays.y

        # Ray object point coordinates (x1, y1) on front conjugate plane
        if (self.NA1 == 0.0):
            x1 = (L1 / N1) * z1
            y1 = (M1 / N1) * z1
            r1_mag = xp.sqrt((x1 - u1) ** 2 + (y1 - v1) ** 2 + z1 ** 2)

            L1_est = -(x1 - u1) / r1_mag
            M1_est = -(y1 - v1) / r1_mag
        else:
            x1 = (L1 / N1) * z1 + u1
            y1 = (M1 / N1) * z1 + v1
            r1_mag = xp.sqrt((x1 - u1) ** 2 + (y1 - v1) ** 2 + z1 ** 2)

        # Principal Ray directions
        if (self.NA1 == 0.0):
            L1_p = rays.dx / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            M1_p = rays.dy / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            N1_p = 1 / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            p1_mag = xp.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
        else:
            p1_mag = xp.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)

            # Obtain direction cosines of principal ray from second principal plane to image point
            L1_p = (x1 / p1_mag) * z1 / xp.abs(z1)
            M1_p = (y1 / p1_mag) * z1 / xp.abs(z1)
            N1_p = xp.sqrt(1 - L1_p ** 2 - M1_p ** 2)

        # Coordinates in image plane or focal plane
        if xp.abs(m) <= 1.0:
            x2 = z2 * (L1_p / N1_p)
            y2 = z2 * (M1_p / N1_p)

            p2_mag = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
            L2_p = (x2 / p2_mag) * (z2 / xp.abs(z2))
            M2_p = (y2 / p2_mag) * (z2 / xp.abs(z2))
            N2_p = xp.sqrt(1 - L2_p ** 2 - M2_p ** 2)
        else:
            a = x1 / z1
            b = y1 / z1
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p
            x2 = (L2_p / N2_p) * z2
            y2 = (M2_p / N2_p) * z2
            p2_mag = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

        # Calculation to back propagate to right hand side principal plane
        Cx = m * L2_p - L1_p
        Cy = m * M2_p - M1_p

        if (NA1 == 0.0):
            L2 = (L1_est + Cx) / m
            M2 = (M1_est + Cy) / m
            N2 = xp.sqrt(1 - L2 ** 2 - M2 ** 2)
        else:
            L2 = (L1 + Cx) / m
            M2 = (M1 + Cy) / m
            N2 = xp.sqrt(1 - L2 ** 2 - M2 ** 2)

        # We use a mask to find rays that have gone to the centre,
        # because we are not inputting one ray at a time, but a vector of rays.
        mask = xp.sqrt(u1 ** 2 + v1 ** 2) < 1e-7

        # Initialize the output arrays
        u2 = xp.zeros_like(u1)
        v2 = xp.zeros_like(v1)

        # Handle the case where the mask is true and NA2 = 0.0
        if NA2 == 0.0:
            a = -x1 / f
            b = -y1 / f
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p

            L2[mask] = L2_p[mask]
            M2[mask] = M2_p[mask]
            N2[mask] = N2_p[mask]
            u2[mask] = 0.0
            v2[mask] = 0.0

        # For the case where the mask is false, (rays are not going through the centre of the lens)
        not_mask = ~mask
        u2[not_mask] = -(L2[not_mask] / N2[not_mask]) * z2 + x2[not_mask]
        v2[not_mask] = -(M2[not_mask] / N2[not_mask]) * z2 + y2[not_mask]

        if NA2 == 0:
            a = -x1 / f
            b = -y1 / f
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p

            L2[not_mask] = L2_p[not_mask]
            M2[not_mask] = M2_p[not_mask]
            N2[not_mask] = N2_p[not_mask]

        # Calculate final distance from image/focal plane to point
        # ray leaves lens for optical path length
        r2_mag = xp.sqrt((x2 - u2) ** 2 + (y2 - v2) ** 2 + z2 ** 2)

        opl1 = r1_mag + r2_mag  # Ray opl
        opl0 = p1_mag + p2_mag  # Principal ray opl

        dopl = (opl0 - opl1)

        return x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        # x1 - object plane x coordinate of ray
        # y1 - object plane y coordinate of ray
        # u2 - exit pupil x coordinate of ray
        # v2 - exit pupil y coordinate of ray
        # x2 - image plane x coordinate of ray
        # y2 - image plane y coordinate of ray
        # L2, M2, N2 - direction cosines of the ray at the exit pupil
        # R - reference sphere radius
        # dopl - optical path length change

        xp = rays.xp
        x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl = self.get_exit_pupil_coords(rays, xp=xp)

        rays.x = u2
        rays.y = v2
        rays.dx = L2 / N2
        rays.dy = M2 / N2
        rays.path_length += dopl

        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import PerfectLensGUI
        return PerfectLensGUI


class AberratedLens(PerfectLens):
    def __init__(self, z: float,
                 f: float,
                 m: Optional[float] = None,
                 z1: Optional[Tuple[float]] = None,
                 z2: Optional[Tuple[float]] = None,
                 name: Optional[str] = None,
                 coeffs: Tuple = [0, 0, 0, 0, 0]):  # 5 aberration coefficients (C, K, A, D, F)

        super().__init__(z=z, f=f, m=m, z1=z1, z2=z2, name=name)
        self.coeffs = coeffs

        # self._f = f
        # self._m = m

        # # Initial Numerical Aperture
        # self.NA1 = 0.1
        # self.NA2 = 0.1

        # self._z1, self._z2, self._m = self.initialise_m_and_image_planes(z1, z2, m, f)

    def step(self, rays: Rays) -> Generator[Rays, None, None]:
        # # Call the step function of the parent class
        # yield from super().step(rays)

        M = self._m

        xp = rays.xp

        z2 = self._z2
        x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl = self.get_exit_pupil_coords(rays, xp=xp)

        coeffs = self.coeffs

        # Reference sphere radius
        R = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

        # Coordinates of reference sphere
        u2_circle = x2 - L2 * R
        v2_circle = y2 - M2 * R
        z2_circle = z2 - N2 * R

        psi_a = np.arctan2(v2_circle, u2_circle)
        psi_o = np.arctan2(y1, x1)
        psi = psi_a - psi_o

        # Calculate the aberration in x and y (Approximate)
        eps_x = -dopd_dx(u1, v1, x1, y1, psi, coeffs, z2, M, xp=xp) * z2
        eps_y = -dopd_dy(u1, v1, x1, y1, psi, coeffs, z2, M, xp=xp) * z2

        W = opd(u1, v1, x1, y1, psi, coeffs, z2, M, xp=xp)

        # Get aberration direction cosines - remember the aberrated rays must
        # go through the same point on the reference sphere
        # as the perfect rays
        nx, ny, nz = calculate_direction_cosines(x2 + eps_x, y2 + eps_y, z2,
                                                 u1, v1, z2_circle, xp=xp)

        # Calculate the new aberrated ray coordinates in the image plane
        # x2_aber = x2 + eps_x
        # y2_aber = y2 + eps_y

        # Calculate the new aberrated ray coordinates in the exit pupil plane
        u2_aber = u1  # x2_aber - nx / nz * (z2)
        v2_aber = v1  # y2_aber - ny / nz * (z2)

        # u2_aber = -nx / nz * (phi_zn) + phi_xn
        # v2_aber = -ny / nz * (phi_zn) + phi_yn

        rays.path_length += W

        rays.x = u2_aber
        rays.y = v2_aber
        rays.dx = nx / nz
        rays.dy = ny / nz

        # Return the modified rays
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import AberratedLensGUI
        return AberratedLensGUI


class Sample(Component):
    def __init__(self, z: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        rays.location = self
        yield rays

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI


class STEMSample(Sample):
    def __init__(
        self,
        z: float,
        overfocus: NonNegativeFloat = 0.,
        semiconv_angle: PositiveFloat = 0.01,
        scan_shape: Tuple[int, int] = (8, 8),
        scan_step_yx: Tuple[float, float] = (0.01, 0.01),
        scan_rotation: Degrees = 0.,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx
        self.scan_rotation = scan_rotation  # converts to radians in setter

    @property
    def scan_rotation(self) -> Degrees:
        return np.rad2deg(self.scan_rotation_rad)

    @scan_rotation.setter
    def scan_rotation(self, val: Degrees):
        self.scan_rotation_rad: Radians = np.deg2rad(val)

    @property
    def scan_rotation_rad(self) -> Radians:
        return self._scan_rotation

    @scan_rotation_rad.setter
    def scan_rotation_rad(self, val: Radians):
        self._scan_rotation = val

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step_yx
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y
        if self.scan_rotation_rad != 0.:
            pos_r, pos_a = R2P(scan_position_x + scan_position_y * 1j)
            pos_c = P2R(pos_r, pos_a + self.scan_rotation_rad)
            scan_position_y, scan_position_x = pos_c.imag, pos_c.real
        return (scan_position_y, scan_position_x)

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.scan_shape,
            # This needs to be refactored...
            pixel_size=self.scan_step_yx[0],
            rotation=self.scan_rotation,
            as_int=as_int,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import STEMSampleGUI
        return STEMSampleGUI


class Source(Component):
    def __init__(
        self,
        z: float,
        tilt_yx: Tuple[float, float] = (0., 0.),
        centre_yx: Tuple[float, float] = (0., 0.),
        voltage: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx
        self.centre_yx = centre_yx
        self.phi_0 = voltage
        self.random = False

    @property
    def voltage(self):
        return self.phi_0

    @abc.abstractmethod
    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        raise NotImplementedError

    def set_centre(self, centre_yx: tuple[float, float]):
        self.centre_yx = centre_yx

    def _rays_args(self, r: NDArray, backend: BackendT = 'cpu'):
        # Add beam tilt (if any)
        if self.tilt_yx[1] != 0:
            r[1, :] += self.tilt_yx[1]
        if self.tilt_yx[0] != 0:
            r[3, :] += self.tilt_yx[0]

        # Add beam shift (if any)
        if self.centre_yx[1] != 0:
            r[0, :] += self.centre_yx[1]
        if self.centre_yx[0] != 0:
            r[2, :] += self.centre_yx[0]

        wavelength = None
        if self.phi_0 is not None:
            wavelength = calculate_wavelength(self.phi_0)

        r = get_array_module(backend).asarray(r)

        return dict(
            data=r,
            location=self,
            wavelength=wavelength,
        )

    def _make_rays(self, r: NDArray, backend: BackendT = 'cpu') -> Rays:
        return Rays.new(
            **self._rays_args(r, backend=backend),
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Source has no effect after get_rays was called
        yield rays.new_with(
            location=self
        )


class ParallelBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float,
        voltage: Optional[float] = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        centre_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name, voltage=voltage, centre_yx=centre_yx)
        self.radius = radius

    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        r = circular_beam(num_rays, self.radius,
                          random=random if random is not None else self.random)
        return self._make_rays(r, backend=backend)

    @staticmethod
    def gui_wrapper():
        from .gui import ParallelBeamGUI
        return ParallelBeamGUI


class XAxialBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
        r = np.zeros((5, num_rays))
        r[0, :] = np.random.uniform(
            -self.radius, self.radius, size=num_rays
        )
        return self._make_rays(r)


class RadialSpikesBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
        xvals = np.linspace(
            0., self.radius, num=num_rays // 4, endpoint=True
        )
        yvals = np.zeros_like(xvals)
        origin_c = xvals + yvals * 1j

        orad, oang = R2P(origin_c)
        radius1 = P2R(orad * 0.75, oang + np.pi * 0.4)
        radius2 = P2R(orad * 0.5, oang + np.pi * 0.8)
        radius3 = P2R(orad * 0.25, oang + np.pi * 1.2)
        r_c = np.concatenate((origin_c, radius1, radius2, radius3))

        r = np.zeros((5, r_c.size))
        r[0, :] = r_c.real
        r[2, :] = r_c.imag
        return self._make_rays(r)


class PointBeam(Source):
    def __init__(
        self,
        z: float,
        voltage: Optional[float] = None,
        semi_angle: Optional[float] = 0.,
        tilt_yx: Tuple[float, float] = (0., 0.),
        centre_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z, voltage=voltage, centre_yx=centre_yx)
        self.semi_angle = semi_angle
        self.tilt_yx = tilt_yx
        self.centre_yx = centre_yx

    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        r = point_beam(num_rays, self.semi_angle,
                       random=random if random is not None else self.random)
        return self._make_rays(r, backend=backend)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class XPointBeam(PointBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
        r = np.zeros((5, num_rays))
        r[1, :] = np.linspace(
            -self.semi_angle, self.semi_angle, num=num_rays, endpoint=True
            )
        return self._make_rays(r, backend=backend)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class GaussBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float,
        wo: NDArray,
        amplitude: NDArray,
        path_length: float = 0.,
        voltage: Optional[float] = None,
        offset_yx: Tuple[float, float] = (0., 0.),
        tilt_yx: Tuple[float, float] = (0., 0.),
        semi_angle: Optional[float] = 0.,
        name: Optional[str] = None,
        random_subset: Optional[int] = 1,
    ):
        super().__init__(name=name, z=z, voltage=voltage)
        self.wo = wo
        self.amplitude = amplitude
        self.path_length = path_length
        self.radius = radius
        self.tilt_yx = tilt_yx
        self.semi_angle = semi_angle
        self.random_subset = random_subset
        self.offset_yx = offset_yx

    def get_rays(
        self,
        num_rays: int,
        random: Optional[bool] = None,
        backend='cpu',
    ) -> Rays:

        xp = get_array_module(backend)

        wavelength = calculate_wavelength(self.voltage, xp=xp)

        # Ensure that these variables are on the correct device
        wo = xp.array(self.wo)

        # if random:
        #     raise NotImplementedError
        # else:

        r = gauss_beam_rayset(
            num_rays,
            self.radius,
            self.semi_angle,
            wo,
            wavelength,
            offset_yx=self.offset_yx,
            random=random if random is not None else self.random,
            random_subset=self.random_subset,
            xp=xp,
        )

        return self._make_rays(r, backend=backend)

    def _make_rays(self, r: NDArray, backend: BackendT = 'cpu') -> Rays:
        return GaussianRays.new(
            **self._rays_args(r, backend=backend),
            wo=self.wo,
            path_length=self.path_length,
            amplitude=self.amplitude
        )

    @staticmethod
    def gui_wrapper():
        from .gui import GaussBeamGUI
        return GaussBeamGUI


class Detector(Component):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        rotation: Degrees = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
        interference: Optional[str] = 'ray'
    ):
        """
        The intention of rotation is to rotate the detector
        realative to the common y/x coordinate system of the optics.
        A positive rotation would rotate the detector clockwise
        looking down a ray , and the image will appear
        to rotate anti-clockwise.

        In STEMModel the scan grid is aligned with the optics
        y/x coordinate system default, but can also
        be rotated using the "scan_rotation" parameter.

        The detector flip_y acts only at the image generation step,
        the scan grid itself can be flipped by setting negative
        scan step values
        """
        super().__init__(name=name, z=z)
        self.pixel_size = pixel_size
        self.shape = shape
        self.rotation = rotation  # converts to radians in setter
        self.flip_y = flip_y
        self.center = center
        self.interference = interference
        self.buffer = None

    @property
    def rotation(self) -> Degrees:
        return np.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = np.deg2rad(val)

    @property
    def rotation_rad(self) -> Radians:
        return self._rotation

    @rotation_rad.setter
    def rotation_rad(self, val: Radians):
        self._rotation = val

    def set_center_px(self, center_px: Tuple[int, int]):
        """
        For the desired image center in pixels (after any flip / rotation)
        set the image center in the physical coordinates of the microscope

        The continuous coordinate can be set directly on detector.center
        """
        iy, ix = center_px
        sy, sx = self.shape
        cont_y = (iy - sy // 2) * self.pixel_size
        cont_x = (ix - sx // 2) * self.pixel_size
        if self.flip_y:
            cont_y = -1 * cont_y
        mag, angle = R2P(cont_x + 1j * cont_y)
        coord: complex = P2R(mag, angle + self.rotation_rad)
        self.center = coord.imag, coord.real

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Detector has no effect on rays
        yield rays.new_with(
            location=self
        )

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def get_det_coords_for_gauss_rays(self, xEnd, yEnd, xp=np):
        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = xp.linspace(-det_size_y / 2, det_size_y / 2, self.shape[0], dtype=xEnd.dtype)
        y_det = xp.linspace(-det_size_x / 2, det_size_x / 2, self.shape[1], dtype=yEnd.dtype)
        x, y = xp.meshgrid(x_det, y_det)

        r = xp.stack((x, y), axis=-1).reshape(-1, 2)
        endpoints = xp.stack((xEnd, yEnd), axis=-1)
        # r = xp.broadcast_to(r, [n_rays, *r.shape])
        # r = xp.swapaxes(r, 0, 1)
        # has form (n_px, n_gauss, 2:[x, y])]
        # this entire section can be optimised !!!
        return r[:, xp.newaxis, :] - endpoints[xp.newaxis, ...]


    def image_dtype(self, xp=np):
        if self.interference is None:
            return xp.int32
        # Setting this next line reduces the bitdepth
        # of the image computation which can improve the speed
        # quite substantially
        return xp.complex128

    def get_image(
        self,
        rays: Rays,
        out: Optional[NDArray] = None,
    ) -> NDArray:

        xp = rays.xp

        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = self.on_grid(rays, as_int=True)
        sy, sx = self.shape
        mask = xp.logical_and(
            xp.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            xp.logical_and(
                0 <= pixel_coords_x,
                pixel_coords_x < sx
            )
        )

        image_dtype = self.image_dtype(xp=xp)
        if self.interference == 'ray':
            # If we are doing interference, we add a complex number representing
            # the phase of the ray for now to each pixel.
            # Amplitude is 1.0 for now for each complex ray.
            wavefronts = rays.amplitude * xp.exp(-1j * (2 * xp.pi / rays.wavelength) * rays.path_length)

            if isinstance(rays, GaussianRays):
                valid_wavefronts = wavefronts[0::5][mask]
            else:
                valid_wavefronts = wavefronts[mask]
            
            # image_dtype = valid_wavefronts.dtype
        elif self.interference == 'gauss':
            pass
            # image_dtype = xp.complex128
            # Setting this next line reduces the bitdepth
            # of the image computation which can improve the speed
            # quite substantially
            # image_dtype = xp.complex64
        elif self.interference is None:
            # If we are not doing interference, we simply add 1 to each pixel that a ray hits
            valid_wavefronts = 1
            # image_dtype = xp.int32

        if out is None:
            out = xp.zeros(
                self.shape,
                dtype=image_dtype,
            )
        else:
            assert out.dtype == image_dtype
            assert out.shape == self.shape

        if self.interference == 'gauss':

            self.get_gauss_image(rays, out)
        else:
            flat_icds = xp.ravel_multi_index(
                    [
                        pixel_coords_y[mask],
                        pixel_coords_x[mask],
                    ],
                    out.shape
                )

            # Increment at each pixel for each ray that hits

            if xp == np:
                np.add.at(
                    out.ravel(),
                    flat_icds,
                    valid_wavefronts,
                )
            else:
                if xp.iscomplexobj(out):
                    # Separate the real and imaginary parts
                    real_out = out.real
                    imag_out = out.imag

                    # Perform the addition separately for real and imaginary parts
                    xp.add.at(real_out.reshape(-1), flat_icds, valid_wavefronts.real)
                    xp.add.at(imag_out.reshape(-1), flat_icds, valid_wavefronts.imag)

                    # Combine the real and imaginary parts back into the out array
                    out = real_out + 1j * imag_out
                else:
                    # Perform the addition directly for non-complex arrays
                    xp.add.at(out.reshape(-1), flat_icds, valid_wavefronts)

        # Convert always to array on cpu device.
        return get_array_from_device(out)

    def get_gauss_image(
        self,
        rays: GaussianRays,
        out: NDArray,
    ) -> NDArray:

        float_dtype = out.real.dtype.type
        xp = rays.xp

        wo = rays.wo
        wavelength = rays.wavelength
        amplitude = rays.amplitude

        div = rays.wavelength / (xp.pi * wo)
        k = float_dtype(2 * xp.pi / wavelength)
        z_r = xp.pi * wo ** 2 / wavelength

        dPx = wo
        dPy = wo
        dHx = div
        dHy = div

        # rays layout
        # [5, n_rays] where n_rays = 5 * n_gauss
        # so rays.reshape(5, 5, -1)
        #  => [(x, dx, y, dy, 1), (*gauss_beams), n_gauss]

        n_gauss = rays.num // 5

        # end_rays = rays.data[0:4, :].T
        path_length = rays.path_length[0::5].astype(float_dtype)
        amplitude = rays.amplitude[0::5].astype(float_dtype)
        # split_end_rays = xp.split(end_rays, n_gauss, axis=0)
        # rayset1 = xp.stack(split_end_rays, axis=-1)

        rayset1 = xp.moveaxis(
            rays.data[0:4, :].reshape(4, n_gauss, 5),
            -1,
            0,
        )
        rayset1 = rayset1.astype(float_dtype)

        # rayset1 layout
        # [5g, (x, dx, y, dy), n_gauss]

        # rayset1 = cp.array(rayset1)

        A, B, C, D = differential_matrix(rayset1, dPx, dPy, dHx, dHy, xp=xp)
        # A, B, C, D all have shape (n_gauss, 2, 2)
        Qinv = calculate_Qinv(z_r, n_gauss, xp=xp)
        # matmul, addition and mat inverse inside
        # on operands with form (n_gauss, 2, 2)
        # matmul broadcasts in the last two indices
        # inv can be broadcast with xp.linalg.inv last 2 idcs
        # if all inputs are stacked in the first dim
        Qpinv = calculate_Qpinv(A, B, C, D, Qinv, xp=xp)
        wnew = xp.sqrt(wavelength / (xp.pi * xp.abs(Qpinv[:, 0, 0].imag)))
        # det_coords = cp.array(det_coords)
        # p2m = cp.array(p2m)
        # path_length = cp.array(path_length)
        # k = cp.array(k)

        phi_x2m = rays.data[1, 0::5]  # slope that central ray arrives at
        phi_y2m = rays.data[3, 0::5]  # slope that central ray arrives at
        p2m = xp.array([phi_x2m, phi_y2m]).T.astype(float_dtype)

        xEnd, yEnd = rayset1[0, 0], rayset1[0, 2]
        # central beam final x , y coords
        det_coords = self.get_det_coords_for_gauss_rays(xEnd, yEnd, xp=xp)
        propagate_misaligned_gaussian(
            Qinv, Qpinv, det_coords,
            p2m, k, A, B, amplitude, path_length, out.ravel(), xp=xp
        )

    @staticmethod
    def gui_wrapper():
        from .gui import DetectorGUI
        return DetectorGUI


class AccumulatingDetector(Detector):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        buffer_length: int,
        rotation: Degrees = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
        interference: Optional[str] = 'gauss'
    ):
        super().__init__(
            z=z,
            pixel_size=pixel_size,
            shape=shape,
            rotation=rotation,
            flip_y=flip_y,
            center=center,
            name=name,
            interference=interference,
        )
        self.buffer = None
        self.buffer_length = buffer_length

    @property
    def buffer_frame_shape(self) -> Optional[Tuple[int, int]]:
        if self.buffer is None:
            return
        return self.buffer.shape[1:]

    def delete_buffer(self):
        self.buffer = None

    def reset_buffer(self, rays: Rays):
        xp = rays.xp
        image_dtype = self.image_dtype(xp=xp)
        self.buffer = xp.zeros(
            (self.buffer_length, *self.shape),
            dtype=image_dtype,
        )
        # the next index to write into
        self.buffer_idx = 0

    def get_image(self, rays: Rays) -> NDArray:
        if self.buffer is None or self.buffer_frame_shape != self.shape:
            self.reset_buffer(rays)
        else:
            self.buffer[self.buffer_idx] = 0.

        super().get_image(
            rays,
            out=self.buffer[self.buffer_idx],
        )

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_length
        # Convert always to array on cpu device.
        return get_array_from_device(self.buffer.sum(axis=0))

    @staticmethod
    def gui_wrapper():
        from .gui import AccumulatingDetectorGUI
        return AccumulatingDetectorGUI


class Deflector(Component):
    '''Creates a single deflector component and handles calls to GUI creation, updates to GUI
        and stores the component matrix. See Double Deflector component for a more useful version
    '''
    def __init__(
        self,
        z: float,
        defx: float = 0.,
        defy: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        defy : float, optional
            deflection kick in slope units to the incoming ray y angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.defy = defy

    @staticmethod
    def deflector_matrix(def_x, def_y, xp=np):
        '''Single deflector ray transfer matrix

        Parameters
        ----------
        def_x : float
            deflection in x in slope units
        def_y : _type_
            deflection in y in slope units

        Returns
        -------
        ndarray
            Output ray transfer matrix
        '''

        return xp.array(
            [[1, 0, 0, 0,     0],
             [0, 1, 0, 0, def_x],
             [0, 0, 1, 0,     0],
             [0, 0, 0, 1, def_y],
             [0, 0, 0, 0,     1]],
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        xp = rays.xp
        yield rays.new_with(
            data=xp.matmul(
                self.deflector_matrix(xp.float64(self.defx), xp.float64(self.defy), xp=xp),
                rays.data,
            ),
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import DeflectorGUI
        return DeflectorGUI


class DoubleDeflector(Component):
    def __init__(
        self,
        first: Deflector,
        second: Deflector,
        name: Optional[str] = None,
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self._first = first
        self._second = second
        self._validate_component()

    @classmethod
    def from_params(
        cls,
        z: float,
        distance: float = 0.1,
        name: Optional[str] = None
    ):
        return cls(
            Deflector(
                z - distance / 2.
            ),
            Deflector(
                z + distance / 2.
            ),
            name=name,
        )

    def _validate_component(self):
        if self.first.z >= self.second.z:
            raise InvalidModelError("First deflector must be before second")

    @property
    def length(self) -> float:
        return self._second.z - self._first.z

    @property
    def first(self) -> Deflector:
        return self._first

    @property
    def second(self) -> Deflector:
        return self._second

    @property
    def z(self):
        self._z = (self.first.z + self.second.z) / 2
        return self._z

    def _set_z(self, new_z: float):
        dz = new_z - self.z
        self.first._set_z(self.first.z + dz)
        self.second._set_z(self.second.z + dz)

    @property
    def entrance_z(self) -> float:
        return self.first.z

    @property
    def exit_z(self) -> float:
        return self.second.z

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        for rays in self.first.step(rays):
            yield rays.new_with(
                location=(self, self.first)
            )
        rays = rays.propagate_to(self.second.entrance_z)
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays.new_with(
                location=(self, self.second)
            )

    @staticmethod
    def _send_ray_through_pts_1d(
        in_zp: Tuple[float, float],
        z_out: float,
        pt1_zp: Tuple[float, float],
        pt2_zp: Tuple[float, float],
        in_slope: float = 0.
    ) -> Tuple[float, float]:
        """
        Choose first/second deflector values such that a ray arriving
        at (in_zp) with slope (in_slope), will leave at (z_out, ...) and
        pass through (pt1_zp) then (pt2_zp)
        """

        in_zp = np.asarray(in_zp)
        pt1_zp = np.asarray(pt1_zp)
        pt2_zp = np.asarray(pt2_zp)
        dp = pt1_zp - pt2_zp
        out_zp = np.asarray(
            (
                z_out,
                pt2_zp[1] + dp[1] * (z_out - pt2_zp[0]) / dp[0],
            )
        )
        dd = out_zp - in_zp
        first_def = dd[1] / dd[0]
        first_def += in_slope
        out_slope = dp[1] / dp[0]
        second_def = out_slope - first_def
        return first_def, second_def

    def send_ray_through_points(
        self,
        in_ray: Tuple[float, float],
        pt1: Tuple[float, float, float],
        pt2: Tuple[float, float, float],
        in_slope: Tuple[float, float] = (0., 0.)
    ):
        """
        in_ray is (y, x), z is implicitly the z of the first deflector
        pt1 and pt2 are (z, y, x) after the second deflector
        in_slope is (dy, dx) at the incident point
        """
        self.first.defy, self.second.defy = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[0]),
            self.second.z,
            pt1[:2],
            pt2[:2],
            in_slope=in_slope[0],
        )
        self.first.defx, self.second.defx = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[1]),
            self.second.z,
            (pt1[0], pt1[2]),
            (pt2[0], pt2[2]),
            in_slope=in_slope[1],
        )

    @staticmethod
    def gui_wrapper():
        from .gui import SimpleDoubleDeflectorGUI
        return SimpleDoubleDeflectorGUI


class Biprism(Component):
    def __init__(
        self,
        z: float,
        offset: float = 0.,
        rotation: Degrees = 0.,
        deflection: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        offset: float
            Offset distance of biprism line
        rotation: float
            Rotation of biprism in z-plane
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.deflection = deflection
        self.offset = offset
        self.rotation = rotation

    @property
    def rotation(self) -> Degrees:
        return np.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = np.deg2rad(val)

    @property
    def rotation_rad(self) -> Radians:
        return self._rotation

    @rotation_rad.setter
    def rotation_rad(self, val: Radians):
        self._rotation = val

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:

        xp = rays.xp
        deflection = xp.array(self.deflection)
        offset = xp.array(self.offset)
        rot = xp.array(self.rotation_rad)
        pos_x = rays.x
        pos_y = rays.y
        rays_v = xp.array([pos_x, pos_y]).T

        biprism_loc_v = xp.array([offset*xp.cos(rot), offset*xp.sin(rot)])

        biprism_v = xp.array([-xp.sin(rot), xp.cos(rot)])
        biprism_v /= xp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = xp.dot(rays_v_centred, biprism_v) / xp.dot(biprism_v, biprism_v)
        projection = xp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection/xp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = xp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        rays.dx += xdeflection_mag * deflection
        rays.dy += ydeflection_mag * deflection

        rays.path_length += (
            xdeflection_mag * deflection * rays.x
            + ydeflection_mag * deflection * rays.y
        )
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import BiprismGUI
        return BiprismGUI


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius: float,
        x: float = 0.,
        y: float = 0.,
        name: Optional[str] = None,
    ):
        '''
        An Aperture that lets through rays within a radius centered on (x, y)

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default 'Aperture'
        radius : float, optional
           The radius of the aperture
        x : int, optional
            X position of the centre of the aperture, by default 0
        y : int, optional
            Y position of the centre of the aperture, by default 0
        '''

        super().__init__(z, name)

        self.x = x
        self.y = y
        self.radius = radius

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x, pos_y = rays.x, rays.y
        xp = rays.xp
        distance = xp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        yield rays.with_mask(
            distance < self.radius,
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import ApertureGUI
        return ApertureGUI


class DiffractingPlanes(Component):
    def __init__(
        self,
        z: float,  # z position of the first diffracting plane
        z_step: float,  # z step between diffracting planes
        field: np.complex128,  # Field amplitude of the diffracting plane
        pixel_size: float,
        name: Optional[str] = None,
    ):
        super().__init__(z=z, name=name)
        self.field = field
        self.z_step = z_step
        self.pixel_size = pixel_size
        self.shape = self.field.shape[1:]

    def get_gauss_image(
        self,
        rays: GaussianRays,
        out: NDArray,
    ) -> NDArray:

        float_dtype = out.real.dtype.type
        xp = rays.xp

        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = xp.linspace(-det_size_y / 2, det_size_y / 2, self.shape[0], dtype=xp.float64)
        y_det = xp.linspace(-det_size_x / 2, det_size_x / 2, self.shape[1], dtype=xp.float64)
        x, y = xp.meshgrid(x_det, y_det)

        self.grid_coords = xp.stack((x, y), axis=-1).reshape(-1, 2)

        wo = rays.wo
        wavelength = rays.wavelength
        amplitude = rays.amplitude

        div = rays.wavelength / (xp.pi * wo)
        k = 2 * xp.pi / wavelength
        z_r = xp.pi * wo ** 2 / wavelength

        dPx = wo
        dPy = wo
        dHx = div
        dHy = div

        n_gauss = rays.num // 5

        path_length = rays.path_length[0::5].astype(float_dtype)

        rayset1 = xp.moveaxis(
            rays.data[0:4, :].reshape(4, n_gauss, 5),
            -1,
            0,
        )
        rayset1 = rayset1.astype(float_dtype)

        A, B, C, D = differential_matrix(rayset1, dPx, dPy, dHx, dHy, xp=xp)
        Qinv = calculate_Qinv(z_r, n_gauss, xp=xp)
        Qpinv = calculate_Qpinv(A, B, C, D, Qinv, xp=xp)

        phi_x2m = rays.data[1, 0::5]  # slope that central ray arrives at
        phi_y2m = rays.data[3, 0::5]  # slope that central ray arrives at
        p2m = xp.array([phi_x2m, phi_y2m]).T.astype(float_dtype)

        xEnd, yEnd = rayset1[0, 0], rayset1[0, 2]
        endpoints = xp.stack((xEnd, yEnd), axis=-1)

        # Compute coordinates of detector points within mask radius
        det_coords = self.grid_coords[:, xp.newaxis, :] - endpoints[xp.newaxis, ...]   # pixel coordinates centred on end point of ray

        # Call propagate_misaligned_gaussian and get contributions
        propagate_misaligned_gaussian(
                Qinv, Qpinv, det_coords,
                p2m, k, A, B, amplitude, path_length, out.ravel(), xp=xp
        )

        return out, self.grid_coords[:, xp.newaxis, :]

    def step(
        self, rays: GaussianRays,
    ) -> Generator[Rays, None, None]:

        xp = rays.xp

        z_step = xp.array(self.z_step)
        field = xp.array(self.field)

        # Calculate atomic coordinates and indices of mask for this potential slice
        shape = self.shape
        pixel_size = self.pixel_size

        out = xp.zeros(
            shape,
            dtype=xp.complex128,
        )
        num_slices = field.shape[0]

        for i in range(0, num_slices):

            gauss_field, det_coords = self.get_gauss_image(rays, out)

            scattered_field = gauss_field * field[i]

            rays_x = det_coords[:, :, 0]
            rays_y = det_coords[:, :, 1]
            rays_dx = xp.zeros(len(rays_x))
            rays_dy = xp.zeros(len(rays_y))

            wo = xp.full(len(rays_x), pixel_size / 2)
            wavelength = rays.wavelength
            k = 2 * xp.pi / wavelength

            div = wavelength / (xp.pi * wo)
            dPx = wo
            dPy = wo
            dHx = div
            dHy = div

            # this multiplies n_rays by 5
            scattered_r = initial_r_rayset(len(rays_x), xp=xp)

            # Central coords
            scattered_r[0] = xp.repeat(rays_x, 5)
            scattered_r[2] = xp.repeat(rays_y, 5)
            scattered_r[1] = xp.repeat(rays_dx, 5)
            scattered_r[3] = xp.repeat(rays_dy, 5)

            # Offset in x
            scattered_r[0, 1::5] += dPx
            # Offset in y
            scattered_r[2, 2::5] += dPy
            # Slope in x from origin
            scattered_r[1, 3::5] += dHx
            # Slope in y from origin
            scattered_r[3, 4::5] += dHy

            scattered_r_amplitude = xp.abs(scattered_field.ravel())
            scattered_r_opl = xp.angle(scattered_field.ravel()) / k
            scattered_r_opl = xp.repeat(scattered_r_opl, 5)

            rays_data = scattered_r
            rays_opl = scattered_r_opl
            rays_amplitude = scattered_r_amplitude
            rays_wo = wo

            rays = GaussianRays.new(
                data=rays_data,
                amplitude=rays_amplitude,
                path_length=rays_opl,
                location=self,
                wavelength=rays.wavelength,
                wo=rays_wo,
            )

            if i < num_slices - 1:
                rays = rays.propagate(self.z_step)

        yield rays

    def get_det_coords_for_gauss_rays(self, xEnd, yEnd, xp=np):
        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = xp.linspace(-det_size_y / 2, det_size_y / 2, self.shape[0], dtype=xEnd.dtype)
        y_det = xp.linspace(-det_size_x / 2, det_size_x / 2, self.shape[1], dtype=yEnd.dtype)
        x, y = xp.meshgrid(x_det, y_det)

        r = xp.stack((x, y), axis=-1).reshape(-1, 2)
        endpoints = xp.stack((xEnd, yEnd), axis=-1)

        return r[:, xp.newaxis, :] - endpoints[xp.newaxis, ...]


class PotentialSample(Sample):
    def __init__(
        self,
        z: float,
        potential,
        Ex,
        Ey,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)

        # We're renaming here some terms to be closer to the math in Hawkes
        # Not sure if this is recommended or breaks any convetions
        self.phi = potential
        self.dphi_dx = Ex
        self.dphi_dy = Ey

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        xp = rays.xp
        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)  # Equation 3.16
        phi_0_plus_phi = (rays.phi_0 + self.phi((rays.x, rays.y)))  # Part of Equation 2.18

        phi_hat = (phi_0_plus_phi) * (1 + EPSILON * (phi_0_plus_phi))  # Equation 2.18

        # Between Equation 2.22 & 2.23
        dphi_hat_dx = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dx((rays.x, rays.y))
        dphi_hat_dy = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dy((rays.x, rays.y))

        # Perform deflection to ray in slope coordinates
        rays.dx += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dx  # Equation 3.22
        rays.dy += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dy  # Equation 3.22

        # Note here we are ignoring the Ez component (dphi/dz) of 3.22,
        # since we have modelled the potential of the atom in a plane
        # only, we won't have an Ez component (At least I don't think this is the case?
        # I could be completely wrong here though - it might actually have an effect.
        # But I'm not sure I can get an Ez from an infinitely thin slice.

        # Equation 5.16 & 5.17 & 3.16, where ds of 5.16 is replaced by ds/dz * dz,
        # where ds/dz = rho (See 3.16 and a little below it)
        rays.path_length += rho * xp.sqrt(phi_hat / rays.phi_0)

        # Currently the modifications are all inplace so we only need
        # to change the location, this should be made clearer
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI


class AttenuatingSample(Sample):
    def __init__(
        self,
        z: float,
        x_width: float,
        y_width: float,
        centre_yx: Tuple[float, float],
        thickness: float,
        attenuation_coefficient: float,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)

        # We're renaming here some terms to be closer to the math in Hawkes
        # Not sure if this is recommended or breaks any convetions
        self.x_width = x_width
        self.y_width = y_width
        self.centre_yx = centre_yx
        self.thickness = thickness
        self.mu = attenuation_coefficient

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        xp = rays.xp

        centre_yx = self.centre_yx
        x_width = self.x_width
        y_width = self.y_width
        thickness = self.thickness

        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)  # Equation 3.16

        dx = rays.dx / rho
        dy = rays.dy / rho
        dz = xp.ones(rays.num) / rho

        # Define the box boundaries
        x_min = centre_yx[1] - x_width / 2
        x_max = centre_yx[1] + x_width / 2
        y_min = centre_yx[0] - y_width / 2
        y_max = centre_yx[0] + y_width / 2
        z_min = self.z
        z_max = z_min + thickness

        x0 = rays.x
        y0 = rays.y
        z0 = xp.ones(rays.num) * self.z

        # Line starting point (outside the box) and direction
        p0 = xp.array([x0, y0, z0])    # Starting point
        d = xp.array([dx, dy, dz])     # Direction vector (should be normalized)

        # Normalize the direction vector
        d = d / xp.linalg.norm(d, axis=0)

        # Compute inverse of direction components
        inv_dx = 1 / d[0]
        inv_dy = 1 / d[1]
        inv_dz = 1 / d[2]

        # Compute t1 and t2 for x-axis
        t1x = (x_min - p0[0]) * inv_dx
        t2x = (x_max - p0[0]) * inv_dx
        tmin_x = xp.minimum(t1x, t2x)
        tmax_x = xp.maximum(t1x, t2x)

        # Compute t1 and t2 for y-axis
        t1y = (y_min - p0[1]) * inv_dy
        t2y = (y_max - p0[1]) * inv_dy
        tmin_y = xp.minimum(t1y, t2y)
        tmax_y = xp.maximum(t1y, t2y)

        # Compute t1 and t2 for z-axis
        t1z = (z_min - p0[2]) * inv_dz
        t2z = (z_max - p0[2]) * inv_dz
        tmin_z = xp.minimum(t1z, t2z)
        tmax_z = xp.maximum(t1z, t2z)

        # Compute overall tmin and tmax
        tmin = xp.maximum(tmin_x, xp.maximum(tmin_y, tmin_z))
        tmax = xp.minimum(tmax_x, xp.minimum(tmax_y, tmax_z))

        # Check for intersection
        attenuation_mask = (tmax >= 0) & (tmin <= tmax)

        # Calculate the path length inside the box
        path_length = xp.zeros_like(tmin)
        path_length[attenuation_mask] = tmax[attenuation_mask] - tmin[attenuation_mask]

        # Apply attenuation
        rays.amplitude[attenuation_mask] *= xp.exp(-self.mu * path_length[attenuation_mask])

        rays = rays.propagate(thickness)

        yield rays.new_with(
            location=self.z + thickness,
        )


        # Calculate the path length inside the box for intersecting rays
        path_length = xp.zeros_like(tmin)
        path_length[attenuation_mask] = tmax[attenuation_mask] - tmin[attenuation_mask]

        # Apply attenuation based on the calculated path lengths
        rays.amplitude[attenuation_mask] *= xp.exp(-self.mu * path_length[attenuation_mask])

        rays = rays.propagate(thickness)

        yield rays.new_with(
            location=self.z + thickness,
        )

        # Determine if the ray intersects the box
        attenuation_mask = (tmax >= tmin) & (tmax > 0)

        # Calculate the path length inside the box for intersecting rays
        path_length = xp.zeros_like(tmin)
        path_length[attenuation_mask] = tmax[attenuation_mask] - tmin[attenuation_mask]

        # Apply attenuation based on the calculated path lengths
        rays.amplitude[attenuation_mask] *= xp.exp(-self.mu * path_length[attenuation_mask])

        rays = rays.propagate(thickness)

        yield rays.new_with(
            location=self.z + thickness,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import AttenuatingSampleGUI
        return AttenuatingSampleGUI


class ProjectorLensSystem(Component):
    def __init__(
        self,
        first: Lens,
        second: Lens,
        magnification: float = -1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self.magnification = magnification

        self._first = first
        self._second = second

        self._validate_component()

        self.adjust_z2_and_z3_from_magnification(self.magnification)

    @classmethod
    def from_params(
        cls,
        z: float,
        z1: float,
        z2: float,
        z3: float,
        z4: float,
        distance: float = 0.1,
        name: Optional[str] = None
    ):
        return cls(
            Lens(
                z=z - distance / 2.,
                z1=z1,
                z2=z2,
            ),
            Lens(
                z=z + distance / 2.,
                z3=z3,
                z4=z4,
            ),
            name=name,
        )

    def _validate_component(self):
        if self.first.z >= self.second.z:
            raise InvalidModelError("First Projector Lens must be before second")

    @property
    def distance(self) -> float:
        return self._second.z - self._first.z

    @property
    def first(self) -> Lens:
        return self._first

    @property
    def second(self) -> Lens:
        return self._second

    @property
    def z(self):
        self._z = (self.first.z + self.second.z) / 2
        return self._z

    def _set_z(self, new_z: float):
        dz = new_z - self.z
        self.first._set_z(self.first.z + dz)
        self.second._set_z(self.second.z + dz)

    @property
    def entrance_z(self) -> float:
        return self.first.z

    @property
    def exit_z(self) -> float:
        return self.second.z

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        for rays in self.first.step(rays):
            yield rays.new_with(
                location=(self, self.first)
            )
        rays = rays.propagate_to(self.second.entrance_z)
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays.new_with(
                location=(self, self.second)
            )

    def adjust_z2_and_z3_from_magnification(self, magnification):

        z1 = self.first.z1
        dz = self.distance
        z4 = self.second.z2
        z2 = (magnification * z1 * dz) / (magnification * z1 + z4)
        z3 = z2-dz

        self.first.z2 = z2
        self.second.z1 = z3
        self.first.f = 1/(1/z2 - 1/z1)
        self.second.f = 1/(1/z4 - 1/z3)

    @staticmethod
    def gui_wrapper():
        from .gui import ProjectorLensSystemGUI
        return ProjectorLensSystemGUI
