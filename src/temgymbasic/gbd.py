import numpy as np
import numexpr as ne
import line_profiler
xp = np


@line_profiler.profile
def differential_matrix(rayset, dPx, dPy, dHx, dHy):

    x_cen_T = rayset[0, 0, :]
    x_px_T = rayset[1, 0, :]
    x_py_T = rayset[2, 0, :]
    x_pl_T = rayset[3, 0, :]
    x_pm_T = rayset[4, 0, :]

    l_cen_T = rayset[0, 1, :]
    l_px_T = rayset[1, 1, :]
    l_py_T = rayset[2, 1, :]
    l_pl_T = rayset[3, 1, :]
    l_pm_T = rayset[4, 1, :]

    y_cen_T = rayset[0, 2, :]
    y_px_T = rayset[1, 2, :]
    y_py_T = rayset[2, 2, :]
    y_pl_T = rayset[3, 2, :]
    y_pm_T = rayset[4, 2, :]

    m_cen_T = rayset[0, 3, :]
    m_px_T = rayset[1, 3, :]
    m_py_T = rayset[2, 3, :]
    m_pl_T = rayset[3, 3, :]
    m_pm_T = rayset[4, 3, :]

    # Computing each element of the matrix
    ABCD = xp.array([
        [(x_px_T - x_cen_T) / dPx, (x_py_T - x_cen_T) / dPy,
         (x_pl_T - x_cen_T) / dHx, (x_pm_T - x_cen_T) / dHy],
        [(y_px_T - y_cen_T) / dPx, (y_py_T - y_cen_T) / dPy,
         (y_pl_T - y_cen_T) / dHx, (y_pm_T - y_cen_T) / dHy],
        [(l_px_T - l_cen_T) / dPx, (l_py_T - l_cen_T) / dPy,
         (l_pl_T - l_cen_T) / dHx, (l_pm_T - l_cen_T) / dHy],
        [(m_px_T - m_cen_T) / dPx, (m_py_T - m_cen_T) / dPy,
         (m_pl_T - m_cen_T) / dHx, (m_pm_T - m_cen_T) / dHy]
    ])

    ABCD = ABCD.transpose(2, 0, 1)
    A = ABCD[:, 0:2, 0:2]
    B = ABCD[:, 0:2, 2:4]
    C = ABCD[:, 2:4, 0:2]
    D = ABCD[:, 2:4, 2:4]

    return A, B, C, D


def propagate_qpinv_abcd(Qinv, A, B, C, D):
    num = C + D @ Qinv
    den = A + B @ Qinv
    return num @ xp.linalg.inv(den)


def Matmulvec(r0, Mat, r1):
    out = (r0[..., 0]*Mat[..., 0, 0] + r0[..., 1]*Mat[..., 1, 0])*r1[..., 0]
    out = (out + (r0[..., 0]*Mat[..., 0, 1] + r0[..., 1]*Mat[..., 1, 1])*r1[..., 1])
    return out


def misalign_phase(B, A, r1m, r2, k):
    """
    Parameters
    ----------
    B,A : numpy.ndarrays
        elements of the ray transfer matrix
    r1m : numpy.ndarray of size 2
        misalignment in position in x then y
    r2 : numpy.ndarray of dimension 2
        detector coordinates in x and y. First dimension holds x/y, second holds coordinate
    k : float, complex
        wave number of simulation
    """
    Binv = xp.linalg.inv(B)
    BinvA = Binv @ A

    misalign = (r1m[..., 0]*BinvA[..., 0, 0] + r1m[..., 1]*BinvA[..., 1, 0])*r1m[..., 0]
    misalign = (misalign + (r1m[..., 0]*BinvA[..., 0, 1]
                            + r1m[..., 1]*BinvA[..., 1, 1])*r1m[..., 1])

    cross = (r1m[..., 0]*Binv[..., 0, 0] + r1m[..., 1]*Binv[..., 1, 0])*r2[..., 0]
    cross = -2*(cross + (r1m[..., 0]*Binv[..., 0, 1] + r1m[..., 1]*Binv[..., 1, 1])*r2[..., 1])

    return xp.exp(-1j * k / 2 * (misalign + cross))


@line_profiler.profile
def transversal_phase(Qpinv, r, k):
    """compute the transverse gaussian phase of a gaussian beam

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix
    r : numpy.ndarray
        N x 2 radial coordinate vector

    Returns
    -------
    numpy.ndarray
        phase of the gaussian profile
    """
    # r: (n_px, n_gauss, 2:[x ,y])
    # Qpinv: (n_gauss, 2, 2)
    # transversal_ref = (
    #     (
    #         r[..., 0]
    #         * Qpinv[..., 0, 0]
    #     )
    #     + (
    #         r[..., 1]
    #         * Qpinv[..., 1, 0]
    #     )
    # ) * r[..., 0]
    # transversal_ref += (
    #     (
    #         r[..., 0]
    #         * Qpinv[..., 0, 1]
    #     )
    #     + (
    #         r[..., 1]
    #         * Qpinv[..., 1, 1]
    #     )
    # ) * r[..., 1]
    # transversal_ref /= 2

    # The intermediate array here is quite large
    # of shape (n_px, n_gauss, 2, 2), can be improved
    # using some form of one-step sumproduct
    # transversal = (
    #     r[..., np.newaxis]
    #     * Qpinv[np.newaxis, ...] / 2
    # ).sum(axis=-1)
    # transversal *= r
    # transversal = transversal.sum(axis=-1)

    # transversal = (r * Qpinv[np.newaxis, ..., 0] / 2)
    # transversal += (r * Qpinv[np.newaxis, ..., 1] / 2)
    # transversal *= r
    # transversal = transversal.sum(axis=-1)

    transversal = (r[..., 0] ** 2 * Qpinv[np.newaxis, ..., 0, 0] / 2)
    transversal += (r[..., 1] ** 2 * Qpinv[np.newaxis, ..., 1, 1] / 2)
    return transversal
    # return xp.exp(1j * k * transversal)


def phase_correction(r1m, p1m, r2m, p2m, k):
    # See https://www.tandfonline.com/doi/abs/10.1080/09500340600842237
    z1_phase = xp.sum(r1m * p1m, axis=1)
    z2_phase = xp.sum(r2m * p2m, axis=1)
    return xp.exp(-1j * k / 2 * (-z2_phase + z1_phase))


@line_profiler.profile
def gaussian_amplitude(Qinv, A, B):
    den = A + B @ Qinv
    return 1 / xp.sqrt(xp.linalg.det(den))


@line_profiler.profile
def guoy_phase(Qpinv):
    """compute the guoy phase of a complex curvature matrix

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix

    Returns
    -------
    numpy.ndarray
        guoy phase of the complex curvature matrix
    """

    e1, e2 = eigenvalues_2x2(Qpinv)
    guoy = (xp.arctan(xp.real(e1) / xp.imag(e1)) + xp.arctan(xp.real(e2) / xp.imag(e2))) / 2
    # return xp.exp(-1j * guoy)
    return guoy


@line_profiler.profile
def misalign_phase_plane_wave(r2, p2m, k):
    # r2: (n_px, n_gauss, 2:[x ,y])
    # p2m: (n_gauss, 2:[x ,y])
    # l0 = r2 * p2m
    phi = r2[:, :, 0] * p2m[:, 0] * (1 + ((p2m[:, 0] ** 2) / 2))
    # phi_y = l0_y * (1 + ((p2m[:, 1] ** 2) / 2))
    # phi = phi_x + phi_y
    phi += r2[:, :, 1] * p2m[:, 1] * (1 + ((p2m[:, 1] ** 2) / 2))
    return phi
    return (
        r2
        * p2m[np.newaxis, ...]
        * (
            1
            + (p2m[np.newaxis, ...] ** 2) / 2
        )
    ).sum(axis=-1)
    # return xp.exp(1j * k * phi)


@line_profiler.profile
def propagate_misaligned_gaussian(
    Qinv,
    Qpinv,
    r,
    p2m,
    k,
    A,
    B,
    path_length,
    out,
):
    # Qinv : (n_gauss, 2, 2), complex
    # Qpinv : (n_gauss, 2, 2), complex
    # r: (n_px, n_gauss, 2:[x ,y]), float => det coords relative to final, central pos
    # p2m: (n_gauss, 2:[x, y]), float => slopes of arriving central ray
    # k: scalar float
    # A: (n_gauss, 2, 2), float
    # B: (n_gauss, 2, 2), float
    # path_length: (n_gauss,), float => path length of central ray

    misaligned_phase = misalign_phase_plane_wave(r, p2m, k)
    # (n_px, n_gauss): complex
    aligned = transversal_phase(Qpinv, r, k)  # Phase and Amplitude at transversal plane to beam dir
    # (n_px, n_gauss): complex
    # opl = xp.exp(1j * k * path_length)  # Optical path length phase
    # (n_gauss,): complex
    guoy = guoy_phase(Qpinv)  # Guoy phase
    # (n_gauss,): complex
    amplitude = gaussian_amplitude(Qinv, A, B)  # Complex Gaussian amplitude
    # (n_gauss,): complex
    aligned *= 1j
    aligned += 1j * misaligned_phase
    aligned += 1j * path_length[np.newaxis, :]
    aligned *= k
    aligned -= 1j * guoy[np.newaxis, :]
    xp.exp(aligned, out=aligned)
    # xp.exp(aligned, out=aligned)
    aligned *= xp.abs(amplitude)
    # It should be possible to avoid this intermediate .sum
    # if we could reduce directly into out, but I can't find
    # a way to express that with numpy. Numba could be an option
    out += aligned.sum(axis=-1)
    # return aligned.sum(axis=-1)
    # (n_px,): complex
    # return field


def eigenvalues_2x2(array):
    """ Computes the eigenvalues of a 2x2 matrix using a trick

    Parameters
    ----------
    array : numpy.ndarray
        a N x 2 x 2 array that we are computing the eigenvalues of
    Returns
    -------
    e1, e2 : floats of shape N
        The eigenvalues of the array
    """

    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    determinant = a * d - b * c
    mean_ondiag = (a + d) / 2
    e1 = mean_ondiag + xp.sqrt(mean_ondiag ** 2 - determinant)
    e2 = mean_ondiag - xp.sqrt(mean_ondiag ** 2 - determinant)

    return e1, e2


def mat_inv_2x2(array):
    """compute inverse of 2x2 matrix, broadcasted

    Parameters
    ----------
    array : numpy.ndarray
        array containing 2x2 matrices in last dimension. Returns inverse array of shape array.shape

    Returns
    -------
    matinv
        matrix inverse array
    """
    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    det = a * d - b * c

    matinv = xp.array([[d, -b], [-c, a]]) / det
    if matinv.ndim > 2:
        for i in range(matinv.ndim - 2):
            matinv = xp.moveaxis(matinv, -1, 0)

    return matinv


def calculate_Qinv(z_r, num_rays):

    qinv = 1/(-1j*z_r)
    Qinv = xp.zeros((num_rays, 2, 2), dtype=xp.complex128)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


def calculate_Qpinv(A, B, C, D, Qinv):

    NUM = (C + D @ Qinv)

    DEN = mat_inv_2x2(A + B @ Qinv)

    return NUM @ DEN


def center_transversal_plane(r_pixels, r_ray, orthogonal_matrix):

    """Centers the coordinate system on the transversal plane

    Returns
    -------
    r
        coordinates of distances from the center of the transversal plane to the pixels
    """
    r_ray = xp.moveaxis(r_ray, 1, 0)

    # pre-treat r pixel
    r_pixels = xp.broadcast_to(r_pixels, (orthogonal_matrix.shape[0], *r_pixels.shape))
    r_pixels = xp.moveaxis(r_pixels, -1, 0)
    r_pixels = r_pixels[..., xp.newaxis]

    r_pixels = orthogonal_matrix @ r_pixels
    r_origin = r_ray[:, 0]

    r = r_pixels-r_origin
    r = r[..., 0]

    return r


def distance_to_transversal(r_pixel, r_ray, k_ray):
    n = k_ray[0]

    RHS = n @ r_pixel
    RHS = xp.broadcast_to(RHS, (r_ray.shape[0], RHS.shape[0], RHS.shape[1]))

    LHS = xp.sum(n*r_ray, axis=-1)
    DEN = xp.sum(n*k_ray, axis=-1)

    LHS = xp.broadcast_to(LHS, (RHS.shape[-1], LHS.shape[0], LHS.shape[1]))
    LHS = xp.moveaxis(LHS, 0, -1)

    DEN = xp.broadcast_to(DEN, (LHS.shape[-1], DEN.shape[0], DEN.shape[1]))
    DEN = xp.moveaxis(DEN, 0, -1)

    Delta = (RHS-LHS)/DEN
    Delta = Delta[..., xp.newaxis]

    return Delta


def propagate_rays_and_transform(r_ray, k_ray, Delta, orthogonal_matrix):
    """propagate rays in free space

    Parameters
    ----------
    r_rays : ndarray
        position vectors
    k_rays : ndarray
        direction cosine vectors
    Delta : _type_
        distances to propagate along k_rays

    Returns
    -------
    r_rays,k_rays
        broadcasted r and k rays after propagating
    """

    # swap Delta to match rays
    Delta = xp.moveaxis(Delta, -2, 0)

    r_ray = r_ray + k_ray*Delta
    r_ray = xp.moveaxis(r_ray, 1, 0)  # get the ray back in the first index

    r_ray = r_ray[..., xp.newaxis]
    k_ray = k_ray[..., xp.newaxis]

    r_ray = orthogonal_matrix @ r_ray
    k_ray = orthogonal_matrix @ k_ray

    # swap axes so we can broadcast to the right shape
    r_ray = xp.swapaxes(r_ray, 0, 1)

    # broadcast k_ray
    k_ray = xp.broadcast_to(k_ray, r_ray.shape)

    # put the axes back pls
    r_ray = xp.swapaxes(r_ray, 0, 1)
    k_ray = xp.swapaxes(k_ray, 0, 1)

    return r_ray, k_ray


def orthogonal_transformation_matrix(n, normal):
    """generates the orthogonal transformation to the transversal plane

    Parameters
    ----------
    n : N x 3 vector, typically k_ray[0]
        vector normal to the transversal plane, typically the central ray of a beamlet
    normal : N x 3 vector, typically (0,0,1)
        local surface normal of the detector plane

    Returns
    -------
    orthogonal_matrix : Nx3x3 ndarray
        orthogonal transformation matrix
    """
    l_dir = xp.cross(n, -normal)
    aligned_mask = xp.all(xp.isclose(l_dir, 0), axis=-1)

    # Initialize l with an orthogonal vector for the aligned case
    l_dir[aligned_mask] = xp.array([1, 0, 0])

    # If n is [1, 0, 0], use [0, 1, 0] for l in the aligned case
    special_case_mask = aligned_mask & xp.all(xp.isclose(n, [1, 0, 0]), axis=-1)
    l_dir[special_case_mask] = xp.array([0, 1, 0])

    # Normalize l for non-aligned cases
    non_aligned_mask = ~aligned_mask
    l_dir[non_aligned_mask] /= vector_norm(l_dir[non_aligned_mask])[..., xp.newaxis]

    m = xp.cross(n, l_dir)

    orthogonal_matrix = xp.asarray([[l_dir[..., 0], l_dir[..., 1], l_dir[..., 2]],
                    [m[..., 0], m[..., 1], m[..., 2]],
                    [n[..., 0], n[..., 1], n[..., 2]]])

    orthogonal_matrix = xp.moveaxis(orthogonal_matrix, -1, 0)

    return orthogonal_matrix


def vector_norm(vector):
    """computes the magnitude of a vector

    Parameters
    ----------
    vector : numpy.ndarray
        N x 3 array containing a 3-vector

    Returns
    -------
    numpy.ndarray
        magnitude of the vector
    """
    vx = vector[..., 0] * vector[..., 0]
    vy = vector[..., 1] * vector[..., 1]
    vz = vector[..., 2] * vector[..., 2]

    return xp.sqrt(vx + vy + vz)


def optical_path_and_delta(OPD, Delta, k):
    """compute the total optical path experienced by a beamlet

    Parameters
    ----------
    OPD : numpy.ndarray
        optical path difference from raytracing code
    Delta : numpy.ndarray
        optical path propagation from evaluation plane to transversal plane

    Returns
    -------
    numpy.ndarray
        the total optical path experienced by a ray
    """
    OPD = OPD[0]  # central ray
    Delta = xp.moveaxis(Delta[0, ..., 0], -1, 0)  # central ray
    opticalpath = OPD + Delta  # grab central ray of OPD

    return xp.exp(1j * k * opticalpath)


def convert_slope_to_direction_cosines(dx, dy):
    l_dir = dx / xp.sqrt(1 + dx ** 2 + dy ** 2)
    m_dir = dy / xp.sqrt(1 + dx ** 2 + dy ** 2)
    n_dir = 1 / xp.sqrt(1 + dx ** 2 + dy ** 2)
    return l_dir, m_dir, n_dir


def differential_matrix_calculation(central_u, central_v, diff_uu, diff_uv,
                                    diff_vu, diff_vv, du, dv):
    """computes a sub-matrix of the ray transfer tensor

    diff_ij means a differential ray with initial differential in dimension i, evaluated in j
    diff_xy means the differential ray with an initial dX in the x dimension on the source plane,
    and these are the coordinates of that ray in the y-axis on the detector plane

    Parameters
    ----------
    central_u : numpy.ndarray
        array describing the central rays position or angle in x or l
    central_v : numpy.ndarray
        array describing the central rays position or angle in y or m
    diff_uu : numpy.ndarray
        array describing the differential rays position or angle in x or l
    diff_uv : numpy.ndarray
        array describing the differential rays position or angle in y or m
    diff_vu : numpy.ndarray
        array describing the differential rays position or angle in y or m
    diff_vv : numpy.ndarray
        array describing the differential rays position or angle in y or m
    du : float
        differential on sourc plane in position or angle in x or l
    dv : float
        differential on sourc plane in position or angle in y or m

    Returns
    -------
    numpy.ndarray
        sub-matrix of the ray transfer tensor
    """
    Mxx = (diff_uu - central_u) / du  # Axx
    Myx = (diff_uv - central_v) / du  # Ayx
    Mxy = (diff_vu - central_u) / dv  # Axy
    Myy = (diff_vv - central_v) / dv  # Ayy

    diffmat = xp.moveaxis(xp.asarray([[Mxx, Mxy], [Myx, Myy]]), -1, 0)
    diffmat = xp.moveaxis(diffmat, -1, 0)

    return diffmat


def det_2x2(array):
    """compute determinant of 2x2 matrix, broadcasted"""
    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    det = a * d - b * c

    return det