# """
# Demonstrates GLVolumeItem for displaying volumetric data with rotation applied.
# """

# import numpy as np
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import RegularGridInterpolator as RGI
# import matplotlib.pyplot as plt

# app = pg.mkQApp("GLVolumeItem Example")
# w = gl.GLViewWidget()
# w.show()
# w.setWindowTitle('pyqtgraph example: GLVolumeItem')
# w.setCameraPosition(distance=200)

# overall_cube_shape = (300, 300, 100)
# cube_edges = gl.GLLinePlotItem()

# voxel_shape = (100, 100, 10)
# Nx, Ny, Nz = overall_cube_shape

# x_width = 1
# y_width = 1
# thickness = 0.1
# # Define the vertices of the cube
# vertices = np.array([
#     [-x_width / 2, -y_width / 2, -thickness / 2],
#     [x_width / 2, -y_width / 2, -thickness / 2],
#     [x_width / 2, y_width / 2, -thickness / 2],
#     [-x_width / 2, y_width / 2, -thickness / 2],
#     [-x_width / 2, -y_width / 2, thickness / 2],
#     [x_width / 2, -y_width / 2, thickness / 2],
#     [x_width / 2, y_width / 2, thickness / 2],
#     [-x_width / 2, y_width / 2, thickness / 2],
# ])

# # Define the edges by connecting vertices
# edges = [
#     (0, 1), (1, 2), (2, 3), (3, 0),
#     (4, 5), (5, 6), (6, 7), (7, 4),
#     (0, 4), (1, 5), (2, 6), (3, 7)
# ]

# # Create line segments
# lines = []
# for edge in edges:
#     lines.extend([vertices[edge[0]], vertices[edge[1]]])

# lines = np.array(lines)

# # Create the GLLinePlotItem
# cube_edges = gl.GLLinePlotItem(pos=lines, color=(1, 1, 1, 1), width=1, antialias=True)

# x, dx = np.linspace(-x_width / 2, x_width / 2, Nx, retstep=True)
# y, dy = np.linspace(-y_width / 2, y_width / 2, Ny, retstep=True)
# z, dz = np.linspace(-thickness / 2, thickness / 2, Nz, retstep=True)

# x_centre = (x.max() - np.abs(x.min())) / 2
# y_centre = (y.max() - np.abs(y.min())) / 2
# z_centre = (z.max() - np.abs(z.min())) / 2

# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# # Determine the start and end indices to center the voxel within the overall cube
# start_x = (Nx - voxel_shape[0]) // 2
# end_x = start_x + voxel_shape[0]
# start_y = (Ny - voxel_shape[1]) // 2
# end_y = start_y + voxel_shape[1]
# start_z = (Nz - voxel_shape[2]) // 2
# end_z = start_z + voxel_shape[2]

# # Create the lattice pattern only within the voxel region
# X_voxel = X[start_x:end_x, start_y:end_y, start_z:end_z] - x_centre
# Y_voxel = Y[start_x:end_x, start_y:end_y, start_z:end_z] - y_centre

# # Parameters for the lattice of gratings
# line_width = 0.075  # Thickness of each grating line
# grating_period = 0.2  # Spacing between gratings

# # Create the lattice pattern in the voxel region
# grating_X = (np.mod(X_voxel - line_width + grating_period, grating_period) < line_width)
# grating_Y = (np.mod(Y_voxel - line_width + grating_period, grating_period) < line_width)
# attenuation_voxel = np.where(grating_X | grating_Y, 1e11, 0)

# # Initialize the attenuation array and place the voxel data within it
# attenuation = np.zeros((Nx, Ny, Nz))
# attenuation[start_x:end_x, start_y:end_y, start_z:end_z] = attenuation_voxel

# # Sum the attenuation over the z-axis to get a 2D image
# attenuation_2d = np.sum(attenuation, axis=2)

# coor = np.array([X - x_centre, Y - y_centre, Z - z_centre])

# # Apply a random rotation using Euler angles along x, y, and z axes
# random_angles = np.random.uniform(0, 20, size=3)
# rotation = R.from_euler('xyz', random_angles, degrees=True)
# rotation_matrix_inv = np.linalg.inv(rotation.as_matrix())

# coor_prime = np.tensordot(rotation_matrix_inv, coor, axes=((1), (0)))
# xx_prime = coor_prime[0] + x_centre
# yy_prime = coor_prime[1] + y_centre
# zz_prime = coor_prime[2] + z_centre

# # Don't forget to update the interpolation points accordingly
# interpolator = RGI((x, y, z), attenuation, method="linear", bounds_error=False, fill_value=0.0)

# interp_points = np.array([
#     xx_prime.flatten(),
#     yy_prime.flatten(),
#     zz_prime.flatten()
# ]).T.reshape(Nx, Ny, Nz)


# interp_result = interpolator(interp_points).reshape(Nx, Ny, Nz)

# volume_data = interp_result
# volume_data_normalized = ((volume_data - volume_data.min()) / (volume_data.ptp()) * 255).astype(np.uint8)

# # Create an RGBA volume
# volume_data_rgba = np.zeros(volume_data.shape + (4,), dtype=np.uint8)
# volume_data_rgba[..., 0] = volume_data_normalized  # Red channel
# volume_data_rgba[..., 1] = volume_data_normalized  # Green channel
# volume_data_rgba[..., 2] = volume_data_normalized  # Blue channel
# volume_data_rgba[..., 3] = np.where(volume_data_normalized > 0, volume_data_normalized, 0)  # Alpha channel

# # Create the GLVolumeItem
# v = gl.GLVolumeItem(volume_data_rgba, smooth=True, sliceDensity=5)

# v.translate(-150, -150, -45)
# v.scale(dx, dy, dz, local=False)
# w.addItem(v)
# w.addItem(cube_edges)

# if __name__ == '__main__':
#     pg.exec()
