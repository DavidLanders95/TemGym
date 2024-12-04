import numpy as np
from scipy.ndimage import rotate
from scipy import special
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

ix, iy, iz = 10, 10, 5
x_px = np.arange(ix)
y_px = np.arange(iy)
z_px = np.arange(iz)

Z, Y, X = np.meshgrid(x_px, y_px, z_px, indexing='ij')
voxelarray = np.ones_like(X)

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, edgecolor='k')


random_angles = np.random.uniform(45, 45, size=1)
rotation = R.from_euler('z', random_angles, degrees=True)
rotation_matrix = rotation.as_matrix()[0]

# Define the coordinates of the 8 corners of the volume
corners = np.array([[0, 0, 0],
                    [0, 0, iz],
                    [0, iy, 0],
                    [0, iy, iz],
                    [ix, 0, 0],
                    [ix, 0, iz],
                    [ix, iy, 0],
                    [ix, iy, iz]]).T

coords = np.array([X.ravel(), Y.ravel(), Z.ravel()])

# # Apply rotation matrix to the corners
rotated_corners = rotation_matrix @ corners
rotated_coords = rotation_matrix @ coords
rotated_coords = np.round(rotated_coords).astype(int)

# # Compute the output shape
out_volume_shape = (np.ptp(rotated_corners, axis=1) + 0.5).astype(int)

out_volume = np.zeros(out_volume_shape)

out_volume[rotated_coords[0], rotated_coords[1], rotated_coords[2]] = voxelarray.ravel()


# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(out_volume, edgecolor='k')

plt.show()