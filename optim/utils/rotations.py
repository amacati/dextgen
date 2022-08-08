import numpy as np
import jax.numpy as jnp


def mat2quat(mat):
    assert mat.shape == (3, 3)
    tr0 = 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]
    tr1 = 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]
    tr2 = 1.0 - mat[0, 0] + mat[1, 1] - mat[2, 2]
    tr3 = 1.0 - mat[0, 0] - mat[1, 1] + mat[2, 2]

    # Calculate which conversion to take for which matrix for best numeric stability
    q = np.empty(mat.shape[:-2] + (4,))

    if tr0 > 0:
        s0 = np.sqrt(tr0) * 2
        q[0] = 0.25 * s0
        q[1] = (mat[2, 1] - mat[1, 2]) / s0
        q[2] = (mat[0, 2] - mat[2, 0]) / s0
        q[3] = (mat[1, 0] - mat[0, 1]) / s0

    elif tr1 > tr2 and tr1 > tr3:
        s1 = np.sqrt(tr1) * 2
        q[0] = (mat[2, 1] - mat[1, 2]) / s1
        q[1] = 0.25 * s1
        q[2] = (mat[0, 1] + mat[1, 0]) / s1
        q[3] = (mat[0, 2] + mat[2, 0]) / s1
    elif tr2 > tr1 and tr2 > tr3:
        s2 = np.sqrt(tr2) * 2
        q[0] = (mat[0, 2] - mat[2, 0]) / s2
        q[1] = (mat[0, 1] + mat[1, 0]) / s2
        q[2] = 0.25 * s2
        q[3] = (mat[1, 2] + mat[2, 1]) / s2
    else:
        s3 = np.sqrt(tr3) * 2
        q[0] = (mat[1, 0] - mat[0, 1]) / s3
        q[1] = (mat[0, 2] + mat[2, 0]) / s3
        q[2] = (mat[1, 2] + mat[2, 1]) / s3
        q[3] = 0.25 * s3

    if q[0] < 0:
        q *= -1  # Prefer quaternion with positive w
    return q


def quat2mat(q):
    r11 = 1 - 2. * (q[2]**2 + q[3]**2)
    r12 = 2. * (q[1] * q[2] - q[0] * q[3])
    r13 = 2. * (q[1] * q[3] + q[0] * q[2])
    r21 = 2. * (q[1] * q[2] + q[0] * q[3])
    r22 = 1 - 2. * (q[1]**2 + q[3]**2)
    r23 = 2. * (q[2] * q[3] - q[0] * q[1])
    r31 = 2. * (q[1] * q[3] - q[0] * q[2])
    r32 = 2. * (q[2] * q[3] + q[0] * q[1])
    r33 = 1 - 2. * (q[1]**2 + q[2]**2)
    return jnp.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def q2mat(q):
    Nq = np.dot(q, q)
    s = 2.0 / Nq
    w, x, y, z = q
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    mat = np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                    [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])
    return mat
