from jax import jacfwd
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from optim.grippers.kinematics.tf import tf_matrix, zrot_matrix

_FLOAT_EPS = jnp.finfo(jnp.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def visualize_frames(frames, frame_pairs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    minc = np.array([np.inf, np.inf, np.inf])  # Necessary for scaling 3D plot
    maxc = np.array([-np.inf, -np.inf, -np.inf])
    for frame in frames:
        pos = frame[0:3, 3]
        minc = np.minimum(minc, pos)
        maxc = np.maximum(maxc, pos)
        ex, ey, ez = frame[0:3, 0] * .1 + pos, frame[0:3, 1] * .1 + pos, frame[0:3, 2] * .1 + pos
        ax.scatter(*pos, color="0")
        for ei, color in zip([ex, ey, ez], ["r", "g", "b"]):
            minc = np.minimum(minc, ei)
            maxc = np.maximum(maxc, ei)
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=color)
    for frame in frames:
        pos = frame[0:3, 3]
        minc = np.minimum(minc, pos)
        maxc = np.maximum(maxc, pos)
    for idx in range(1, len(frames)):
        color = "#cc0000"
        base = frame_pairs[idx]
        x = [frames[base][0, 3], frames[idx][0, 3]]
        y = [frames[base][1, 3], frames[idx][1, 3]]
        z = [frames[base][2, 3], frames[idx][2, 3]]
        ax.plot(x, y, z, color=color)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_box_aspect((maxc - minc))
    plt.show()


def jaxmat2euler(mat):
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = jnp.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = jnp.empty(mat.shape[:-1])
    euler = euler.at[..., 2].set(
        jnp.where(condition, -jnp.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                  -jnp.arctan2(-mat[..., 1, 0], mat[..., 1, 1])))
    euler = euler.at[..., 1].set(
        jnp.where(condition, -jnp.arctan2(-mat[..., 0, 2], cy), -jnp.arctan2(-mat[..., 0, 2], cy)))
    euler = euler.at[...,
                     0].set(jnp.where(condition, -jnp.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0))
    return euler


def k1(x):
    l1_T_l2 = tf_matrix(np.array([0.85, 0, 0, 0, 0, 0]))
    l2_T_c1 = tf_matrix(np.array([0.3, -0.05, 0, 0, 0, -np.pi / 2]))

    l3_T_l4 = tf_matrix(np.array([0, 0, 0.4, 0, np.pi / 4, 0])) @ zrot_matrix(np.pi)
    l4_T_l5 = tf_matrix(np.array([0.45, 0, 0, -np.pi / 2, 0, 0]))
    l5_T_c2 = tf_matrix(np.array([0.2, 0, 0, 0, 0, 0]))

    w_T_l1 = zrot_matrix(x[0])
    w_T_l2 = w_T_l1 @ l1_T_l2 @ zrot_matrix(x[1])
    w_T_c1 = w_T_l2 @ l2_T_c1

    w_T_l3 = tf_matrix(np.array([1, 0, 0, -np.pi / 2, 0, -0.4])) @ zrot_matrix(x[2])
    w_T_l4 = w_T_l3 @ l3_T_l4 @ zrot_matrix(x[3])
    w_T_l5 = w_T_l4 @ l4_T_l5 @ zrot_matrix(x[4])
    w_T_c2 = w_T_l5 @ l5_T_c2

    l7, l3 = np.abs(np.linalg.inv(w_T_c1) @ w_T_l2)[:2, 3]
    l2, l1 = np.abs(np.linalg.inv(w_T_c1)[:2, 3])
    l6 = 0.2
    l4, l5 = np.abs(np.linalg.inv(w_T_c2) @ w_T_l4)[:2, 3]

    print(f"l1: {l1}, l2: {l2}, l3: {l3}, l4: {l4}, l5: {l5}, l6: {l6}, l7: {l7}")
    return w_T_l1, w_T_l2, w_T_c1, w_T_l3, w_T_l4, w_T_l5, w_T_c2


def jaxk1(x):
    l1_T_l2 = tf_matrix(np.array([0.85, 0, 0, 0, 0, 0]))
    l2_T_c1 = tf_matrix(np.array([0.3, -0.05, 0, 0, 0, -np.pi / 2]))

    l3_T_l4 = tf_matrix(np.array([0, 0, 0.4, 0, np.pi / 4, 0])) @ zrot_matrix(np.pi)
    l4_T_l5 = tf_matrix(np.array([0.45, 0, 0, -np.pi / 2, 0, 0]))
    l5_T_c2 = tf_matrix(np.array([0.2, 0, 0, 0, 0, 0]))

    w_T_l1 = zrot_matrix(x[0])
    w_T_l2 = w_T_l1 @ l1_T_l2 @ zrot_matrix(x[1])
    w_T_c1 = w_T_l2 @ l2_T_c1

    w_T_l3 = tf_matrix(np.array([1, 0, 0, -np.pi / 2, 0, -0.4])) @ zrot_matrix(x[2])
    w_T_l4 = w_T_l3 @ l3_T_l4 @ zrot_matrix(x[3])
    w_T_l5 = w_T_l4 @ l4_T_l5 @ zrot_matrix(x[4])
    w_T_c2 = w_T_l5 @ l5_T_c2
    euler = jaxmat2euler(w_T_c1[:3, :3])
    return jnp.concatenate(
        (w_T_c1[:3, 3], w_T_c1[:3, :3].flatten(), w_T_c2[:3, 3], w_T_c2[:3, :3].flatten()))


def jaxk1_euler(x):
    l1_T_l2 = tf_matrix(np.array([0.85, 0, 0, 0, 0, 0]))
    l2_T_c1 = tf_matrix(np.array([0.3, -0.05, 0, 0, 0, -np.pi / 2]))

    l3_T_l4 = tf_matrix(np.array([0, 0, 0.4, 0, np.pi / 4, 0])) @ zrot_matrix(np.pi)
    l4_T_l5 = tf_matrix(np.array([0.45, 0, 0, -np.pi / 2, 0, 0]))
    l5_T_c2 = tf_matrix(np.array([0.2, 0, 0, 0, 0, 0]))

    w_T_l1 = zrot_matrix(x[0])
    w_T_l2 = w_T_l1 @ l1_T_l2 @ zrot_matrix(x[1])
    w_T_c1 = w_T_l2 @ l2_T_c1

    w_T_l3 = tf_matrix(np.array([1, 0, 0, -np.pi / 2, 0, -0.4])) @ zrot_matrix(x[2])
    w_T_l4 = w_T_l3 @ l3_T_l4 @ zrot_matrix(x[3])
    w_T_l5 = w_T_l4 @ l4_T_l5 @ zrot_matrix(x[4])
    w_T_c2 = w_T_l5 @ l5_T_c2
    return jnp.concatenate(
        (w_T_c1[:3, 3], jaxmat2euler(w_T_c1[:3, :3]), w_T_c2[:3, 3], jaxmat2euler(w_T_c2[:3, :3])))


def skew_matrix(x):
    assert x.shape == (3,)
    return np.array([[0, -x[2], x[1]], [-x[2], 0, -x[0]], [-x[1], x[0], 0]])


def grasp_jac_k1(x):
    frames = k1(x)
    Jbar = np.zeros((12, 5))

    Jbar[0:3, 0] = np.cross(frames[0][:3, 3] - frames[2][:3, 3], frames[0][:3, 2])
    Jbar[3:6, 0] = frames[0][:3, 2]
    Jbar[0:3, 1] = np.cross(frames[1][:3, 3] - frames[2][:3, 3], frames[1][:3, 2])
    Jbar[3:6, 1] = frames[1][:3, 2]
    Jbar[6:9, 2] = np.cross(frames[3][:3, 3] - frames[6][:3, 3], frames[3][:3, 2])
    Jbar[9:12, 2] = frames[3][:3, 2]
    Jbar[6:9, 3] = np.cross(frames[4][:3, 3] - frames[6][:3, 3], frames[4][:3, 2])
    Jbar[9:12, 3] = frames[4][:3, 2]
    Jbar[6:9, 4] = np.cross(frames[5][:3, 3] - frames[6][:3, 3], frames[5][:3, 2])
    Jbar[9:12, 4] = frames[5][:3, 2]

    _zero = np.zeros((3, 3))
    Rbar_1 = np.block([[frames[2][:3, :3], _zero], [_zero, frames[2][:3, :3]]])
    Rbar_2 = np.block([[frames[6][:3, :3], _zero], [_zero, frames[6][:3, :3]]])

    H = np.zeros((8, 12))
    for idx in range(4):
        H[idx, idx] = 1
        H[idx + 4, idx + 6] = 1

    Jbar[:6, :] = Rbar_1.T @ Jbar[:6, :]
    Jbar[6:, :] = Rbar_2.T @ Jbar[6:, :]
    return H @ Jbar


_jac_k1 = jacfwd(jaxk1)


def jac_k1(x):
    _jac = _jac_k1(x)
    jac = np.zeros((_jac.shape[0] // 2, _jac.shape[1]))
    _jac_idx = np.indices(_jac.shape)[0][:, 0] % 12
    jac_idx = np.indices(jac.shape)[0][:, 0] % 6
    jac[jac_idx < 3, :] = _jac[_jac_idx < 3, :]
    # Extract wx, wy and wz from the skew symmetric elements
    jac[jac_idx == 3, :] = -_jac[_jac_idx == 8, :]
    jac[jac_idx == 4, :] = _jac[_jac_idx == 5, :]
    jac[jac_idx == 5, :] = -_jac[_jac_idx == 6, :]
    # Normalize rotations
    jac_norm = jac[jac_idx >= 3, :].T.reshape((-1, 3))
    norm = np.linalg.norm(jac_norm, axis=1, keepdims=True)
    norm[norm == 0] = 1
    jac[jac_idx >= 3, :] = (jac_norm / norm).reshape((jac[jac_idx >= 3, :].shape[1], -1)).T
    return jac


_jac_k1_euler = jacfwd(jaxk1_euler)


def jac_k1_euler(x):
    FRAMES = (2, 6)
    frames = k1(x)
    jac = _jac_k1_euler(x)
    # Transform to local contact frame
    R = np.zeros((jac.shape[0], jac.shape[0]))
    for i, frame in enumerate(FRAMES):
        R[i * 6:i * 6 + 3, i * 6:i * 6 + 3] = frames[frame][:3, :3]
        R[i * 6 + 3:(i + 1) * 6, i * 6 + 3:(i + 1) * 6] = frames[frame][:3, :3]
    rjac = R.T @ jac
    # Zero out non-transferable forces (torque elements at index 4 and 5)
    rjac = rjac.at[np.indices(rjac.shape)[0][:, 0] % 6 > 3, :].set(0)
    # Transform to global frame
    rjac = R @ rjac
    return rjac


def main():
    jnp.set_printoptions(precision=3, suppress=True)
    x1 = jnp.array([1.3, 0.8, 0, 0, np.pi / 4])
    print(grasp_jac_k1(x1))
    print(jac_k1_euler(x1))


if __name__ == "__main__":
    main()
