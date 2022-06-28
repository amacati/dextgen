import matplotlib.pyplot as plt
import numpy as np


def visualize_frames(frames, gripper_type):
    if gripper_type == "sh":
        frame_pairs = {key: value for key, value in zip(range(1, 27), range(0, 26))}
        for idx in [8, 12, 16, 20]:
            frame_pairs[idx] = 2
    elif gripper_type == "pj":
        frame_pairs = {1: 0, 2: 1, 3: 0, 4: 3}
    elif gripper_type == "bh":
        frame_pairs = {key: value for key, value in zip(range(1, 12), range(0, 11))}
        for idx in [5, 9]:
            frame_pairs[idx] = 0
    else:
        raise RuntimeError(f"Gripper type {gripper_type} not supported")
    return _visualize_frames(frames, frame_pairs)


def _visualize_frames(frames, frame_pairs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    minc = np.array([np.inf, np.inf, np.inf])  # Necessary for scaling 3D plot
    maxc = np.array([-np.inf, -np.inf, -np.inf])
    for frame in frames:
        pos = frame[0:3, 3]
        minc = np.minimum(minc, pos)
        maxc = np.maximum(maxc, pos)
        ex, ey, ez = frame[0:3, 0] * .01 + pos, frame[0:3, 1] * .01 + pos, frame[0:3, 2] * .01 + pos
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
