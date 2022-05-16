from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_grasp


def object_type(contact_info):
    if "sphere" in contact_info[0].values():
        return "sphere"
    elif "cube" in contact_info[0].values():
        return "cube"
    elif "cylinder" in contact_info[0].values():
        return "cylinder"
    raise RuntimeError("Unexpected object encountered in contact_info")


def reconstruct_object(state, object_type):
    if object_type == "sphere":
        return {"type": "sphere", "com": state[:3], "radius": 0.025}
    elif object_type == "cube":
        return reconstruct_cube(state)
    elif object_type == "cylinder":
        return reconstruct_cylinder(state)
    raise RuntimeError("Unsupported object_type")


def reconstruct_cube(state):
    com = state[:3]
    rotation = state[3:12].reshape(3, 3)
    # Plane definition:
    # Surface origin, surface normal
    # Surface border0 origin, surface border0 normal
    # Surface border1 origin, surface border1 normal
    # Surface border2 origin, surface border2 normal
    # Surface border3 origin, surface border3 normal
    cube_size = 0.025
    surface0 = np.array([[[cube_size, 0, 0], [1, 0, 0]], [[cube_size, cube_size, 0], [0, 1, 0]],
                         [[cube_size, -cube_size, 0], [0, -1, 0]],
                         [[cube_size, 0, cube_size], [0, 0, 1]],
                         [[cube_size, 0, -cube_size], [0, 0, -1]]])

    surface1 = np.array([[[-cube_size, 0, 0], [-1, 0, 0]], [[-cube_size, cube_size, 0], [0, 1, 0]],
                         [[-cube_size, -cube_size, 0], [0, -1, 0]],
                         [[-cube_size, 0, cube_size], [0, 0, 1]],
                         [[-cube_size, 0, -cube_size], [0, 0, -1]]])

    surface2 = np.array([[[0, cube_size, 0], [0, 1, 0]], [[cube_size, cube_size, 0], [1, 0, 0]],
                         [[-cube_size, cube_size, 0], [-1, 0, 0]],
                         [[0, cube_size, cube_size], [0, 0, -1]],
                         [[0, cube_size, -cube_size], [0, 0, 1]]])

    surface3 = np.array([[[0, -cube_size, 0], [0, -1, 0]], [[cube_size, -cube_size, 0], [1, 0, 0]],
                         [[-cube_size, -cube_size, 0], [-1, 0, 0]],
                         [[0, -cube_size, cube_size], [0, 0, -1]],
                         [[0, -cube_size, -cube_size], [0, 0, 1]]])

    surface4 = np.array([[[0, 0, cube_size], [0, 0, 1]], [[cube_size, 0, cube_size], [1, 0, 0]],
                         [[-cube_size, 0, cube_size], [-1, 0, 0]],
                         [[0, cube_size, cube_size], [0, 1, 0]],
                         [[0, -cube_size, cube_size], [0, -1, 0]]])

    surface5 = np.array([[[0, 0, -cube_size], [0, 0, -1]], [[cube_size, 0, -cube_size], [1, 0, 0]],
                         [[-cube_size, 0, -cube_size], [-1, 0, 0]],
                         [[0, cube_size, -cube_size], [0, 1, 0]],
                         [[0, -cube_size, -cube_size], [0, -1, 0]]])

    surfaces = np.array([surface0, surface1, surface2, surface3, surface4, surface5])

    for surface in surfaces:
        for plane in surface:
            for idx, vector in enumerate(plane):
                vector[:] = rotation @ vector + (1 - idx) * com

    return {"type": "cube", "com": com, "surfaces": surfaces}


def reconstruct_cylinder(state):
    com = state[:3]
    rotation = state[3:12].reshape(3, 3)
    cylinder_axis = rotation @ np.array([0, 0, 1.])
    radius = 0.025 / 2
    length = 0.025
    return {
        "type": "cylinder",
        "com": com,
        "cylinder_axis": cylinder_axis,
        "rotation": rotation,
        "radius": radius,
        "length": length
    }


def filter_contact_info(contact_info):
    filtered = filter(lambda x: x["geom1"] is not None, contact_info)
    filtered = filter(lambda x: x["geom2"] is not None, filtered)
    filtered = filter(lambda x: "robot0" in x["geom1"] or "robot0" in x["geom2"], filtered)
    return list(filtered)


if __name__ == "__main__":
    theta_z = np.pi / 4
    rot_matrix = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.],
                           [np.sin(theta_z), np.cos(theta_z), 0.], [0, 0, 1.]])

    rot_matrix = np.array([[1, 0, 0.], [0, np.cos(theta_z), -np.sin(theta_z)],
                           [0, np.sin(theta_z), np.cos(theta_z)]])

    state = np.concatenate(([1.3697697, 0.8, 0.4], rot_matrix.flatten()))

    otype = "sphere"

    file = Path(__file__).parent / ("contact_info_" + otype + ".json")
    with open(file, "r") as f:
        contact_info = json.load(f)["contact_info"]

    contact_info = filter_contact_info(contact_info)

    object_description = reconstruct_object(state, object_type(contact_info))

    fig, ax = visualize_grasp(object_description, contact_info)
    plt.show()
