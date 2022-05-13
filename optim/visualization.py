import matplotlib.pyplot as plt
import numpy as np


def visualize_cylinder(object_description, contact_points=None):
    radius = object_description["radius"]
    length = object_description["length"]
    com = object_description["com"]
    rotation = object_description["rotation"]

    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    ax[0].set_title("Initial contact points")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim(com[0] + length * np.array([-2, 2]))
    ax[0].set_ylim(com[1] + length * np.array([-2, 2]))
    ax[0].set_zlim(com[2] + length * np.array([-2, 2]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    z = np.linspace(-length / 2, length / 2, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    xrot = x * rotation[0, 0] + y * rotation[0, 1] + z * rotation[0, 2] + com[0]
    yrot = x * rotation[1, 0] + y * rotation[1, 1] + z * rotation[1, 2] + com[1]
    zrot = x * rotation[2, 0] + y * rotation[2, 1] + z * rotation[2, 2] + com[2]
    ax[0].plot_surface(xrot, yrot, zrot, alpha=0.4)

    R = np.linspace(0, radius, 20)
    u = np.linspace(0, 2 * np.pi, 20)
    x = np.outer(R, np.cos(u))
    y = np.outer(R, np.sin(u))
    z = np.ones_like(x) * length / 2
    xrot = x * rotation[0, 0] + y * rotation[0, 1] + z * rotation[0, 2] + com[0]
    yrot = x * rotation[1, 0] + y * rotation[1, 1] + z * rotation[1, 2] + com[1]
    zrot = x * rotation[2, 0] + y * rotation[2, 1] + z * rotation[2, 2] + com[2]
    ax[0].plot_surface(xrot, yrot, zrot, alpha=0.4)
    z = -np.ones_like(x) * length / 2
    xrot = x * rotation[0, 0] + y * rotation[0, 1] + z * rotation[0, 2] + com[0]
    yrot = x * rotation[1, 0] + y * rotation[1, 1] + z * rotation[1, 2] + com[1]
    zrot = x * rotation[2, 0] + y * rotation[2, 1] + z * rotation[2, 2] + com[2]
    ax[0].plot_surface(xrot, yrot, zrot, alpha=0.4)
    return fig, ax


def visualize_cube(object_description):
    surfaces = object_description["surfaces"]
    com = object_description["com"]

    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    dst = np.linalg.norm(surfaces[0, 1, 0] - surfaces[0, 2, 0])
    ax[0].set_title("cube")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim(com[0] + dst * np.array([-2, 2]))
    ax[0].set_ylim(com[1] + dst * np.array([-2, 2]))
    ax[0].set_zlim(com[2] + dst * np.array([-2, 2]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    for surface in surfaces:
        for i in range(1, len(surface)):
            j = 3 if i < 3 else 1
            p0 = surface[i, 0] + (surface[j, 0] - surface[0, 0])
            p1 = surface[i, 0] + (surface[j + 1, 0] - surface[0, 0])
            ax[0].plot3D(*zip(p0, p1), color="k")
    return fig, ax


def visualize_sphere(object_description):
    com = object_description["com"]
    radius = object_description["radius"]

    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    ax[0].set_title("Initial contact points")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim(com[0] + radius * np.array([-2, 2]))
    ax[0].set_ylim(com[1] + radius * np.array([-2, 2]))
    ax[0].set_zlim(com[2] + radius * np.array([-2, 2]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v) * radius + com[0]
    y = np.sin(u) * np.sin(v) * radius + com[1]
    z = np.cos(v) * radius + com[2]
    ax[0].plot_surface(x, y, z, alpha=0.4)
    return fig, ax


def visualize_object(object_description):
    if object_description["type"] == "sphere":
        return visualize_sphere(object_description)
    if object_description["type"] == "cube":
        return visualize_cube(object_description)
    if object_description["type"] == "cylinder":
        return visualize_cylinder(object_description)
    raise RuntimeError("Unsupported object type")