from functools import singledispatch

import numpy as np
import matplotlib.pyplot as plt

from optim.geometry import Cube, Cylinder, Sphere


@singledispatch
def visualize_object(obj):
    raise TypeError(f"Object type {type(obj)} not supported")


@visualize_object.register
def _(obj: Cube):
    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    dst = np.linalg.norm(obj.planes[0, 1, 0] - obj.planes[0, 2, 0])
    ax[0].set_title("cube")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim(obj.com[0] + dst * np.array([-2, 2]))
    ax[0].set_ylim(obj.com[1] + dst * np.array([-2, 2]))
    ax[0].set_zlim(obj.com[2] + dst * np.array([-2, 2]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    for k, surface in enumerate(obj.planes):
        for i in range(1, len(surface)):
            j = 3 if i < 3 else 1
            p0 = surface[i, 0] + (surface[j, 0] - surface[0, 0])
            p1 = surface[i, 0] + (surface[j + 1, 0] - surface[0, 0])
            ax[0].plot3D(*zip(p0, p1), color="k")
        # ax[0].scatter(*surface[0, 0], color=("r", "g", "b", "c", "m", "y")[k])
    # Plot table surface
    xsupport = np.linspace(*(obj.com[0] + dst * np.array([-1, 1])), 2)
    ysupport = np.linspace(*(obj.com[1] + dst * np.array([-1, 1])), 2)
    x, y = np.meshgrid(xsupport, ysupport)
    z = np.ones_like(x) * 0.4
    ax[0].plot_surface(x, y, z, alpha=0.4)
    return fig


@visualize_object.register
def _(obj: Cylinder):
    radius = ...
    length = ...
    com = ...
    rotation = ...

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
    return fig


@visualize_object.register
def _(obj: Sphere):
    com = ...
    radius = ...

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

    # for contact in contact_info:
    #     ax[0].scatter(*contact["pos"])
    return
