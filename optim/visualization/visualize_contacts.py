def visualize_contacts(fig, obj, gripper):
    ax = fig.axes[0]
    grasp_forces = gripper.create_grasp_forces(obj.con_links, obj.con_pts)
    kinematics = gripper.create_full_frames(obj.con_links, obj.con_pts)
    frames = kinematics(gripper.state)
    forces = grasp_forces(gripper.state)
    for frame, con_force, con_link in zip(frames, forces, obj.con_links):
        pos = frame[:3, 3]
        ex, ey, ez = frame[:3, 0] * .01 + pos, frame[:3, 1] * .01 + pos, frame[:3, 2] * .01 + pos
        color = "r" if con_link == "robot0:r_gripper_finger_link" else "g"
        ax.scatter(*pos, color=color)
        for ei, color in zip([ex, ey, ez], ["r", "g", "b"]):
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=color)
        con_force /= 1e5
        fx, fy, fz = con_force[:3] + pos
        ax.plot([pos[0], fx], [pos[1], fy], [pos[2], fz], color="r")
    return fig
