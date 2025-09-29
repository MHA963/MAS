import numpy as np
from irsim.lib import register_behavior
from irsim.util.util import WrapToPi

@register_behavior("diff", "circle_custom")
def beh_diff_circle(ego_object, objects=None, **kwargs):
    """
    Robot circles around the first obstacle exactly once and stops.
    """
    # Get obstacle dynamically
    if objects is None or len(objects) < 2:
        center = np.array([5.0, 5.0])
    else:
        obstacle = objects[1]
        center = obstacle.state[:2, 0]

    state = ego_object.state.flatten()
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.05)

    return circle_vel(ego_object, state, center, max_vel, angle_tolerance)


def circle_vel(ego_object, state, center, max_vel, angle_tolerance=0.05):
    pos = state[:2]
    theta = float(state[2])

    rel = center - pos
    dist = np.linalg.norm(rel)

    # Initialize persistent variables
    if not hasattr(ego_object, 'des_radius'):
        ego_object.des_radius = dist
        ego_object.start_radial = np.arctan2(rel[1], rel[0])
        ego_object.finished = False
        return np.zeros((2,1))  # wait 1 step

    if ego_object.finished:
        return np.zeros((2,1))

    # Current angle relative to start
    radial_ang = np.arctan2(rel[1], rel[0])
    total_rot = WrapToPi(radial_ang - ego_object.start_radial)
    if total_rot < 0:
        total_rot += 2*np.pi  # normalize to [0, 2pi]

    # Stop after one full circle
    if total_rot >= 2*np.pi:
        ego_object.finished = True
        return np.zeros((2,1))

    # Tangential CCW velocity with radius correction
    kp = 0.5
    adjust_ang = -kp * (dist - ego_object.des_radius)
    tang_ang = radial_ang + np.pi/2 + adjust_ang

    diff_radian = WrapToPi(tang_ang - theta)
    linear = max_vel[0,0] * np.cos(diff_radian)
    angular = 0 if abs(diff_radian) < angle_tolerance else max_vel[1,0] * np.sign(diff_radian)

    return np.array([[linear],[angular]])
