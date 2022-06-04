"""Functions to define basic CAD building blocks and logic for their construction"""

import math
from madcad import Axis, Circle, Segment, extrusion, flatsurface, thicken, vec3, web
from stl import Mesh


def build_plate(x: float, y: float, z: float) -> Mesh:
    """Construct a plate.

    Args:
        x: Dimension in x direction.
        y: Dimension in y direction.
        z: Dimension in z direction.

    Returns:
        A plate as a mesh object.
    """
    A = vec3(0, 0, 0)
    B = vec3(x, 0, 0)
    line = web([Segment(A, B)])
    face = extrusion(vec3(0, y, 0), line)
    plate = thicken(face, z)
    return plate


def build_cylinder(x: float, y: float, h: float, r: float) -> Mesh:
    """Construct a cylinder.

    Args:
        x: position in x.
        y: position in y.
        h: cylinder height.
        r: cylinder radius.

    Returns:
        mesh: A cylinder object.
    """
    A = vec3(x, y, h)
    B = vec3(0.0, 0.0, 1.0)
    axis = Axis(A, B)
    circle = flatsurface(Circle(axis, r))
    cylinder = thicken(circle, 3 * h)
    return cylinder


def is_on_circle(p_x: float, p_y: float, c_x: float, c_y: float, r: float) -> bool:
    """Returns whether a 2D-point is on a circle

    Args:
        p_x: x-coordinate of the point
        p_y: y-coordinate of the point
        c_x: x_coordinate of the center of the circle
        c_y: y-coordinate of the center of the circle
        r: radius of the circle

    Returns:
        Whether the point is on the circle
    """
    d_x = abs(p_x - c_x)
    d_y = abs(p_y - c_y)
    d = math.sqrt(d_x ** 2 + d_y ** 2)

    return d <= r
