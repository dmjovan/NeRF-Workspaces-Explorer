from collections import namedtuple

HW = namedtuple("HW", ["h", "w"])
HW.__new__.__defaults__ = (0, 0)
HW.__str__ = lambda d: f"h = {d.h}, w = {d.w}"
HW.__ge__ = lambda self, other: (self.h >= other.h) and (self.w >= other.w)
HW.__le__ = lambda self, other: (self.h <= other.h) and (self.w <= other.w)
HW.__copy__ = lambda self: HW(self.h, self.w)

XYZ = namedtuple("XYZ", ["x", "y", "z"])
XYZ.__new__.__defaults__ = (0.0, 0.0, 0.0)
XYZ.__str__ = lambda c: f"x = {c.x}, y = {c.y}, z = {c.z}"
XYZ.__copy__ = lambda self: XYZ(self.x, self.y, self.z)

COORD = namedtuple("COORD", ["x", "y", "z", "yaw", "pitch", "roll"])
COORD.__new__.__defaults__ = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
COORD.__str__ = lambda c: f"x = {c.x:.3f}, \n" \
                          f"y = {c.y:.3f}, \n" \
                          f"z = {c.z:.3f}, \n" \
                          f"pitch (rotation over X axis) = {c.pitch:.3f}, \n" \
                          f"yaw (rotation over Y axis) = {c.yaw:.3f}, \n" \
                          f"roll (rotation over Z axis) = {c.roll:.3f}"
COORD.__copy__ = lambda self: COORD(self.x, self.y, self.z, self.yaw, self.pitch, self.roll)
