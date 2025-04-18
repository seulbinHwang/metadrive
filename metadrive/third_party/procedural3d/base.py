# Author: Epihaius
# Date: 2019-06-21
#
# This module contains a couple of imports and a base class needed for the
# creation of all 3D primitives.

from panda3d.core import *
from math import pi, sin, cos, acos, atan2, sqrt
import array


class ModelMaker:

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    @property
    def inverted(self):
        return self._inverted

    @inverted.setter
    def inverted(self, inverted):
        self._inverted = inverted

    @property
    def vertex_color(self):
        return self._vertex_color

    @vertex_color.setter
    def vertex_color(self, vertex_color):
        self._vertex_color = vertex_color

    @property
    def has_uvs(self):
        return self._has_uvs

    @has_uvs.setter
    def has_uvs(self, has_uvs):
        self._has_uvs = has_uvs

    @property
    def tex_units(self):
        return self._tex_units

    @tex_units.setter
    def tex_units(self, tex_units):
        self._tex_units = tex_units

    @property
    def tex_offset(self):
        return self._tex_offset

    @tex_offset.setter
    def tex_offset(self, tex_offset):
        self._tex_offset = tex_offset

    @property
    def tex_rotation(self):
        return self._tex_rotation

    @tex_rotation.setter
    def tex_rotation(self, tex_rotation):
        self._tex_rotation = tex_rotation

    @property
    def tex_scale(self):
        return self._tex_scale

    @tex_scale.setter
    def tex_scale(self, tex_scale):
        self._tex_scale = tex_scale

    @property
    def vertex_ranges(self):
        """
        The ranges of vertex indices for each surface, as a dict of
        (start_index, end_index) tuples (empty tuple if the surface
        was not created), with end_index not included in the range.

        See derived class documentation for available surfaces.

        """
        return self._vert_ranges

    def __init__(self, segments, inverted, vertex_color, has_uvs, tex_units,
                 tex_offset, tex_rotation, tex_scale, surface_ids):
        """
        This class generates model primitives with the given parameters, common to
        all primitive types:

            segments (dict of ints):
                the number of subdivisions of each surface (listed in derived class
                documentation);
                default = None (use defaults for all surfaces);
            inverted (bool):
                whether or not the geometry should be rendered inside-out;
                default is False;
            vertex_color (tuple or None):
                the color applied to all vertices (if not specified, the vertex format
                of the created geometry will not allow any color data to be set);
                default = None;
            has_uvs (bool):
                whether or not the model should have texture coordinates;
                default is True;
            tex_units (dict of float tuples):
                the texture size (width, height) in object-space units for each
                surface (listed in derived class documentation);
                default = None;
            tex_offset, tex_rotation, tex_scale (dicts of float tuples):
                the 2D texture transforms in object-space for each surface
                (listed in derived class documentation):
                    tex_offset: (u offset, v offset);
                    tex_rotation: angle in degrees;
                    tex_scale: (u scale, v scale);
                default = None.

        """

        self._segments = segments
        self._inverted = inverted
        self._vertex_color = vertex_color
        self._has_uvs = has_uvs
        self._tex_units = tex_units
        self._tex_offset = tex_offset
        self._tex_rotation = tex_rotation
        self._tex_scale = tex_scale
        self._surface_ids = surface_ids
        self._vert_ranges = {s_id: () for s_id in surface_ids}

    def reset(self):

        self._segments = None
        self._inverted = False
        self._vertex_color = None
        self._has_uvs = True
        self._tex_units = None
        self._tex_offset = None
        self._tex_rotation = None
        self._tex_scale = None
        self._vert_ranges = {s_id: () for s_id in self._surface_ids}

    def _make_flat_shaded(self, indices, verts):

        points = [Point3(verts[i]["pos"]) for i in indices[:3]]
        normal = Plane(*points).get_normal()

        for i in indices:
            verts[i]["normal"] = normal

    def _average_normals(self, index1, index2, verts):

        normal = (verts[index1]["normal"] +
                  verts[index2]["normal"]).normalized()
        verts[index1]["normal"] = normal
        verts[index2]["normal"] = normal

    def _get_tex_xform(self, surface_id):

        tex_offset = self._tex_offset
        tex_rotation = self._tex_rotation
        tex_scale = self._tex_scale

        has_tex_offset = tex_offset and surface_id in tex_offset
        has_tex_rot = tex_rotation and surface_id in tex_rotation
        has_tex_scale = tex_scale and surface_id in tex_scale
        has_tex_xform = has_tex_offset or has_tex_rot or has_tex_scale

        if has_tex_xform:

            mat = Mat3.ident_mat()

            if has_tex_scale:
                mat = mat * Mat3.scale_mat(*tex_scale[surface_id])
            if has_tex_rot:
                mat = mat * Mat3.rotate_mat(tex_rotation[surface_id])
            if has_tex_offset:
                mat = mat * Mat3.translate_mat(*tex_offset[surface_id])

            return mat
