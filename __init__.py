bl_info = {
    "name": "FK Foam Skeleton",
    "author": "AI research drawing",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > FK Foam",
    "description": "Frank-Kasper (A15) foam metal skeleton with an arbitrary clipping shape",
    "category": "Add Mesh",
}

"""FK foam skeleton - UI add-on (runtime registration, no preferences touched).

Registers:
  * a sidebar panel   View3D > N > "FK Foam"
  * the operator       fk_foam.build

Every value in the panel is stored on the Scene, so the panel can be reopened
any time.  The clipping shape is simply another object in the scene: pick any
mesh (closed or not - it is voxel-remeshed into a solid first), press Build.

Install (either way works):
  * open `fk_foam_ui.py` in the Text Editor and press Run Script, or
  * Edit > Preferences > Add-ons > Install... and pick this file.
"""
import math

import bpy
import bmesh
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

import traceback

from . import fk_nodes
from . import fk_shape


# ------------------------------------------------------------------ helpers --
def _domain_object(scene, ctx):
    """The clipping shape chosen in the panel - never guessed from selection."""
    ob = getattr(scene, "fk_domain_obj", None)
    if ob is None or ob.name not in bpy.data.objects or ob.type != 'MESH':
        return None
    if ob.name in ("FK_Node_Network", "FK_Foam_Edges", "_fk_graph", "FK_Domain"):
        return None
    if len(ob.data.polygons) == 0:
        return None
    return ob


def ensure_domain(scene):
    """Return the clipping shape; create a 30 mm cube when none is set."""
    ob = getattr(scene, "fk_domain_obj", None)
    if ob is not None and ob.name in bpy.data.objects and ob.type == 'MESH':
        return ob
    bpy.ops.mesh.primitive_cube_add(size=30.0, location=(15.0, 15.0, 15.0))
    ob = bpy.context.active_object
    ob.name = "FK_DomainSource"
    ob.display_type = 'WIRE'
    ob.hide_render = True
    scene.fk_domain_obj = ob
    return ob


def build(context, report=None):
    """Shape-first build: the FK graph itself is restricted to the domain."""
    scene = context.scene
    net = bpy.data.objects.get("FK_Node_Network")
    if net is None:
        net = bpy.data.objects.new("FK_Node_Network", bpy.data.meshes.new("FK_Node_Network"))
        scene.collection.objects.link(net)

    src = bpy.data.objects.get("FK_Foam_Edges")
    if src is None or len(src.data.edges) == 0:
        src = make_source_graph(int(getattr(scene, "fk_cells", 3)), 10.0)
        if report:
            report({'INFO'}, "created FK_Foam_Edges")

    dom = _domain_object(scene, context)
    if dom is None:
        if report:
            report({'ERROR'},
                   "请先选择一个源形状" if ui_lang() == "ZH"
                   else "Please select a source shape first")
        return None
    if dom is not None:
        V, E, kept = fk_shape.complete_cell_graph(
            src, dom, N=3, keep_ratio=scene.fk_cell_keep)
        me = bpy.data.meshes.new("_fk_graph")
        me.from_pydata([tuple(v) for v in V], E, [])
        me.update()
        tmp = bpy.data.objects.new("_fk_graph", me)
        scene.collection.objects.link(tmp)
        try:
            fk_nodes.rebuild_object(net, tmp,
                                    node_size=scene.fk_node_size,
                                    bevel=scene.fk_bevel,
                                    legs=True,
                                    leg_scale=1.0,
                                    edge_bevel=scene.fk_edge_bevel,
                                    bevel_segments=1,
                                    cap_inset=scene.fk_cap_inset,
                                    subsurf=scene.fk_subsurf,
                                    complete_nodes=True)
        finally:
            bpy.data.objects.remove(tmp, do_unlink=True)
            bpy.data.meshes.remove(me)
        net["cells"] = kept
    else:
        fk_nodes.rebuild_object(net, src,
                                node_size=scene.fk_node_size,
                                bevel=scene.fk_bevel,
                                legs=True,
                                leg_scale=1.0,
                                edge_bevel=scene.fk_edge_bevel,
                                bevel_segments=1,
                                cap_inset=scene.fk_cap_inset,
                                subsurf=scene.fk_subsurf)

    net["node_size"] = scene.fk_node_size
    net["bevel"] = scene.fk_bevel
    net["edge_bevel"] = scene.fk_edge_bevel
    net["cap_inset"] = scene.fk_cap_inset
    for poly in net.data.polygons:
        poly.use_smooth = True
    if report:
        report({'INFO'}, "FK foam: %d faces" % len(net.data.polygons))
    return net




def make_source_graph(cells=3, a=10.0, delta=0.0024802625):
    """Create (or replace) FK_Foam_Edges - the A15 foam edge graph."""
    import numpy as np
    from . import fk_geom

    pol = fk_geom.build_cells(delta)
    verts, index, edges = [], {}, set()

    def vid(pt):
        k = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
        i = index.get(k)
        if i is None:
            i = len(verts)
            index[k] = i
            verts.append(k)
        return i

    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                off = np.array([i, j, k], float)
                for cv, cf in pol:
                    pts = (np.asarray(cv, float) + off) * a
                    for f in cf:
                        for p_, q_ in zip(f, f[1:] + f[:1]):
                            edges.add((min(vid(pts[p_]), vid(pts[q_])),
                                       max(vid(pts[p_]), vid(pts[q_]))))

    old = bpy.data.objects.get("FK_Foam_Edges")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new("FK_Foam_Edges")
    me.from_pydata([tuple(v) for v in verts], sorted(edges), [])
    me.update()
    ob = bpy.data.objects.new("FK_Foam_Edges", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.hide_render = True
    ob.display_type = 'WIRE'
    return ob


class FK_OT_source(bpy.types.Operator):
    """Create the A15 source edge graph (FK_Foam_Edges)"""
    bl_idname = "fk_foam.make_source"
    bl_label = "创建源边图 / Source graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = make_source_graph(int(context.scene.fk_cells), 10.0)
        self.report({'INFO'}, "FK_Foam_Edges: %d edges" % len(ob.data.edges))
        return {'FINISHED'}


# ---------------------------------------------------------------- operator --
class FK_OT_build(bpy.types.Operator):
    """Rebuild the FK foam skeleton and clip it by the chosen domain object"""
    bl_idname = "fk_foam.build"
    bl_label = "生成泡沫骨架 / Build"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if build(context, self.report) is None:
            return {'CANCELLED'}
        return {'FINISHED'}


class FK_OT_pick_domain(bpy.types.Operator):
    """Use the active object as the clipping domain"""
    bl_idname = "fk_foam.pick_domain"
    bl_label = "用选中物体"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = context.active_object
        if ob is None or ob.type != 'MESH':
            self.report({'ERROR'}, "select a mesh object first")
            return {'CANCELLED'}
        context.scene.fk_domain_obj = ob
        return {'FINISHED'}


class FK_OT_make_domain(bpy.types.Operator):
    """Create a primitive domain (cube / sphere / cylinder)"""
    bl_idname = "fk_foam.make_domain"
    bl_label = "新建形状"
    bl_options = {'REGISTER', 'UNDO'}
    shape: StringProperty(default="CUBE")

    def execute(self, context):
        scene = context.scene
        s = 30.0
        if self.shape == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=s, location=(s * 0.5, s * 0.5, s * 0.5))
        elif self.shape == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=s * 0.45, segments=64, ring_count=32,
                                                 location=(s * 0.5, s * 0.5, s * 0.5))
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=s * 0.45, depth=s, vertices=64,
                                                location=(s * 0.5, s * 0.5, s * 0.5))
        ob = context.active_object
        ob.name = "FK_DomainSource"
        ob.display_type = 'WIRE'
        ob.hide_render = True
        scene.fk_domain_obj = ob
        return {'FINISHED'}


# ------------------------------------------------------------------- panel --
LABELS = {
    "ZH": dict(node_size="节点尺寸", node_cut="节点切角", edge_bevel="棱边倒角",
               cap_inset="端面内插", subsurf="细分层级", fill="边界填充率",
               shape="填充形状", pick="用选中物体", cube="立方体",
               sphere="球", cyl="圆柱", cells="源边图晶胞", rebuild="重建",
               build="生成泡沫骨架"),
    "EN": dict(node_size="Node size", node_cut="Node chamfer", edge_bevel="Edge bevel",
               cap_inset="End inset", subsurf="Subdivision", fill="Border fill ratio",
               shape="Shape", pick="Use selected", cube="Cube",
               sphere="Sphere", cyl="Cylinder", cells="Source cells", rebuild="Rebuild",
               build="Build foam skeleton"),
}


TRANSLATIONS = {
    "zh_HANS": {
        "*": {
            "FK Foam Skeleton": "FK 泡沫金属骨架",
            "Node size": "节点尺寸",
            "Node chamfer": "节点切角",
            "Edge bevel": "棱边倒角",
            "End inset": "端面内插",
            "Subdivision": "细分层级",
        }
    }
}


def ui_lang():
    """'ZH' when Blender's interface is Chinese, otherwise 'EN' (read only)."""
    try:
        lang = bpy.context.preferences.view.language
    except Exception:
        lang = "en_US"
    return "ZH" if str(lang).lower().startswith("zh") else "EN"


def L(scene, key):
    return LABELS[ui_lang()][key]


class FK_PT_panel(bpy.types.Panel):
    bl_label = "FK Foam Skeleton"
    bl_idname = "FK_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FK Foam"

    def draw(self, context):
        scene = context.scene
        col = self.layout.column(align=True)
        col.separator()
        col.prop(scene, "fk_node_size", text=L(scene, "node_size"))
        col.prop(scene, "fk_bevel", text=L(scene, "node_cut"))
        col.separator()
        col.prop(scene, "fk_edge_bevel", text=L(scene, "edge_bevel"))
        col.prop(scene, "fk_cap_inset", text=L(scene, "cap_inset"))
        col.separator()
        col.prop(scene, "fk_subsurf", text=L(scene, "subsurf"))
        col.prop(scene, "fk_cell_keep", text=L(scene, "fill"))

        box = self.layout.box()
        box.label(text=L(scene, "shape"))
        box.prop(scene, "fk_domain_obj", text="")
        row = box.row(align=True)
        row.operator("fk_foam.pick_domain", text=L(scene, "pick"))
        row = box.row(align=True)
        row.operator("fk_foam.make_domain", text=L(scene, "cube")).shape = "CUBE"
        row.operator("fk_foam.make_domain", text=L(scene, "sphere")).shape = "SPHERE"
        row.operator("fk_foam.make_domain", text=L(scene, "cyl")).shape = "CYLINDER"

        self.layout.separator()
        row = self.layout.row(align=True)
        row.prop(scene, "fk_cells", text=L(scene, "cells"))
        row.operator("fk_foam.make_source", text=L(scene, "rebuild"))
        self.layout.separator()
        self.layout.operator("fk_foam.build", text=L(scene, "build"), icon='MESH_ICOSPHERE')


# ---------------------------------------------------------------- register --
CLASSES = (FK_OT_build, FK_OT_pick_domain, FK_OT_make_domain, FK_OT_source, FK_PT_panel)


def _live(self, context):
    try:
        build(context)
    except Exception:
        traceback.print_exc()


def register():
    try:
        bpy.app.translations.register(__name__, TRANSLATIONS)
    except Exception:
        pass
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:
            pass

    Scene = bpy.types.Scene
    Scene.fk_node_size = FloatProperty(name="节点尺寸 / Node size", default=1.2, min=0.05, max=2.0, update=_live)
    Scene.fk_bevel = FloatProperty(name="节点切角 / Node chamfer", default=0.8, min=0.01, max=1.2, update=_live)
    Scene.fk_edge_bevel = FloatProperty(name="棱边倒角 / Edge bevel", default=0.1, min=0.0, max=0.4, update=_live)
    Scene.fk_cap_inset = FloatProperty(name="端面内插 / End inset", default=0.05, min=0.0, max=0.4, update=_live)
    Scene.fk_subsurf = IntProperty(name="细分层级 / Subdivision", default=2, min=0, max=4, update=_live)
    Scene.fk_domain_obj = PointerProperty(name="Domain", type=bpy.types.Object,
                                          poll=lambda s, o: o.type == 'MESH')
    Scene.fk_cells = IntProperty(name="源边图晶胞 / Cells", default=3, min=1, max=8)
    Scene.fk_cell_keep = FloatProperty(name="边界填充率 / Border fill", default=0.6,
                                       min=0.3, max=1.0, update=_live)


def unregister():
    try:
        bpy.app.translations.unregister(__name__)
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
