# FK Foam Skeleton — Blender add-on
Generate Frank–Kasper (A15 / Weaire–Phelan) foam-metal skeletons inside an arbitrary closed shape: a cube, a sphere, a cylinder, or any mesh you already have in the scene.

No boolean, no voxel remesh — the A15 cells are filtered against the shape and the Frank–Kasper edge graph is turned into the skeleton directly.

<img width="800" height="800" alt="QQ_1794447229832" src="https://github.com/user-attachments/assets/a72f8bb9-7fda-4cef-ad9a-d9cb330ecf95" />

# Quick start
· Install — Edit ▸ Preferences ▸ Add-ons ▸ Install… → pick fk_foam_addon.zip → tick FK Foam Skeleton.

· Open the panel — in the 3D View press N and switch to the FK Foam tab.

· Pick a shape — click Cube, Sphere or Cylinder, or select any mesh in the scene and press Use selected.

· Build — press Build foam skeleton. The lattice object FK_Node_Network appears, clipped to your shape.

· Tune — drag the sliders. Every value rebuilds the model immediately; the Subdivision slider is a live modifier, so it is completely free to change.

That is the whole workflow — the panel is a single screen:

  node size        ── arm length of the tetrahedron node

  node chamfer     ── chamfer of the tetrahedron corners

  edge bevel       ── edge bevel width (1 segment)

  end inset        ── inset depth of the hexagonal end faces

  subdivision      ── Catmull-Clark levels

  border fill      ── how much of a boundary cell must be inside the shape

  shape            ── cube / sphere / cylinder / use selected

  source cells     ── A15 lattice size used to seed the graph

  [ Build foam skeleton ]

If nothing is selected as a shape, the build is refused with a warning (Please select a source shape first) instead of an error. If FK_Foam_Edges is missing it is created automatically.

# Parameters
Parameter	Default	Meaning

Node size	1.20 mm	arm length of the node tetrahedron

Node chamfer	0.80 mm	chamfer cut on the tetrahedron corners

Edge bevel	0.10 mm	bevel width of all edges except the port and end triangles

End inset	0.05 mm	inset depth of the hexagonal end faces

Subdivision	2	Catmull-Clark subdivision level

Border fill	0.60	minimum fraction of a cell's vertices that must be inside the shape

<img width="1929" height="732" alt="image" src="https://github.com/user-attachments/assets/66af6054-4932-42fb-9226-b1f05b266ae7" />


# How it works
1. Cells — the A15 (Cr₃Si) lattice, eight equal-volume Laguerre cells per cubic unit cell: two 12-faced (dodecahedra) and six 14-faced (tetrakaidecahedra).
2. Shape fit — cells are laid out over the bounding box of the target mesh and kept only when they lie inside it (vectorised parity ray test against a low-poly proxy of the mesh). Every kept cell stays a complete polyhedron, so no cut faces appear.
3. Graph — the shared edges of the kept cells form the Frank–Kasper edge graph.
4. Skeleton — a tetrahedron is placed at each 4-valent vertex, its corners are chamfered, facing triangles of neighbouring nodes are bridged with quads, and the outer triangles are extruded into legs; boundary vertices missing an arm get one, so every node is a complete tetrahedron.
5. Details — weight-limited edge bevel (segments 1) on everything except the port triangles and the end triangles; the end triangles become hexagons and are inset; finally a Subdivision modifier at level 2.

# Notes and limitations
· Cells stay complete, so a thin irregular gap remains between the outermost cells and the surface — tune it with Border fill (0.6 is the default; higher is more conservative).
· Large shapes (90 mm or more) take 10–20 s to rebuild. Set Subdivision to 0 while tuning parameters and back to 2 for the final look.
· For custom shapes, a closed mesh gives the best result; open meshes are handled but the boundary becomes ragged.
· The camera and lights are not adapted automatically for very large shapes.

# Files
fk_foam_addon/
  __init__.py    panel, operators, A15 source graph
  fk_geom.py     A15 lattice and Laguerre cells
  fk_nodes.py    tetrahedron nodes, bridges, legs, bevel, inset

Dependency: numpy only (ships with Blender). Tested on Blender 5.2, requires 4.2+.
  fk_shape.py    cell filtering inside the shape, edge graph extraction
