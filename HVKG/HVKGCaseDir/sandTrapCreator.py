#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.12.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook

notebook = salome_notebook.NoteBook()
sys.path.insert(0, r"/home/bm424/Projects/sandTrapShapeOptServer")

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Vertex_1 = geompy.MakeVertex(0, 0.5, 0)
Vertex_2 = geompy.MakeVertex(0, 0.5, 1.54)
Vertex_3 = geompy.MakeVertex(23.5, 0.5, 1.54)
geompy.addToStudy(O, "O")
geompy.addToStudy(OX, "OX")
geompy.addToStudy(OY, "OY")
geompy.addToStudy(OZ, "OZ")
geompy.addToStudy(Vertex_1, "Vertex_1")
geompy.addToStudy(Vertex_2, "Vertex_2")
geompy.addToStudy(Vertex_3, "Vertex_3")


tankBottomSpline = "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/spline.txt"

with open(tankBottomSpline, "r") as file:
    splineVertices = file.readlines()

tankBottomVertices = []
i = 0
for entry in splineVertices:
    coords = entry.strip().split()  # Splitting by whitespace
    x = float(coords[0])
    y = float(coords[1])
    vertex = geompy.MakeVertex(x, 0.5, y)  # Create a vertex at (x, y, 0)
    tankBottomVertices.append(vertex)
    geompy.addToStudy(vertex, f"tankBottomVertex_{i}")
    i += 1

bottomEdges = []

for i in range(len(tankBottomVertices) - 1):
    edge = geompy.MakeEdge(tankBottomVertices[i], tankBottomVertices[i + 1])
    bottomEdges.append(edge)
    geompy.addToStudy(edge, f"bottomEdge_{i}")

edge_1 = geompy.MakeEdge(Vertex_1, tankBottomVertices[0])
edge_2 = geompy.MakeEdge(Vertex_1, Vertex_2)
edge_3 = geompy.MakeEdge(Vertex_2, Vertex_3)
edge_4 = geompy.MakeEdge(Vertex_3, tankBottomVertices[-1])

geompy.addToStudy(edge_1, "edge_1")
geompy.addToStudy(edge_2, "edge_2")
geompy.addToStudy(edge_3, "edge_3")
geompy.addToStudy(edge_4, "edge_4")

Wire_1 = geompy.MakeWire([edge_1, edge_2, edge_3, edge_4], 1e-07)

bottomWire = geompy.MakeWire(bottomEdges)
geompy.addToStudy(bottomWire, "bottomWire")

Wire_3 = geompy.MakeWire([bottomWire, Wire_1], 1e-07)
Face_1 = geompy.MakeFaceWires([Wire_3], 1)


Extrusion_4_step_1 = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/Extrusion_4.step",
    False,
    True,
)
Extrusion_2_step_1 = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/Extrusion_2.step",
    False,
    True,
)
Extrusion_3_step_1 = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/Extrusion_3.step",
    False,
    True,
)
Extrusion_1 = geompy.MakePrismVecH2Ways(Face_1, OY, 0.5)
Cut_1 = geompy.MakeCutList(
    Extrusion_1, [Extrusion_4_step_1, Extrusion_2_step_1, Extrusion_3_step_1], True
)

geompy.addToStudy(Wire_1, "Wire_1")
geompy.addToStudy(Wire_3, "Wire_3")
geompy.addToStudy(Face_1, "Face_1")
geompy.addToStudy(Extrusion_4_step_1, "Extrusion_4.step_1")
geompy.addToStudy(Extrusion_2_step_1, "Translation_3.step_1")
geompy.addToStudy(Extrusion_3_step_1, "Extrusion_3.step_1")
geompy.addToStudy(Extrusion_1, "Extrusion_1")
geompy.addToStudy(Cut_1, "Cut_1")


inletStep = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/inlet.step",
    False,
    True,
)
freeSurfaceStep = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/free_surface.step",
    False,
    True,
)
outletStep = geompy.ImportSTEP(
    "/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/cuttingFiles/outlet.step",
    False,
    True,
)


inletFace = geompy.GetInPlace(Cut_1, inletStep, True)
inletSubShape = geompy.SubShapeAll(inletFace, geompy.ShapeType["FACE"])
inlet = geompy.CreateGroup(Cut_1, geompy.ShapeType["FACE"])

inletID = [geompy.GetSubShapeID(Cut_1, face) for face in inletSubShape]

geompy.UnionIDs(inlet, inletID)

# ------------------------------------------#

outletFace = geompy.GetInPlace(Cut_1, outletStep, True)
outletSubShape = geompy.SubShapeAll(outletFace, geompy.ShapeType["FACE"])
outlet = geompy.CreateGroup(Cut_1, geompy.ShapeType["FACE"])

outletID = [geompy.GetSubShapeID(Cut_1, face) for face in outletSubShape]

geompy.UnionIDs(outlet, outletID)

# --------------------------------------------#

freeSurfaceFace = geompy.GetInPlace(Cut_1, freeSurfaceStep, True)
freeSurfaceSubShape = geompy.SubShapeAll(freeSurfaceFace, geompy.ShapeType["FACE"])
freeSurface = geompy.CreateGroup(Cut_1, geompy.ShapeType["FACE"])

freeSurfaceID = [geompy.GetSubShapeID(Cut_1, face) for face in freeSurfaceSubShape]

geompy.UnionIDs(freeSurface, freeSurfaceID)  #


allFaces = geompy.ExtractShapes(Cut_1, geompy.ShapeType["FACE"], True)

walls = geompy.CreateGroup(Cut_1, geompy.ShapeType["FACE"])

geompy.UnionList(walls, allFaces)

geompy.addToStudyInFather(Cut_1, outlet, "outlet")
geompy.addToStudyInFather(Cut_1, inlet, "inlet")
geompy.addToStudyInFather(Cut_1, walls, "walls")
geompy.addToStudyInFather(Cut_1, freeSurface, "free_surface")


###
### SMESH component
###

import SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
# smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
# multiples meshes built in parallel, complex and numerous mesh edition (performance)

NETGEN_2D_Parameters_1 = smesh.CreateHypothesisByAverageLength(
    "NETGEN_Parameters_2D", "NETGENEngine", 0.05, 0
)


wholeGeo = smesh.Mesh(Cut_1, "wholeGeo")
status = wholeGeo.AddHypothesis(NETGEN_2D_Parameters_1)
NETGEN_1D_2D_4 = wholeGeo.Triangle(algo=smeshBuilder.NETGEN_1D2D)
inlet_2 = wholeGeo.GroupOnGeom(inlet, "inlet", SMESH.FACE)
outlet_2 = wholeGeo.GroupOnGeom(outlet, "outlet", SMESH.FACE)
free_surface_2 = wholeGeo.GroupOnGeom(freeSurface, "free_surface", SMESH.FACE)
walls_2 = wholeGeo.GroupOnGeom(walls, "walls", SMESH.FACE)
isDone = wholeGeo.Compute()
[inlet_2, outlet_2, free_surface_2, walls_2] = wholeGeo.GetGroups()

walls = wholeGeo.GetMesh().CutListOfGroups(
    [walls_2], [outlet_2, inlet_2, free_surface_2], "walls"
)


try:
    wholeGeo.ExportSTL(
        r"/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/HVKGCaseDir/surfaceMeshSTLs/inlet.stl",
        1,
        inlet_2,
    )
    pass
except:
    print("ExportPartToSTL() failed. Invalid file name?")
try:
    wholeGeo.ExportSTL(
        r"/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/HVKGCaseDir/surfaceMeshSTLs/outlet.stl",
        1,
        outlet_2,
    )
    pass
except:
    print("ExportPartToSTL() failed. Invalid file name?")
try:
    wholeGeo.ExportSTL(
        r"/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/HVKGCaseDir/surfaceMeshSTLs/free_surface.stl",
        1,
        free_surface_2,
    )
    pass
except:
    print("ExportPartToSTL() failed. Invalid file name?")
try:
    wholeGeo.ExportSTL(
        r"/home/bm424/Projects/sandTrapOptimisationSuite/HVKG/HVKGCaseDir/surfaceMeshSTLs/walls.stl",
        1,
        walls,
    )
    pass
except:
    print("ExportPartToSTL() failed. Invalid file name?")


## Set names of Mesh objects
smesh.SetName(NETGEN_2D_Parameters_1, "NETGEN 2D Parameters_1")

smesh.SetName(wholeGeo.GetMesh(), "wholeGeo")
smesh.SetName(walls_2, "walls")
smesh.SetName(inlet_2, "inlet")
smesh.SetName(outlet_2, "outlet")
smesh.SetName(free_surface_2, "free_surface")


if salome.sg.hasDesktop():
    salome.sg.updateObjBrowser()
