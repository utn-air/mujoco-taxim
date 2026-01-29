import numpy as np

objPaths = [
    "/home/sbien/Documents/Development/Vision2Touch/Taxim/data/objects/cylinder6.ply",
    "/home/sbien/Documents/Development/Vision2Touch/Taxim/data/objects/square.ply",
    "/home/sbien/Documents/Development/Vision2Touch/Taxim/data/objects/plier.ply"
]
for objPath in objPaths:
    f = open(objPath)
    lines = f.readlines()
    verts_num = int(lines[3].split(' ')[-1])
    verts_lines = lines[10:10 + verts_num]
    vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
    print(f"{np.max(vertices, axis=0)} - {np.min(vertices, axis=0)}")