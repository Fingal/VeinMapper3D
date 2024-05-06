import pyopenvdb  as vdb
import numpy as np
arr = np.array([[[min(1000,max(0,k-(i+5)*(j+5))) for i in range(200)] for j in range(200)] for k in range(200)])
grid = vdb.FloatGrid()
grid.copyFromArray(arr)
grid.activeVoxelCount() == arr.size
print(grid.evalActiveVoxelBoundingBox())
vdb.write('mygrids.vdb', grids=[grid])
