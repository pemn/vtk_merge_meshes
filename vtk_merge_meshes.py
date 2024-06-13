#!python
# repair or merge multiple surface/solid meshes
# v1.0 2022/02 paulo.ernesto
'''
usage: $0 mode%surface,solid,boolean,voxel input_files#input_path*vtk,obj,msh operation=min,max,order cell_size=10 output_path*vtk,obj,msh display@
'''
'''
Copyright 2022 Vale

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*** You can contribute to the main repository at: ***

https://github.com/pemn/vtk_merge_meshes
---------------------------------
'''
import sys, os.path
import numpy as np
import pandas as pd
import pyvista as pv
from glob import glob

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, commalist, pd_load_dataframe, pd_save_dataframe,log

from pd_vtk import pv_read, pv_save, vtk_plot_meshes, vtk_Voxel, vtk_meshes_bb, mr_boolean

def vtk_merge_surface(meshes, operation, cell_size):
  bb = vtk_meshes_bb(meshes)

  grid = vtk_Voxel.from_bb(bb, cell_size, 2)

  gz = np.full(grid.n_cells, np.nan)

  for mesh in meshes:
    fn = None
    if operation != 'order':
      fn = eval('np.nan' + operation)
    mz = grid.get_elevation(mesh, fn)

    gz = np.where(np.isnan(gz), mz, gz)
    if fn is not None:
      gz = np.apply_along_axis(fn, 0, [gz, mz])

  grid.cell_data['Elevation'] = gz
  mesh = grid.threshold(None, 'Elevation').ctp()
  mesh.warp_by_scalar('Elevation', inplace=True)

  return mesh

def vtk_merge_solid(meshes, operation, cell_size, tolerance = 0.001):
  bb = vtk_meshes_bb(meshes, cell_size)
  grid = vtk_Voxel.from_bb(bb, cell_size)
  gz = None

  for mesh in meshes:
    #mesh = mesh.extract_surface()
    #if not mesh.is_all_triangles:
      # reduce chance for artifacts, see gh-1743
      # mesh.triangulate(inplace=True)

    # select_enclosed_points(surface, tolerance=0.001, inside_out=False, check_surface=True, progress_bar=False)
    ep = grid.select_enclosed_points(mesh, tolerance, check_surface=False)
    if gz is None:
      gz = ep.point_data['SelectedPoints']
    elif operation == 'min':
      gz &= ep.point_data['SelectedPoints']
    else:
      gz |= ep.point_data['SelectedPoints']

  mask = grid.extract_points(gz.view(np.bool_))
  del mask.cell_data['vtkOriginalCellIds']
  del mask.point_data['vtkOriginalPointIds']
  return mask.extract_surface(False, False)

#def vtk_merge_voxel(meshes, operation, cell_size):
  ''' WIP '''
#  bb = vtk_meshes_bb(meshes)
#  return pv.PolyData(pv.Box(np.ravel(list(zip(bb[0], bb[1])))))

def vtk_merge_meshes(mode, input_files, operation, cell_size, output, display):
  if not mode:
    mode == 'surface'
  # prevent invalid grid sizes
  if not cell_size:
    cell_size = 1
  else:
    cell_size = float(cell_size)

  if not operation:
    operation = 'min'

  input_files = commalist().parse(input_files).split()
  
  meshes = []
  for raw_path in input_files:
    for input_path in glob(raw_path):
      print(input_path)
      mesh = pv_read(input_path)
      meshes.append(mesh)

  mesh = None
  if mode == 'surface':
    mesh = vtk_merge_surface(meshes, operation, cell_size)
  if mode == 'solid':
    mesh = vtk_merge_solid(meshes, operation, cell_size)
  if mode in ['boolean', 'voxel']:
    mesh = mr_boolean(meshes, operation, mode, cell_size)
  
  if output:
    pv_save(mesh, output)

  if int(display):
    vtk_plot_meshes(mesh)

main = vtk_merge_meshes

if __name__=="__main__":
  usage_gui(__doc__)
