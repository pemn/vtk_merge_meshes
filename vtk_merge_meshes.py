#!python
# create a voxelized merge of one or more solids
# v1.0 2022/02 paulo.ernesto

'''
usage: $0 mode%surface,solid input_files#input_path*vtk,obj,msh operation=min,max,order cell_size=10 output_path*vtk,obj,msh display@
'''
import sys, os.path
import numpy as np
import pandas as pd
import pyvista as pv

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, commalist, pd_load_dataframe, pd_save_dataframe,log

from pd_vtk import pv_read, pv_save, vtk_plot_meshes, vtk_Voxel, vtk_meshes_bb

def vtk_merge_surface(meshes, operation, cell_size):
  bb = vtk_meshes_bb(meshes)

  grid = vtk_Voxel.from_bb(bb, cell_size, 2)

  gz = np.full(grid.n_cells, np.nan)

  for mesh in meshes:
    fn = None
    if operation != 'order':
      fn = eval('np.' + operation)
    mz = grid.get_elevation(mesh, fn)

    gz = np.where(np.isnan(gz), mz, gz)
    if fn is not None:
      gz = np.apply_along_axis(fn, 0, [gz, mz])

  grid.cell_data['Elevation'] = gz
  mesh = grid.threshold(None, 'Elevation').ctp()
  mesh.warp_by_scalar('Elevation', inplace=True)

  return mesh

def vtk_merge_solid(meshes, operation, cell_size):

  bb = vtk_meshes_bb(meshes, cell_size)

  grid = vtk_Voxel.from_bb(bb, cell_size, 3)

  gz = np.full(grid.n_cells, np.nan)
  cells = pv.PolyData(grid.cell_centers().points)

  for mesh in meshes:
    # select_enclosed_points(surface, tolerance=0.001, inside_out=False, check_surface=True, progress_bar=False)
    mz = cells.select_enclosed_points(mesh, check_surface=False)
    if operation == 'min':
      gz = np.nanmin([gz, mz['SelectedPoints']], 0)
    else:
      gz = np.nanmax([gz, mz['SelectedPoints']], 0)

  grid.cell_data['SelectedPoints'] = gz
  # contour(isosurfaces=10, scalars=None, compute_normals=False, compute_gradients=False, compute_scalars=True, rng=None, preference='point', method='contour', progress_bar=False)

  return grid.threshold(0.5).extract_surface()


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
  for input_path in input_files:
    if not os.path.exists(input_path):
      print(input_path,"not found")
      continue
    mesh = pv_read(input_path)
    meshes.append(mesh)

  mesh = None
  if mode == 'surface':
    mesh = vtk_merge_surface(meshes, operation, cell_size)
  if mode == 'solid':
    mesh = vtk_merge_solid(meshes, operation, cell_size)

  if output:
    pv_save(mesh, output)
  if int(display):
    vtk_plot_meshes([mesh] + meshes)


main = vtk_merge_meshes

if __name__=="__main__":
  usage_gui(__doc__)
