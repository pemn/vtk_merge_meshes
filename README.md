## 📌 Description
This package leverates funcionality from multiple mature python modules in a easy to use tool to perform common operations in solids and surfaces.  
Common use cases are:
 - Repair low quality meshes using voxelization.
 - Accumulate multiple surfaces into a single result.
 - Performe boolean AND/OR operations in solid meshes
 - Insert smaller surface meshes inside wider surfaces
 - Convert meshes between supported file formats
## 📸 Screenshot
![screenshot1](https://github.com/pemn/assets/blob/main/vtk_merge_meshes1.png?raw=true)
## Installation Instructions
### Install a Python Distribution
A python distribution such as [WinPython](https://winpython.github.io/) is required. Version must be >= 3.7. Version 3.8.x recomended.   
Do **not** use the installer from Python.org because it will lack countless required modules. Distros exist for a reason!  
After Python is installed you can test its really working by using the following command:  
`python -V`  
Only one required module is not included by default in WinPython and other mainstrean distros:  
 - pyvista  

Install using this folowing command in the **WinPython Command Prompt** (not the Python Interpreter!):  
`pip install pyvista`  
![winpython](https://github.com/pemn/assets/blob/main/winpython.png?raw=true)
### Install vtk_merge_meshes
Download this complete repository as a zip and extract to a local folder.  
The folder must have execute permissions. On Windows this means a folder outside the user directories, because locations such as Downloads, Documents, Desktop, etc do not allow .cmd files to run.  
![downloadcode](assets/downloadcode.png?raw=true)
### Run
The simplest way to run is to execute (double click on windows explorer) the supplied [script name].cmd file. This batch script shoud detect a WinPython distribution and use it automatically to call the main py file ([script name].py).  For other distributions, manually call the main script by using the following command in the distro eqivalent of  **Python Command Prompt** (not any kind of Python Interpreter!):  
`python vtk_merge_meshes.py`  
Either way the user interface should appear.

### Suported file formats
The surfaces and solids can be in those supported formats:
 - csv (ASCII, with x,y,z and faces)
 - obj (wavefront)
 - msh (leapfrog)
 - vtk (containing PolyData Mesh object types).  
## 📝 Parameters
name|optional|description
---|---|------
input_files|❎|input data in a supported file format: vtk,ply,obj,stl,msh
operation|❎|how the meshes will be combined
||min|surface minimum (cut)
||max|surface maximum (fill)
||order|surface patching, use the first information that cover each cell
||and|solid intersection usign vtk
||or|solid union using vtk
||...|other solid operations using meshlib, check module documentation
cell_size|☑️|cell size in metersfor the voxel based operation
output_path|☑️|file path to save result
display||show result in a 3d windows
## 🧊 Sample Data
There is some simple artificial dataset on the sample_data folder of this repository for testing and reference.
## 📚 Examples
### VTK surface `min`
![screenshot2](https://github.com/pemn/assets/blob/main/vtk_merge_meshes2.png?raw=true)  
### VTK solid `or`
![screenshot3](https://github.com/pemn/assets/blob/main/vtk_merge_meshes3.png?raw=true)  
### meshlib operations
![screenshot5](https://github.com/pemn/assets/blob/main/vtk_merge_meshes5.png?raw=true)  
## 🧩 Compatibility
distribution|status
---|---
![winpython_icon](https://github.com/pemn/assets/blob/main/winpython_icon.png?raw=true)|✔
![vulcan_icon](https://github.com/pemn/assets/blob/main/vulcan_icon.png?raw=true)|❌
![anaconda_icon](https://github.com/pemn/assets/blob/main/anaconda_icon.png?raw=true)|✔
## 🙋 Support
Any question or problem contact:
 - paulo.ernesto
## 💎 License
Apache 2.0
Copyright ![vale_logo_only](https://github.com/pemn/assets/blob/main/vale_logo_only_r.svg?raw=true) Vale 2024

