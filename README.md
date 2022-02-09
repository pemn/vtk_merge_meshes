# Description
This package leverates funcionality from multiple mature python modules in a easy to use tool to perform common operations in solids and surfaces.  
Common use cases are:
 - Repair low quality meshes using voxelization.
 - Accumulate multiple surfaces into a single result.
 - Performe boolean AND/OR operations in solid meshes
 - Insert smaller surface meshes inside wider surfaces
 - Convert meshes between supported file formats

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
![winpython1](assets/winpython1.png?raw=true)
### Install vtk_reserves
Download this complete repository as a zip and extract to a local folder.  
The folder must have execute permissions. On Windows this means a folder outside the user directories, because locations such as Downloads, Documents, Desktop, etc do not allow .cmd files to run.  
![winpython2](assets/winpython2.png?raw=true)
## Run
The simplest way to run is to execute (double click on windows explorer) the supplied vtk_reserves.cmd file. This batch script shoud detect a WinPython distribution and use it automatically to call the main py file (vtk_reserves.py).  For other distributions, manually call the main script by using the following command in the distro eqivalent of  **Python Command Prompt** (not any kind of Python Interpreter!):  
`python vtk_merge_meshes.py`  
Either way the user interface should appear.


## screenshots
![screenshot1](assets/screenshot1.png?raw=true)  

![screenshot2](assets/screenshot2.png?raw=true)  

![screenshot3](assets/screenshot3.png?raw=true)  

![screenshot4](assets/screenshot4.png?raw=true)  

![screenshot5](assets/screenshot5.png?raw=true)  

