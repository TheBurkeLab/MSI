# Pdocs usage
## Installation
`pip3 install pdoc3` install pdocs
## Code Preparation
In order to use pdocs, the code must be commented using docstrings of the form
`'''Docstring here'''` right under the function or class that you wish to
comment.

Additionally, the folder must contain an `__init__.py` file to show that it is
a module. The code in the folder must be free of bugs and errors. Before
running pdocs, test using the module outside the folder.
## Usage
Given this file structure run as follows:

MSI

 --> src

      --> `__init__.py`

      --> `module.py`

`export PYTHONPATH="/path_to_MSI/"` set the path to search the module from

`pdoc --html src --html-dir /path_to_documentation_folder/` create the
documentation
