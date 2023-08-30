'''
File structure

gdrn root/
├── scripts/
├── core/
└── lib/
    └── gdrn_simple/
        └── __init__.py (this file)
'''

# Add access to all modules in the project
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
print(root_path)
sys.path.insert(0, str(root_path))

# BOP renderer
from pathlib import Path
bop_renderer_path = Path(__file__).resolve().parent.parent.parent / 'bop_renderer'
if len(list(bop_renderer_path.glob("*.so"))) == 0: 
    print(f"BOP renderer not found in {bop_renderer_path}")
    print("Make sure to build the BOP renderer and to soft link the folder containing \
          the shared object file to the root of the project")

# Hardcoded path
# bop_renderer_path =  '/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan/bop_renderer/'
sys.path.append(str(bop_renderer_path))

import bop_renderer #Needs to be here or a import error occurs. Needs investigation 
