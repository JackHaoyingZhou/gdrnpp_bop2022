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
bop_renderer_path = '/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan/bop_renderer/'
sys.path.append(bop_renderer_path)
import bop_renderer #Needs to be here or a import error occurs. Needs investigation 
