'''
SimBEV2X Tools

Copyright © 2026 Goodarz Mehr

Post-processing and visualization utilities for SimBEV2X datasets.
'''

__version__ = '1.0.0'
__author__ = 'Goodarz Mehr'
__email__ = 'goodarzm@vt.edu'

# Note: Import functions only when needed to avoid triggering argparse at
# import time

# Handlers
from .visualization_handlers import *

# Interactive visualization
from .visualization_interactive import *

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'visualization_handlers',
    'visualization_interactive'
]
