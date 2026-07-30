'''
SimBEV2X: Scalable Vehicle-to-Everything Data Generation Tool for Multi-Task
Cooperative Perception

Copyright © 2026 Goodarz Mehr

Main package exports for SimBEV2X.
'''

__version__ = '1.0.0'
__author__ = 'Goodarz Mehr'
__email__ = 'goodarzm@vt.edu'

# Core simulation components
from .carla_core import CarlaCoreV2X
from .rsu_manager import RSUManagerV2X
from .world_manager import WorldManagerV2X
from .sensor_manager import SensorManagerV2X
from .vehicle_manager import VehicleManagerV2X
from .ground_truth_manager import GTManagerV2X
from .scenario_manager import ScenarioManagerV2X

# Sensor classes
from .sensors import (
    RGBCameraV2X,
    SemanticCameraV2X, 
    InstanceCameraV2X,
    DepthCameraV2X,
    FlowCameraV2X,
    LidarV2X,
    SemanticLidarV2X,
    RadarV2X,
    GNSSV2X,
    IMUV2X
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Core components
    'CarlaCoreV2X',
    'GTManagerV2X',
    'RSUManagerV2X',
    'WorldManagerV2X',
    'SensorManagerV2X', 
    'VehicleManagerV2X',
    'ScenarioManagerV2X',
    
    # Sensors
    'RGBCameraV2X',
    'SemanticCameraV2X',
    'InstanceCameraV2X',
    'DepthCameraV2X',
    'FlowCameraV2X',
    'LidarV2X',
    'SemanticLidarV2X',
    'RadarV2X',
    'GNSSV2X',
    'IMUV2X',

    # Utilities
    'utils',
]
