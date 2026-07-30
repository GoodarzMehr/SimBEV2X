# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that spawns a data collection vehicle and manages its behavior.
'''

import time
import carla
import random
import logging

import numpy as np

from simbev.utils import kill_all_servers

from simbev.sensors import SemanticBEVCamera

try:
    from .sensors import *

    from .sensor_manager import SensorManagerV2X
    from .ground_truth_manager import GTManagerV2X

except ImportError:
    from sensors import *

    from sensor_manager import SensorManagerV2X
    from ground_truth_manager import GTManagerV2X


logger = logging.getLogger(__name__)


class RSUManagerV2X:
    '''
    The RSU Manager V2X spawns an RSU and manages its behavior.

    Args:
        config: dictionary of configuration parameters.
        world: CARLA world.
        map_name: name of the CARLA map.
        idx: index of the vehicle.
    '''
    def __init__(self, config: dict, world: carla.World, map_name: str, idx: int):
        self._config = config
        self._world = world
        self._map_name = map_name
        self._idx = idx
    
    def get_sensor_manager(self):
        '''Get the Sensor Manager.'''
        return self._sensor_manager
    
    def get_ground_truth_manager(self):
        '''Get the Ground Truth Manager.'''
        return self._ground_truth_manager
    
    def _spawn_sensors(self):
        '''Spawn the sensors attached to a data collection vehicle.'''
        logger.debug(f'Creating the Sensor Manager for RSU {self._idx}...')

        self._sensor_manager = SensorManagerV2X(self._config, self.rsu, self._idx, 'rsu')

        logger.debug(f'Sensor Manager for RSU {self._idx} created.')
        logger.debug(f'Creating the sensors for RSU {self._idx}...')
        
        # Set up camera locations.
        camera_location_front_left = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.6),
            carla.Rotation(yaw=-55.0)
        )
        camera_location_front = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.6),
            carla.Rotation(yaw=0.0)
        )
        camera_location_front_right = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.6),
            carla.Rotation(yaw=55.0)
        )

        camera_locations = [camera_location_front_left, camera_location_front, camera_location_front_right]

        # Create the cameras.
        if self._config['use_rsu_rgb_camera']:
            for location in camera_locations:
                RGBCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_camera_width'],
                    self._config['rsu_camera_height'],
                    self._config['rsu_rgb_camera_properties']
                )
        
        if self._config['use_rsu_semantic_camera']:
            for location in camera_locations:
                SemanticCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_camera_width'],
                    self._config['rsu_camera_height'],
                    self._config['rsu_semantic_camera_properties']
                )
        
        if self._config['use_rsu_instance_camera']:
            for location in camera_locations:
                InstanceCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_camera_width'],
                    self._config['rsu_camera_height'],
                    self._config['rsu_instance_camera_properties']
                )
            
        
        if self._config['use_rsu_depth_camera']:
            for location in camera_locations:
                DepthCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_camera_width'],
                    self._config['rsu_camera_height'],
                    self._config['rsu_depth_camera_properties']
                )
        
        if self._config['use_rsu_flow_camera']:
            for location in camera_locations:
                FlowCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_camera_width'],
                    self._config['rsu_camera_height'],
                    self._config['rsu_flow_camera_properties']
                )
        
        # Create a lidar.
        if self._config['use_rsu_lidar']:
            LidarV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=3.2)),
                self.rsu,
                self._config['rsu_lidar_channels'],
                self._config['rsu_lidar_range'],
                self._config['rsu_lidar_properties']
            )
        
        # Create a semantic lidar.
        if self._config['use_rsu_semantic_lidar']:
            SemanticLidarV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=3.2), carla.Rotation(yaw=0.0)),
                self.rsu,
                self._config['rsu_semantic_lidar_channels'],
                self._config['rsu_semantic_lidar_range'],
                self._config['rsu_semantic_lidar_properties']
            )
        
        radar_location_front = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.2), carla.Rotation(yaw=0.0))

        radar_locations = [radar_location_front]
        
        # Create the radars
        if self._config['use_rsu_radar']:
            for location in radar_locations:
                RadarV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.rsu,
                    self._config['rsu_radar_range'],
                    self._config['rsu_radar_horizontal_fov'],
                    self._config['rsu_radar_vertical_fov'],
                    self._config['rsu_radar_properties']
                )
        
        # Create BEV semantic cameras for obtaining the ground truth.
        bev_location_above = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=self._config['bev_camera_height']),
            carla.Rotation(pitch=-90)
        )
        bev_location_below = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=-self._config['bev_camera_height']),
            carla.Rotation(pitch=90)
        )

        bev_locations = [bev_location_above, bev_location_below]
        
        for location in bev_locations:
            SemanticBEVCamera(
                self._world,
                self._sensor_manager,
                location,
                self.rsu,
                self._config['bev_dim'],
                self._config['bev_dim'],
                self._config['bev_properties']
            )
        
        # Create voxel detector for obtaining the 3D ground truth.
        if self._config['use_rsu_voxel_detector']:
            VoxelDetectorV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=0.02)),
                self.rsu,
                self._config['rsu_voxel_detector_range'],
                self._config['rsu_voxel_size'],
                self._config['rsu_voxel_detector_upper_limit'],
                self._config['rsu_voxel_detector_lower_limit'],
                self._config['rsu_voxel_detector_properties']
            )
        
        self._world.tick()

        logger.debug(f'Sensors for RSU {self._idx} created.')
    
    def spawn_rsu(self, bp: carla.ActorBlueprint, spawn_point: carla.Transform) -> dict:
        '''
        Spawn an RSU and its sensors.
        
        Args:
            bp: pole blueprint.
            spawn_point: spawn point of the RSU.
        '''
        try:
            rsu_info = {}
            
            bev_fov = 2 * np.rad2deg(
                np.arctan(self._config['bev_dim'] * self._config['bev_res'] / (2 * self._config['bev_camera_height']))
            )

            self._config['bev_properties'] = {'fov': str(bev_fov)}
            
            # Instantiate the RSU.
            logger.debug(f'Spawning RSU {self._idx}...')
            
            self.rsu = None
            
            attempts = 0
            
            while self.rsu is None:
                self.rsu = self._world.spawn_actor(bp, spawn_point)

                if attempts > self._config['spawn_attempts']:
                    raise Exception(f'Cannot spawn RSU {self._idx} for some reason. Good bye!')
                
                attempts += 1

            self.rsu.set_simulate_physics(self._config['simulate_physics'])

            logger.debug(f'RSU {self._idx} spawned.')

            self._spawn_sensors()

            # Instantiate the Ground Truth Manager.
            logger.debug(f'Creating the Ground Truth Manager for RSU {self._idx}...')

            self._ground_truth_manager = GTManagerV2X(
                self._config,
                self._world,
                self.rsu,
                self._sensor_manager,
                self._map_name,
                self._idx,
                'rsu'
            )

            logger.debug(f'Ground Truth Manager for RSU {self._idx} created.')

            self._world.tick()

            return rsu_info

        except Exception as e:
            logger.error(f'Error while spawning RSU {self._idx}: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot spawn the RSU. Good bye!')
    
    def find_rsu(self):
        '''Find the RSU in the world.'''
        try:
            logger.debug(f'Finding RSU {self._idx} in the world...')

            actors = self._world.get_actors()

            for actor in actors:
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == f'rsu_{self._idx}':
                    self.rsu = actor

                    logger.debug(f'RSU {self._idx} found.')

                    self._spawn_sensors()

                    return
            
            raise Exception(f'RSU {self._idx} not found in the world.')

        except Exception as e:
            logger.error(f'Error while finding RSU {self._idx}: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot find RSU. Good bye!')
    
    def destroy_rsu(self):
        '''Destroy the Sensor Manager and the RSU.'''
        logger.debug(f'Destroying the Sensor Manager for RSU {self._idx}...')
        
        self._sensor_manager.destroy()

        logger.debug(f'Sensor Manager for RSU {self._idx} destroyed.')
        logger.debug(f'Destroying RSU {self._idx}...')
        
        self.rsu.destroy()

        logger.debug(f'RSU {self._idx} destroyed.')