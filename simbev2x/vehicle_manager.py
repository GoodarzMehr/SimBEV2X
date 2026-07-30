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
from simbev.vehicle_manager import VehicleManager

try:
    from .sensors import *

    from .sensor_manager import SensorManagerV2X
    from .ground_truth_manager import GTManagerV2X

except ImportError:
    from sensors import *

    from sensor_manager import SensorManagerV2X
    from ground_truth_manager import GTManagerV2X


logger = logging.getLogger(__name__)


class VehicleManagerV2X(VehicleManager):
    '''
    The Vehicle Manager V2X spawns a data collection vehicle and manages its
    behavior.

    Args:
        config: dictionary of configuration parameters.
        world: CARLA world.
        traffic_manager: CARLA traffic manager.
        map_name: name of the CARLA map.
        idx: index of the vehicle.
    '''
    def __init__(
            self,
            config: dict,
            world: carla.World,
            traffic_manager: carla.TrafficManager,
            map_name: str,
            idx: int
        ):
        super().__init__(config, world, traffic_manager, map_name)

        self._idx = idx
    
    def _spawn_sensors(self):
        '''Spawn the sensors attached to a data collection vehicle.'''
        logger.debug(f'Creating the Sensor Manager for vehicle {self._idx}...')

        self._sensor_manager = SensorManagerV2X(self._config, self.vehicle, self._idx)

        logger.debug(f'Sensor Manager for vehicle {self._idx} created.')
        logger.debug(f'Creating the sensors for vehicle {self._idx}...')
        
        # Set up camera locations.
        camera_location_front_left = carla.Transform(
            carla.Location(x=0.4, y=-0.4, z=1.6),
            carla.Rotation(yaw=-55.0)
        )
        camera_location_front = carla.Transform(
            carla.Location(x=0.6, y=0.0, z=1.6),
            carla.Rotation(yaw=0.0)
        )
        camera_location_front_right = carla.Transform(
            carla.Location(x=0.4, y=0.4, z=1.6),
            carla.Rotation(yaw=55.0)
        )
        camera_location_back_left = carla.Transform(
            carla.Location(x=0.0, y=-0.4, z=1.6),
            carla.Rotation(yaw=-110)
        )
        camera_location_back = carla.Transform(
            carla.Location(x=-1.0, y=0.0, z=1.6),
            carla.Rotation(yaw=180.0)
        )
        camera_location_back_right = carla.Transform(
            carla.Location(x=0.0, y=0.4, z=1.6),
            carla.Rotation(yaw=110.0)
        )

        camera_locations = [
            camera_location_front_left,
            camera_location_front,
            camera_location_front_right,
            camera_location_back_left,
            camera_location_back,
            camera_location_back_right
        ]

        # Create the cameras.
        if self._config['use_rgb_camera']:
            for location in camera_locations:
                RGBCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['rgb_camera_properties']
                )
        
        if self._config['use_semantic_camera']:
            for location in camera_locations:
                SemanticCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['semantic_camera_properties']
                )
        
        if self._config['use_instance_camera']:
            for location in camera_locations:
                InstanceCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['instance_camera_properties']
                )
            
        
        if self._config['use_depth_camera']:
            for location in camera_locations:
                DepthCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['depth_camera_properties']
                )
        
        if self._config['use_flow_camera']:
            for location in camera_locations:
                FlowCameraV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['flow_camera_properties']
                )
        
        # Create a lidar.
        if self._config['use_lidar']:
            LidarV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8)),
                self.vehicle,
                self._config['lidar_channels'],
                self._config['lidar_range'],
                self._config['lidar_properties']
            )
        
        # Create a semantic lidar.
        if self._config['use_semantic_lidar']:
            SemanticLidarV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8), carla.Rotation(yaw=0.0)),
                self.vehicle,
                self._config['semantic_lidar_channels'],
                self._config['semantic_lidar_range'],
                self._config['semantic_lidar_properties']
            )
        
        radar_location_left = carla.Transform(carla.Location(x=0.0, y=-1.0, z=0.6), carla.Rotation(yaw=-90.0))
        radar_location_front = carla.Transform(carla.Location(x=2.4, y=0.0, z=0.6), carla.Rotation(yaw=0.0))
        radar_location_right = carla.Transform(carla.Location(x=0.0, y=1.0, z=0.6), carla.Rotation(yaw=90.0))
        radar_location_back = carla.Transform(carla.Location(x=-2.4, y=0.0, z=0.6), carla.Rotation(yaw=180.0))

        radar_locations = [
            radar_location_left,
            radar_location_front,
            radar_location_right,
            radar_location_back,
        ]
        
        # Create the radars
        if self._config['use_radar']:
            for location in radar_locations:
                RadarV2X(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['radar_range'],
                    self._config['radar_horizontal_fov'],
                    self._config['radar_vertical_fov'],
                    self._config['radar_properties']
                )
        
        # Create a GNSS sensor.
        if self._config['use_gnss']:
            GNSSV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(),
                self.vehicle,
                self._config['gnss_properties']
            )
        
        # Create an IMU sensor.
        if self._config['use_imu']:
            IMUV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(),
                self.vehicle,
                self._config['imu_properties']
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
                self.vehicle,
                self._config['bev_dim'],
                self._config['bev_dim'],
                self._config['bev_properties']
            )
        
        # Create voxel detector for obtaining the 3D ground truth.
        if self._config['use_voxel_detector']:
            VoxelDetectorV2X(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=0.02)),
                self.vehicle,
                self._config['voxel_detector_range'],
                self._config['voxel_size'],
                self._config['voxel_detector_upper_limit'],
                self._config['voxel_detector_lower_limit'],
                self._config['voxel_detector_properties']
            )
        
        self._world.tick()

        logger.debug(f'Sensors for vehicle {self._idx} created.')
    
    def spawn_vehicle(self, bp: carla.ActorBlueprint, spawn_point: carla.Transform, tm_port: int) -> dict:
        '''
        Spawn a data collection vehicle and its sensors.
        
        Args:
            bp: ego vehicle blueprint.
            spawn_point: spawn point of the vehicle.
            tm_port: Traffic Manager port.
        '''
        try:
            vehicle_info = {}
            
            bev_fov = 2 * np.rad2deg(
                np.arctan(self._config['bev_dim'] * self._config['bev_res'] / (2 * self._config['bev_camera_height']))
            )

            self._config['bev_properties'] = {'fov': str(bev_fov)}
            
            # Instantiate the vehicle.
            logger.debug(f'Spawning vehicle {self._idx}...')
            
            self.vehicle = None

            if self._idx == 0:
                self._world.get_spectator().set_transform(
                    carla.Transform(
                        spawn_point.location + carla.Location(z=160.0),
                        carla.Rotation(pitch=-90.0, yaw=spawn_point.rotation.yaw)
                    )
                )

                for _ in range(100):
                    self._world.tick()
            
            attempts = 0
            
            while self.vehicle is None:
                self.vehicle = self._world.try_spawn_actor(bp, spawn_point)

                # Move the spawn point around a bit if the vehicle cannot be
                # spawned.
                if attempts > self._config['spawn_attempts'] - 4:
                    spawn_point.location += carla.Location(x=1.0, y=1.0, z=0.0)
                if attempts > self._config['spawn_attempts'] - 3:
                    spawn_point.location += carla.Location(x=-1.0, y=1.0, z=0.0)
                if attempts > self._config['spawn_attempts'] - 2:
                    spawn_point.location += carla.Location(x=1.0, y=-1.0, z=0.0)
                if attempts > self._config['spawn_attempts'] - 1:
                    spawn_point.location += carla.Location(x=-1.0, y=1.0, z=0.0)
                if attempts > self._config['spawn_attempts']:
                    raise Exception(f'Cannot spawn vehicle {self._idx} for some reason. Good bye!')
                
                attempts += 1

            self.vehicle.set_autopilot(True, tm_port)
            self.vehicle.set_simulate_physics(self._config['simulate_physics'])
            self.vehicle.show_debug_telemetry(self._config['show_debug_telemetry'])

            self._traffic_manager.update_vehicle_lights(self.vehicle, True)

            vehicle_info['vehicle'] = self.vehicle.type_id

            logger.debug(f'Vehicle {self._idx} spawned.')

            # Set the percentage of time the vehicle ignores traffic lights,
            # traffic signs, other vehicles, and walkers.
            logger.debug(f'Configuring behavior for vehicle {self._idx}...')

            self._traffic_manager.ignore_lights_percentage(self.vehicle, self._config['ignore_lights_percentage'])
            self._traffic_manager.ignore_signs_percentage(self.vehicle, self._config['ignore_signs_percentage'])
            self._traffic_manager.ignore_vehicles_percentage(self.vehicle, self._config['ignore_vehicles_percentage'])
            self._traffic_manager.ignore_walkers_percentage(self.vehicle, self._config['ignore_walkers_percentage'])

            # Determine whether the vehicle is reckless (ignores all traffic
            # rules).
            vehicle_info['reckless_ego'] = False
            vehicle_info['distracted_ego'] = False

            p = self._config['reckless_ego_percentage'] / 100.0
            
            if np.random.choice(2, p=[1 - p, p]):
                logger.warning(f'Vehicle {self._idx} is reckless!')

                self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                vehicle_info['reckless_ego'] = True
            else:
                p = self._config['distracted_ego_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning(f'Vehicle {self._idx} is distracted!')

                    self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)

                    vehicle_info['distracted_ego'] = True

            if 'speed_difference' not in self._config:
                self._traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))
            
            if 'distance_to_leading' not in self._config:
                self._traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(4.2, 1.0))
            
            logger.debug(f'Vehicle {self._idx} behavior configured.')

            self._spawn_sensors()

            # Instantiate the Ground Truth Manager.
            logger.debug(f'Creating the Ground Truth Manager for vehicle {self._idx}...')

            self._ground_truth_manager = GTManagerV2X(
                self._config,
                self._world,
                self.vehicle,
                self._sensor_manager,
                self._map_name,
                self._idx
            )

            logger.debug(f'Ground Truth Manager for vehicle {self._idx} created.')

            self._world.tick()

            return vehicle_info

        except Exception as e:
            logger.error(f'Error while spawning vehicle {self._idx}: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot spawn the vehicle. Good bye!')
    
    def find_vehicle(self):
        '''Find the data collection vehicle in the world.'''
        try:
            logger.debug(f'Finding vehicle {self._idx} in the world...')

            role_name = 'hero' if self._idx == 0 else f'vehicle_{self._idx}'

            actors = self._world.get_actors()

            for actor in actors:
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == role_name:
                    self.vehicle = actor

                    logger.debug(f'Vehicle {self._idx} found.')

                    self._spawn_sensors()

                    return
            
            raise Exception(f'Vehicle {self._idx} not found in the world.')

        except Exception as e:
            logger.error(f'Error while finding vehicle {self._idx}: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot find a data collection vehicle. Good bye!')
    
    def destroy_vehicle(self):
        '''Destroy the Sensor Manager and the vehicle.'''
        logger.debug(f'Destroying the Sensor Manager for vehicle {self._idx}...')

        self._sensor_manager.destroy()

        logger.debug(f'Sensor Manager for vehicle {self._idx} destroyed.')
        logger.debug(f'Destroying vehicle {self._idx}...')

        self.vehicle.destroy()

        logger.debug(f'Vehicle {self._idx} destroyed.')