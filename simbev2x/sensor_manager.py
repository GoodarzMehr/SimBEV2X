# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that collects data from all sensors on a vehicle and renders or saves
them.
'''

import carla

import numpy as np

from scipy.spatial.transform import Rotation as R

from simbev.sensor_manager import SensorManager

class SensorManagerV2X(SensorManager):
    '''
    Sensor Manager class that manages data collection.

    Args:
        config: SimBEV configuration.
        entity: entity the Sensor Manager belongs to.
        idx: index of the entity.
        entity_type: type of the entity ('vehicle' or 'rsu').
    '''
    def __init__(self, config: dict, entity: carla.Actor, idx: int, entity_type: str = 'vehicle'):
        super().__init__(config, entity)

        self._idx = idx
        self._entity_type = entity_type

        if entity_type == 'rsu':
            self.sensor_list = {
                'rgb_camera': [],
                'semantic_camera': [],
                'instance_camera': [],
                'depth_camera': [],
                'flow_camera': [],
                'lidar': [],
                'semantic_lidar': [],
                'radar': [],
                'semantic_bev_camera': [],
                'voxel_detector': []
            }
            
            self._name_list = {
                'camera': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
                'radar': ['RAD_FRONT'],
                'bev_camera': ['TOP_VIEW', 'BOTTOM_VIEW'],
                'voxel_detector': ['3D Ground Truth']
            }

            self._other_sensor_abbrevs = {
                'lidar': 'LIDAR',
                'semantic_lidar': 'SEG-LIDAR',
                'voxel_detector': 'VOXEL-GRID'
            }
    
    def render(self):
        '''Render sensor data.'''
        name = 'Vehicle' if self._entity_type == 'vehicle' else 'RSU'
        
        for type, abbrev in self._camera_type_abbrevs.items():
            for camera, window_name in zip(self.sensor_list[f'{type}_camera'], self._name_list['camera']):
                camera.render(f'{name} {self._idx} {abbrev}-{window_name}')
        
        for type in ['lidar', 'semantic_lidar']:
            for sensor in self.sensor_list[type]:
                sensor.render(f'{name} {self._idx} {self._other_sensor_abbrevs[type]}')
        
        for radar, window_name in zip(self.sensor_list['radar'], self._name_list['radar']):
            radar.render(f'{name} {self._idx} {window_name}')
        
        for voxel_detector, window_name in zip(self.sensor_list['voxel_detector'], self._name_list['voxel_detector']):
            voxel_detector.render(f'{name} {self._idx} {window_name}')

        if self._config['render_bev_camera_images']:
            for semantic_bev_camera, window_name in zip(
                self.sensor_list['semantic_bev_camera'],
                self._name_list['bev_camera']
            ):
                semantic_bev_camera.render(f'{name} {self._idx} {window_name}')
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save sensor data.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        # Submit all I/O operations asynchronously.
        for key in self.sensor_list:
            if key in ['rgb_camera', 'semantic_camera', 'instance_camera', 'depth_camera', 'flow_camera']:
                for camera, camera_name in zip(self.sensor_list[key], self._name_list['camera']):
                    self._io_futures.append(
                        self._io_executor.submit(
                            camera.save,
                            camera_name,
                            path,
                            scene,
                            frame,
                            self._idx,
                            self._entity_type
                        )
                    )
            elif key in ['radar']:
                for radar, radar_name in zip(self.sensor_list[key], self._name_list['radar']):
                    self._io_futures.append(
                        self._io_executor.submit(
                            radar.save,
                            radar_name,
                            path,
                            scene,
                            frame,
                            self._idx,
                            self._entity_type
                        )
                    )
            elif key in ['lidar', 'semantic_lidar', 'gnss', 'imu', 'voxel_detector']:
                for sensor in self.sensor_list[key]:
                    self._io_futures.append(
                        self._io_executor.submit(sensor.save, path, scene, frame, self._idx, self._entity_type)
                    )
        
        entity_data = {}

        ego_transform = self._vehicle.get_transform()

        entity_data['ego2global_translation'] = [ego_transform.location.x,
                                                 -ego_transform.location.y,
                                                 ego_transform.location.z]
        entity_data['ego2global_rotation'] = np.roll(
            R.from_euler(
                'xyz',
                [ego_transform.rotation.roll, -ego_transform.rotation.pitch, -ego_transform.rotation.yaw],
                degrees=True
            ).as_quat(),
            1
        ).tolist()
        
        entity_data['timestamp'] = round(self._timer.time() * 10e6)

        modifier = 'rsu_' if self._entity_type == 'rsu' else ''

        for camera_name in self._name_list['camera']:
            for type, abbrev in self._camera_type_abbrevs.items():
                if self._config[f'use_{modifier}{type}_camera']:
                    entity_data[f'{abbrev}-{camera_name}'] = f'{path}/simbev2x/sweeps/{self._entity_type}-{self._idx}/{abbrev}-{camera_name}' \
                        f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-{abbrev}-{camera_name}.' + \
                            ('jpg' if type == 'rgb' else 'png' if type in ['semantic', 'instance', 'depth'] else 'npz')

        if self._config[f'use_{modifier}radar']:
            for radar_name in self._name_list['radar']:
                entity_data[f'{radar_name}'] = f'{path}/simbev2x/sweeps/{self._entity_type}-{self._idx}/{radar_name}' \
                    f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-{radar_name}.npz'
        
        for type, abbrev in self._other_sensor_abbrevs.items():
            if self._config[f'use_{modifier}{type}']:
                entity_data[f'{abbrev}'] = f'{path}/simbev2x/sweeps/{self._entity_type}-{self._idx}/{abbrev}' \
                    f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-{abbrev}.' + \
                        ('npz' if type in ['lidar', 'semantic_lidar', 'voxel_detector'] else 'bin')
            
            if type == 'voxel_detector':
                entity_data['VOXEL-GRID-FILLED'] = f'{path}/simbev2x/sweeps/{self._entity_type}-{self._idx}/VOXEL-GRID-FILLED' \
                    f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-VOXEL-GRID-FILLED.npz'
        
        entity_data['GT_SEG'] = f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/seg' \
            f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_SEG.npz'
        entity_data['GT_SEG_VIZ'] = f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/seg_viz' \
            f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_SEG_VIZ.jpg'
        entity_data['GT_DET'] = f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/det' \
            f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_DET.bin'
        entity_data['HD_MAP'] = f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/hd_map' \
            f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-HD_MAP.json'

        entity_data['scene'] = scene
        entity_data['frame'] = frame

        self._data.append(entity_data)
