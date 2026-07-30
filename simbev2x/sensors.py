# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
SimBEV2X perception and navigation sensors.
'''

import cv2
import carla

import numpy as np

from matplotlib import colormaps as cm

from simbev.sensors import (
    RGBCamera,
    SemanticCamera,
    InstanceCamera,
    DepthCamera,
    FlowCamera,
    Lidar,
    SemanticLidar,
    Radar,
    GNSS,
    IMU,
    VoxelDetector
)


RANGE = np.linspace(0.0, 1.0, 256)

RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]


class RGBCameraV2X(RGBCamera):
    '''
    RGB camera class that manages the creation and data acquisition of RGB
    cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)

    def save(self, camera_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save RGB image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        cv2.imwrite(
            f'{path}/simbev2x/sweeps/{entity}-{index}/RGB-{camera_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-RGB-{camera_name}.jpg',
            self._save_queue.get(True, 10.0),
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )


class SemanticCameraV2X(SemanticCamera):
    '''
    Semantic segmentation camera class that manages the creation and data
    acquisition of semantic segmentation cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save semantic segmentation image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        cv2.imwrite(
            f'{path}/simbev2x/sweeps/{entity}-{index}/SEG-{camera_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-SEG-{camera_name}.png',
            self._save_queue.get(True, 10.0)
        )


class InstanceCameraV2X(InstanceCamera):
    '''
    Instance segmentation camera class that manages the creation and data
    acquisition of instance segmentation cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save instance segmentation image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        cv2.imwrite(
            f'{path}/simbev2x/sweeps/{entity}-{index}/IST-{camera_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-IST-{camera_name}.png',
            self._save_queue.get(True, 10.0)
        )


class DepthCameraV2X(DepthCamera):
    '''
    Depth camera class that manages the creation and data acquisition of depth
    cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save depth image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        cv2.imwrite(
            f'{path}/simbev2x/sweeps/{entity}-{index}/DPT-{camera_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-DPT-{camera_name}.png',
            self._save_queue.get(True, 10.0)
        )


class FlowCameraV2X(FlowCamera):
    '''
    Optical flow camera class that manages the creation and data acquisition
    of optical flow cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save flow image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/FLW-{camera_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-FLW-{camera_name}.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


class LidarV2X(Lidar):
    '''
    Lidar class that manages the creation and data acquisition of lidars.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the lidar belongs to.
        transform: the lidar's transform relative to what it is attached to.
        attached: CARLA object the lidar is attached to.
        channels: number of lidar channels (beams).
        range: maximum range of the lidar.
        options: dictionary of lidar options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        channels: int,
        range: float,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)
    
    def render(self, window_name: str = 'Lidar Point Cloud'):
        '''
        Render point cloud.
        
        Args:
            window_name: window name of the point cloud visualizer.
        '''
        if self._frame == 0:
            self._create_visualizer(window_name=window_name)

        # Generate point cloud colors based on intensity values.
        distance = np.linalg.norm(self._points, axis=1)
        distance_log = np.log(distance)
        distance_log_normalized = (
            distance_log - distance_log.min()
        ) / (
            distance_log.max() - distance_log.min() + 1e-6
        )
        intensity_color = np.c_[
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 2])
        ]
        
        self._draw_points(intensity_color)
    
    def save(self, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save point cloud to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/LIDAR' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-LIDAR.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


class SemanticLidarV2X(SemanticLidar):
    '''
    Semantic lidar class that manages the creation and data acquisition of
    semantic lidars.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the lidar belongs to.
        transform: the lidar's transform relative to what it is attached to.
        attached: CARLA object the lidar is attached to.
        channels: number of lidar channels (beams).
        range: maximum range of the lidar.
        options: dictionary of lidar options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        channels: int,
        range: float,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)
    
    def render(self, window_name: str ='Semantic Lidar Point Cloud'):
        '''Render point cloud.'''
        if self._frame == 0:
            self._create_visualizer(window_name=window_name)

        self._draw_points(self._label_color)
    
    def save(self, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save point cloud to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/SEG-LIDAR' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-SEG-LIDAR.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


class RadarV2X(Radar):
    '''
    Radar class that manages the creation and data acquisition of radars.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the radar belongs to.
        transform: the radar's transform relative to what it is attached to.
        attached: CARLA object the radar is attached to.
        range: maximum range of the radar.
        hfov: horizontal field of view of the radar.
        vfov: vertical field of view of the radar.
        options: dictionary of radar options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        range: float,
        hfov: float,
        vfov: float,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, range, hfov, vfov, options)
    
    def save(self, radar_name: str, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save point cloud to file.

        Args:
            radar_name: name of the radar.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/{radar_name}' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-{radar_name}.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


class GNSSV2X(GNSS):
    '''
    GNSS class that manages the creation and data acquisition of GNSS sensors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the GNSS belongs to.
        transform: the GNSS' transform relative to what it is attached to.
        attached: CARLA object the GNSS is attached to.
        options: dictionary of GNSS options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, options)
    
    def save(self, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save GNSS data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/GNSS' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-GNSS.bin',
            'wb'
        ) as f:
            np.save(f, self._save_queue.get(True, 10.0))


class IMUV2X(IMU):
    '''
    IMU class that manages the creation and data acquisition of IMU sensors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManagerV2X instance that the IMU belongs to.
        transform: the IMU's transform relative to what it is attached to.
        attached: CARLA object the IMU is attached to.
        options: dictionary of IMU options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, options)
    
    def save(self, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save IMU data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            index: index of the entity.
            entity: type of the entity ('vehicle' or 'rsu').
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/IMU' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-IMU.bin',
            'wb'
        ) as f:
            np.save(f, self._save_queue.get(True, 10.0))

class VoxelDetectorV2X(VoxelDetector):
    '''
    Voxel detector class that manages the creation and data acquisition of
    voxel detectors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the voxel detector belongs
            to.
        transform: the voxel detector's transform relative to what it is
            attached to.
        attached: CARLA object the voxel detector is attached to.
        range: maximum range of the detection area.
        voxel_size: size of each voxel.
        upper_limit: upper limit of the detection area.
        lower_limit: lower limit of the detection area.
        options: dictionary of voxel detector options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        range: float,
        voxel_size: float,
        upper_limit: float,
        lower_limit: float,
        options: dict
    ):
        super().__init__(
            world,
            sensor_manager,
            transform,
            attached,
            range,
            voxel_size,
            upper_limit,
            lower_limit,
            options
        )
    
    def save(self, path: str, scene: int, frame: int, index: int, entity: str = 'vehicle'):
        '''
        Save voxel grid to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(
            f'{path}/simbev2x/sweeps/{entity}-{index}/VOXEL-GRID/' \
                f'SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{entity}-{index:04d}-VOXEL-GRID.npz',
                'wb'
        ) as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))