# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import time
import json
import threading

import numpy as np
import open3d as o3d

import open3d.visualization.gui as gui

from pyquaternion import Quaternion as Q

from .visualization_handlers import RSU_RAD_NAME

from simbev_tools.visualization_utils import get_global2sensor, transform_bbox
from simbev_tools.visualization_utils import RANGE, RAINBOW, RAD_NAME, LABEL_COLORS

from simbev_tools.visualization_interactive import VizDataLoader, InteractiveVisualizer


ENTITY_COLORS = np.array([
    [230,  25,  75], # Red
    [ 60, 180,  75], # Green
    [255, 225,  25], # Yellow
    [  67, 99, 216], # Blue
    [245, 130,  49], # Orange
    [145,  30, 180], # Purple
    [ 66, 212, 244], # Cyan
    [240,  50, 230], # Magenta
    [191, 239,  69], # Lime
    [250, 190, 212], # Pink
    [ 70, 153, 144], # Teal
    [160,  82,  45]  # Brown
]) / 255.0


class V2XVizDataLoader(VizDataLoader):
    '''
    Data loader that loads point cloud, voxel, and bounding box data for
    multiple entities (vehicles and RSUs) for interactive visualization.

    Args:
        path: root directory of the dataset.
        metadata: dataset metadata.
        ignore_valid_flag: whether to ignore the valid_flag of object bounding
            boxes.
        max_workers: number of workers for loading data in parallel.
        max_cached: maximum number of (scene, entity) pairs to keep in
            cache.
        filled_voxels: whether to use filled voxel grids.
        trim_step: step size for trimming point clouds by lidar channel.
    '''
    def __init__(
        self,
        path: str,
        metadata: dict,
        ignore_valid_flag: bool = False,
        max_workers: int = 8,
        max_cached: int = 3,
        filled_voxels: bool = False,
        trim_step: int = 1
    ):
        super().__init__(
            path,
            metadata,
            ignore_valid_flag=ignore_valid_flag,
            max_workers=max_workers,
            filled_voxels=filled_voxels,
        )

        # Cache: {cache_key: {sensor_type: [frame_data, ...]}}
        self._cache = {}
        
        # Track cache access order for LRU eviction.
        self._cache_access_order = []
        
        self.max_cached = max_cached
        self._trim_step = trim_step

    def _load_scene_structure(self) -> list:
        '''
        Load scene metadata and frame paths without loading the actual data.

        Returns:
            List of scene information dictionaries.
        '''
        scene_info = []

        for split in ['train', 'val', 'test']:
            info_path = f'{self._path}/simbev2x/infos/simbev2x_infos_{split}.json'

            if not os.path.exists(info_path):
                continue

            with open(info_path, 'r') as f:
                infos = json.load(f)

            for key, value in infos['data'].items():
                scene_number = int(key.split('_')[1])
                scene_data = value['scene_data']

                frame_count = len(scene_data[next(iter(scene_data))])

                scene_info.append({
                    'scene_number': scene_number,
                    'frame_count': frame_count,
                    'frame_paths': scene_data,
                    'split': split
                })

        scene_info.sort(key=lambda x: x['scene_number'])

        return scene_info

    def is_loaded(self, scene: int, entity: str = None) -> bool:
        '''
        Check if the (scene, entity) pair data is loaded in cache.

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2'). If None, checks
                if the entire scene is loaded.

        Returns:
            True if (scene, entity) is loaded in cache, False otherwise.
        '''
        if entity is None:
            return scene in self._cache
        
        return (scene, entity) in self._cache

    def _mark_accessed(self, scene: int, entity: str = None):
        '''
        Mark a (scene, entity) pair as recently accessed (move to the end of
        the LRU list).

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
        '''
        key = (scene, entity) if entity is not None else scene

        if key in self._cache_access_order:
            self._cache_access_order.remove(key)

        self._cache_access_order.append(key)

    def _evict_oldest(self):
        '''Evict the least recently used item from cache.'''
        if self._cache_access_order:
            # Get the oldest item (first in the list).
            oldest = self._cache_access_order[0]
            
            self.unload_entity(*oldest) if isinstance(oldest, tuple) else self.unload_entity(oldest)
    
    def clear_cache(self):
        '''Evict all cached scenes immediately to free up memory.'''
        self._cache.clear()
        self._cache_access_order.clear()

    def load_entity(self, scene: int, entity: str, progress_callback=None) -> bool:
        '''
        Load the entire data for an entity (all frames, all sensors) into
        cache. Automatically manage cache size by evicting the least recently
        used (LRU) scenes.

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
            progress_callback: optional callback(current, total, message) for
                progress updates while loading.

        Returns:
            True if loaded successfully, False otherwise.
        '''
        scene_info = self._scene_info[scene]

        if entity not in scene_info['frame_paths']:
            print(f'{entity} not present in scene {self.get_scene_number(scene):04d}.')

            if progress_callback:
                progress_callback(1, 1, 'Entity not present in this scene.')

            return False
        
        cache_key = (scene, entity)

        if cache_key in self._cache:
            self._mark_accessed(scene, entity)

            if progress_callback:
                progress_callback(1, 1, 'Scene/entity already loaded.')

            return True

        if len(self._cache) >= self.max_cached:
            self._evict_oldest()

        print(f'\nLoading scene {self.get_scene_number(scene):04d} [{entity}]...')

        start = time.perf_counter()
        
        frame_count = self.get_frame_count(scene)
        
        sensor_types = ['lidar', 'semantic-lidar', 'radar']

        # Initialize the cache for this entity.
        entity_data = {sensor_type: [None] * frame_count for sensor_type in sensor_types}
        
        # Create tasks for parallel loading.
        tasks = []
        
        total_tasks = frame_count * len(sensor_types)

        for sensor_type in sensor_types:
            for frame in range(frame_count):
                task = self._executor.submit(self._load_single_frame, scene, entity, frame, sensor_type)
                
                tasks.append((task, frame, sensor_type))

        # Wait for all tasks to complete and update the cache.
        completed = 0

        for task, frame, sensor_type in tasks:
            try:
                frame_data = task.result()
                
                entity_data[sensor_type][frame] = frame_data
                
                completed += 1

                if progress_callback and completed % 10 == 0:
                    progress_callback(completed, total_tasks, f'Loading: {completed}/{total_tasks}')

            except Exception as e:
                print(f'Error while loading scene {scene} [{entity}], frame {frame}, sensor {sensor_type}: {e}')
                
                entity_data[sensor_type][frame] = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': []}

        # Store in cache.
        self._cache[cache_key] = entity_data
        
        self._mark_accessed(scene, entity)

        elapsed = time.perf_counter() - start
        
        print(f'Scene {self.get_scene_number(scene):04d} [{entity}] loaded in {elapsed:.2f} s.')

        if progress_callback:
            progress_callback(total_tasks, total_tasks, 'Scene loaded.')

        return True

    def load_scene(self, scene: int, progress_callback=None) -> bool:
        '''
        Load the entire data for all entities in a scene (all frames,
        all sensors) into cache. Automatically manage cache size by evicting
        the least recently used (LRU) scenes.

        Args:
            scene: scene index (0-based).
            progress_callback: optional callback(current, total, message) for
                progress updates while loading.

        Returns:
            True if loaded successfully, False otherwise.
        '''
        scene_info = self._scene_info[scene]

        print(f'\nLoading scene {self.get_scene_number(scene):04d}...')

        start = time.perf_counter()
        
        frame_count = self.get_frame_count(scene)
        
        sensor_types = ['lidar', 'radar']

        total_tasks = frame_count * len(sensor_types) * len(scene_info['frame_paths'])

        tasks = []

        scene_data = {}

        for entity in scene_info['frame_paths']:
            for sensor_type in sensor_types:
                for frame in range(frame_count):
                    task = self._executor.submit(self._load_single_frame, scene, entity, frame, sensor_type)
                    
                    tasks.append((task, entity, frame, sensor_type))

            # Initialize the cache for entities.
            scene_data[entity] = {sensor_type: [None] * frame_count for sensor_type in sensor_types}
        
        # Wait for all tasks to complete and update the cache.
        completed = 0
        
        for task, entity, frame, sensor_type in tasks:
            try:
                frame_data = task.result()
                
                scene_data[entity][sensor_type][frame] = frame_data
                
                completed += 1

                if progress_callback and completed % 10 == 0:
                    progress_callback(completed, total_tasks, f'Loading: {completed}/{total_tasks}')

            except Exception as e:
                print(f'Error while loading scene {scene} [{entity}], frame {frame}, sensor {sensor_type}: {e}')
                
                scene_data[entity][sensor_type][frame] = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': []}

        for entity in scene_info['frame_paths']:
            cache_key = (scene, entity)
        
            # Store in cache.
            self._cache[cache_key] = scene_data[entity]

            # Free up memory after caching entity data.
            scene_data[entity] = None  
        
            self._mark_accessed(scene, entity)

        elapsed = time.perf_counter() - start
        
        print(f'Scene {self.get_scene_number(scene):04d} loaded in {elapsed:.2f} s.')

        if progress_callback:
            progress_callback(total_tasks, total_tasks, 'Scene loaded.')

        return True
    
    def _load_single_frame(self, scene: int, entity: str, frame: int, sensor_type: str) -> dict:
        '''
        Worker for loading a single frame for a specific entity and sensor type.

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
            frame: frame index (0-based).
            sensor_type: type of sensor data to load.

        Returns:
            Frame data dictionary with 'points', 'colors', 'bboxes'.
        '''
        scene_info = self._scene_info[scene]
        
        frame_data = scene_info['frame_paths'][entity][frame]

        entity_name = entity.split('_')[0]  # 'vehicle' or 'rsu'
        
        lidar_key = 'LIDAR' if entity_name == 'vehicle' else 'RSU-LIDAR'
        rad_names = RAD_NAME if entity_name == 'vehicle' else RSU_RAD_NAME

        # Load bounding boxes.
        gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
        
        global2lidar = get_global2sensor(frame_data, self.metadata, lidar_key)
        
        corners, labels, difficulty = transform_bbox(gt_det, global2lidar, self.ignore_valid_flag)
        
        bboxes = [{'corners': c, 'label': l, 'difficulty': d} for c, l, d in zip(corners, labels, difficulty)]

        if sensor_type == 'lidar':
            if 'LIDAR' in frame_data:
                points_dict = self._load_lidar(frame_data, bboxes)

                if self._trim_step > 1 and points_dict['points'].shape[0] > 0:
                    points_dict = self._trim_points(points_dict, self._trim_step)
                
                return points_dict
            
            return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}

        elif sensor_type == 'semantic-lidar':
            if 'SEG-LIDAR' in frame_data:
                points_dict = self._load_semantic_lidar(frame_data, bboxes)
                
                if self._trim_step > 1 and points_dict['points'].shape[0] > 0:
                    points_dict = self._trim_points(points_dict, self._trim_step)
                
                return points_dict
            
            return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}

        elif sensor_type == 'radar':
            if all(key in frame_data for key in rad_names):
                return self._load_radar(frame_data, rad_names, entity_name, bboxes)
            
            return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}

        else:
            raise ValueError(f'Unknown sensor type: {sensor_type}')

    def _load_radar(self, frame_data: dict, rad_names: list, entity_name: str, bboxes: list) -> dict:
        '''
        Load radar point clouds.

        Args:
            frame_data: dictionary of frame data.
            rad_names: list of radar names.
            entity_name: 'vehicle' or 'rsu'.
            bboxes: list of bounding boxes.

        Returns:
            Dictionary with 'points', 'colors', 'bboxes'.
        '''
        point_cloud_list = []
        velocity_list = []

        for radar in rad_names:
            # Metadata keys use the 'RSU-' prefix for RSUs; frame data keys are plain.
            meta_key = radar if entity_name == 'vehicle' else f'RSU-{radar}'

            radar2lidar = np.eye(4, dtype=np.float32)
            
            radar2lidar[:3, :3] = Q(self.metadata[meta_key]['sensor2lidar_rotation']).rotation_matrix
            radar2lidar[:3, 3] = self.metadata[meta_key]['sensor2lidar_translation']

            radar_points = np.load(frame_data[radar])['data']
            
            velocity_list.append(radar_points[:, -1])
            
            radar_points = radar_points[:, :-1]

            # Transform depth, altitude, and azimuth data to x, y, and z.
            x = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.cos(radar_points[:, 2])
            y = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.sin(radar_points[:, 2])
            z = radar_points[:, 0] * np.sin(radar_points[:, 1])

            points = np.stack((x, y, z), axis=1)
            
            points_transformed = (radar2lidar @ np.append(points, np.ones((points.shape[0], 1)), 1).T)[:3].T

            point_cloud_list.append(points_transformed)

        point_cloud = np.concatenate(point_cloud_list, axis=0)
        velocity = np.concatenate(velocity_list, axis=0)

        # Velocity-based colors.
        log_velocity = np.log(1.0 + np.abs(velocity))
        
        log_velocity_normalized = (log_velocity - log_velocity.min()) \
            / (log_velocity.max() - log_velocity.min() + 1e-6)

        colors = np.c_[
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 2]),
        ]

        return {'points': point_cloud, 'colors': colors, 'bboxes': bboxes}

    def _trim_points(self, points_dict, trim_step):
        '''
        Trim point cloud data based on the provided trim step.

        Args:
            points_dict: dictionary containing 'points' and 'colors'.
            trim_step: channel step size for trimming the point cloud.

        Returns:
            points_dict: dictionary with trimmed 'points' and 'colors'.
        '''
        points = points_dict['points']

        # Calculate beam angles.
        angles = np.arctan(points[:, 2] / np.linalg.norm(points[:, :2], axis=1))
        angles = np.trunc(angles * 1000.0) / 1000.0

        unique_angles = np.sort(np.unique(angles))

        # Some beams may have duplicate corresponding angles due to truncation,
        # e.g. 0.186 and 0.187. For each set, take one angle as the representative
        # and replace all other duplicates with that one.
        channels = []
        extras = []

        for angle in unique_angles:
            if len(channels) == 0:
                channels.append(angle)
            elif abs(np.array(channels) - angle).min() < 0.0015:
                extras.append(angle)
            else:
                channels.append(angle)
        
        for extra in extras:
            angles[angles == extra] = channels[np.abs(np.array(channels) - extra).argmin()]
        
        # Trim the point cloud based on the provided trim step.
        lidar_angles = np.sort(np.array(channels))[::trim_step]

        mask = np.isin(angles, lidar_angles)

        points_dict['points'] = points[mask]
        points_dict['colors'] = points_dict['colors'][mask] if points_dict['colors'] is not None else None

        return points_dict

    def load_voxels(self, scene: int, entity: str, frame: int) -> dict:
        '''
        Load voxel grids for an entity.

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
            frame: frame index (0-based).

        Returns:
            Dictionary with 'centers', 'colors', 'bboxes', 'voxel_size'.
        '''
        scene_info = self._scene_info[scene]

        if entity not in scene_info['frame_paths']:
            return {'centers': np.empty((0, 3)), 'colors': None, 'bboxes': [], 'voxel_size': 0.0}

        frame_data = scene_info['frame_paths'][entity][frame]

        entity_name = entity.split('_')[0]
        
        lidar_key = 'LIDAR' if entity_name == 'vehicle' else 'RSU-LIDAR'

        bboxes = []

        if 'GT_DET' in frame_data:
            gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
            
            global2lidar = get_global2sensor(frame_data, self.metadata, lidar_key)
            
            corners, labels, difficulty = transform_bbox(gt_det, global2lidar, self.ignore_valid_flag)
            
            bboxes = [{'corners': c, 'label': l, 'difficulty': d} for c, l, d in zip(corners, labels, difficulty)]

        voxel_key = 'VOXEL-GRID-FILLED' if self._filled_voxels else 'VOXEL-GRID'

        if voxel_key not in frame_data or not os.path.exists(frame_data[voxel_key]):
            return {'centers': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes, 'voxel_size': 0.0}

        voxel_grid = np.load(frame_data[voxel_key])['data']
        
        indices = np.argwhere(voxel_grid > 0)

        if indices.shape[0] == 0:
            return {'centers': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes, 'voxel_size': 0.0}

        voxel_meta_key = 'VOXEL-GRID' if entity_name == 'vehicle' else 'RSU-VOXEL-GRID'
        voxel_props_key = 'voxel_detector_properties' if entity_name == 'vehicle' else 'rsu_voxel_detector_properties'

        voxel_size = self.metadata[voxel_props_key]['voxel_size']
        
        grid_origin = np.array(self.metadata[voxel_meta_key]['sensor2lidar_translation']) - \
            np.array([
                self.metadata[voxel_props_key]['range'],
                self.metadata[voxel_props_key]['range'],
                -self.metadata[voxel_props_key]['lower_limit'],
            ])

        # Voxel centers in world coordinates.
        voxel_centers = (indices + 0.5) * voxel_size + grid_origin
        
        # Get semantic labels for coloring
        labels = voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        
        colors = LABEL_COLORS[labels]

        return {'centers': voxel_centers, 'colors': colors, 'bboxes': bboxes, 'voxel_size': voxel_size}

    def get_frame(self, scene: int, frame: int, entity: str, sensor_type: str) -> dict:
        '''
        Get frame data from cache. Scene/entity must be loaded first.

        Args:
            scene: scene index (0-based).
            frame: frame index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'.

        Returns:
            Frame data dictionary with 'points', 'colors', 'bboxes'.
        '''
        cache_key = (scene, entity) if entity is not None else scene

        if cache_key not in self._cache:
            raise RuntimeError(
                f'Scene {scene}{"" if entity is None else f" [{entity}]"} not loaded.'
            )

        self._mark_accessed(scene, entity)

        return self._cache[cache_key][sensor_type][frame]

    def unload_entity(self, scene: int, entity: str = None):
        '''
        Unload scene/entity data from cache to free up memory.

        Args:
            scene: scene index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
        '''
        cache_key = (scene, entity) if entity is not None else scene

        if cache_key in self._cache:
            del self._cache[cache_key]

            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)

            suffix = '' if entity is None else f' [{entity}]'
            
            print(f'Scene {self.get_scene_number(scene):04d}{suffix} unloaded from cache.')

    def get_cache_info(self):
        '''Get information about the current state of the cache.'''
        return {
            'cached_items': len(self._cache),
            'max_cached': self.max_cached,
            'cached_keys': list(self._cache.keys()),
            'access_order': self._cache_access_order.copy(),
        }

    def cleanup(self):
        '''Clean up resources.'''
        self._executor.shutdown(wait=False)
        self._cache.clear()
        self._cache_access_order.clear()

    def get_lidar2global(self, scene: int, frame: int, entity: str) -> np.ndarray:
        '''
        Get the lidar-to-global transformation matrix for an entity at a given
        frame.

        Args:
            scene: scene index (0-based).
            frame: frame index (0-based).
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').

        Returns:
            lidar-to-global transformation matrix.
        '''
        scene_info = self._scene_info[scene]

        if entity not in scene_info['frame_paths']:
            return np.eye(4, dtype=np.float32)

        frame_data = scene_info['frame_paths'][entity][frame]
        
        entity_name = entity.split('_')[0]

        ego2global = np.eye(4, dtype=np.float32)
        
        ego2global[:3, :3] = Q(frame_data['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = frame_data['ego2global_translation']

        sensor_key = 'LIDAR' if entity_name == 'vehicle' else 'RSU-LIDAR'

        sensor2ego = np.eye(4, dtype=np.float32)
        
        sensor2ego[:3, :3] = Q(self.metadata[sensor_key]['sensor2ego_rotation']).rotation_matrix
        sensor2ego[:3, 3] = self.metadata[sensor_key]['sensor2ego_translation']

        return ego2global @ sensor2ego

    def load_combined_bboxes(self, scene: int, frame: int) -> list:
        '''
        Load the combined bounding boxes for all entities at a given frame.
        The bounding boxes are already in the global coordinate system.

        Args:
            scene: scene index (0-based).
            frame: frame index (0-based).

        Returns:
            List of dictionaries with 'corners', 'label', 'difficulty'.
        '''
        scene_info = self._scene_info[scene]

        frame_data = scene_info['frame_paths'][next(iter(scene_info['frame_paths']))][frame]

        if 'GT_DET_COMBINED' not in frame_data:
            return []

        combined_path = frame_data['GT_DET_COMBINED']

        if not os.path.exists(combined_path):
            return []

        if combined_path.endswith('.json'):
            with open(combined_path, 'r') as f:
                gt_det = np.array(json.load(f), dtype=object)
        else:
            gt_det = np.load(combined_path, allow_pickle=True)

        # Bounding boxes are already in the global coordinate system.
        corners, labels, difficulty = transform_bbox(gt_det, np.eye(4, dtype=np.float32), self.ignore_valid_flag)

        return [{'corners': c, 'label': l, 'difficulty': d} for c, l, d in zip(corners, labels, difficulty)]


class SingleEntityInteractiveVisualizer(InteractiveVisualizer):
    '''
    Interactive Open3D visualizer with GUI controls that displays point cloud
    and voxel data for a single selected entity (vehicle or RSU) on demand.

    Args:
        data_loader: V2XVizDataLoader instance.
        all_entities: list of all entity labels (e.g. ['vehicle_0', 'rsu_0']).
        title: window title.
        point_size: point cloud rendering size.
    '''
    def __init__(
        self,
        data_loader: V2XVizDataLoader,
        all_entities: list,
        title: str = 'SimBEV2X Interactive Viewer',
        point_size: float = 2.0
    ):
        # Set up entity state before calling super().__init__(), which triggers
        # _create_control_panel() and _load_and_display_scene() — both need these.
        self._entity_list = all_entities
        self._current_entity = all_entities[0]

        super().__init__(data_loader, title=title, point_size=point_size)

    def _load_and_display_scene(self, scene: int, reset_camera: bool = True):
        '''
        Load the desired scene for the current entity and display the first
        frame.
        
        Args:
            scene: scene index (0-based).
            reset_camera: whether to reset the camera view after loading.
        '''
        if self._data_loader.is_loaded(scene, self._current_entity):
            self._update_frame()
            
            return

        self._is_loading = True
        self._update_loading_label('Loading scene...')

        def progress_callback(current, total, message):
            self._load_progress = current
            self._load_total = total
            
            gui.Application.instance.post_to_main_thread(self._window, lambda: self._update_loading_label(message))

        entity_snapshot = self._current_entity

        def load_worker():
            success = self._data_loader.load_entity(scene, entity_snapshot, progress_callback)
            
            gui.Application.instance.post_to_main_thread(
                self._window,
                lambda: self._on_scene_loaded(success, reset_camera)
            )

        threading.Thread(target=load_worker, daemon=True).start()

    def _format_entity_label(self, entity: str) -> str:
        '''
        Format entity key for UI display.
        
        Args:
            entity: entity key (e.g. 'vehicle_0' or 'rsu_2').
        
        Returns:
            Formatted label (e.g. 'Vehicle 0' or 'RSU 2').
        '''
        tokens = entity.replace('_', ' ').split(' ')

        if tokens:
            if tokens[0].lower() == 'vehicle':
                tokens[0] = 'Vehicle'
            elif tokens[0].lower() == 'rsu':
                tokens[0] = 'RSU'

        return ' '.join(tokens)

    def _create_control_panel(self):
        '''Create UI control panel.'''
        em = self._window.theme.font_size

        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Entity selection.
        self._panel.add_child(gui.Label('Entity:'))

        self._entity_combo = gui.Combobox()
        
        for entity in self._entity_list:
            self._entity_combo.add_item(self._format_entity_label(entity))
        
        self._entity_combo.selected_index = 0
        self._entity_combo.set_on_selection_changed(self._on_entity_changed)

        self._panel.add_child(self._entity_combo)
        self._panel.add_fixed(2 * em)

        # Sensor type selection.
        self._panel.add_child(gui.Label('Sensor Type:'))

        self._sensor_radio = gui.RadioButton(gui.RadioButton.VERT)
        self._sensor_radio.set_items(['Lidar', 'Semantic Lidar', 'Radar', 'Voxels'])
        self._sensor_radio.selected_index = 0
        self._sensor_radio.set_on_selection_changed(self._on_sensor_changed)

        self._panel.add_child(self._sensor_radio)
        self._panel.add_fixed(2 * em)

        # Scene number field.
        self._scene_label = gui.Label(f'Scene: {self._data_loader.get_scene_number(0):04d} (1/{self._max_scene + 1})')
        
        self._panel.add_child(self._scene_label)

        self._scene_number_field = gui.NumberEdit(gui.NumberEdit.INT)
        self._scene_number_field.set_limits(0, self._max_scene)
        self._scene_number_field.set_on_value_changed(self._on_scene_number_field_changed)
        
        self._panel.add_child(self._scene_number_field)
        self._panel.add_fixed(2 * em)

        # Frame slider.
        self._frame_label = gui.Label(f'Frame: 1/{self._max_frame + 1}')
        
        self._panel.add_child(self._frame_label)

        self._frame_slider = gui.Slider(gui.Slider.INT)
        self._frame_slider.set_limits(0, self._max_frame)
        self._frame_slider.set_on_value_changed(self._on_frame_slider_changed)
        
        self._panel.add_child(self._frame_slider)
        self._panel.add_fixed(0.2 * em)

        # Frame navigation buttons.
        frame_button_layout = gui.Horiz()
        frame_button_layout.add_stretch()
        
        self._prev_frame_button = gui.Button('<')
        self._prev_frame_button.set_on_clicked(self._on_prev_frame_clicked)
        
        frame_button_layout.add_child(self._prev_frame_button)
        
        self._next_frame_button = gui.Button('>')
        self._next_frame_button.set_on_clicked(self._on_next_frame_clicked)
        
        frame_button_layout.add_child(self._next_frame_button)
        frame_button_layout.add_stretch()
        
        self._panel.add_child(frame_button_layout)
        self._panel.add_fixed(2 * em)

        # Playback controls.
        playback_layout = gui.Horiz()
        
        self._play_button = gui.Button('Play')
        self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
        self._play_button.set_on_clicked(self._on_play_clicked)
        
        playback_layout.add_child(self._play_button)
        playback_layout.add_fixed(em)
        
        # Playback loop checkbox.
        self._loop_checkbox = gui.Checkbox('Loop Playback')
        self._loop_checkbox.checked = False
        self._loop_checkbox.set_on_checked(self._on_loop_toggle)
        
        playback_layout.add_child(self._loop_checkbox)
        
        self._panel.add_child(playback_layout)
        self._panel.add_fixed(em)

        # Playback speed slider.
        self._panel.add_child(gui.Label('Playback Speed (FPS):'))
        
        self._speed_slider = gui.Slider(gui.Slider.INT)
        self._speed_slider.set_limits(1, 30)
        self._speed_slider.int_value = self._play_speed
        self._speed_slider.set_on_value_changed(self._on_speed_changed)
        
        self._panel.add_child(self._speed_slider)
        self._panel.add_fixed(2 * em)

        # Bounding box toggle.
        self._bbox_checkbox = gui.Checkbox('Show Bounding Boxes')
        self._bbox_checkbox.checked = True
        self._bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        
        self._panel.add_child(self._bbox_checkbox)
        self._panel.add_fixed(2 * em)

        # Point size control.
        self._panel.add_child(gui.Label('Point Size:'))
        
        self._point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self._point_size_slider.set_limits(1.0, 20.0)
        self._point_size_slider.double_value = self._point_size
        self._point_size_slider.set_on_value_changed(self._on_point_size_slider_changed)
        
        self._panel.add_child(self._point_size_slider)
        self._panel.add_fixed(em)

        # Info label.
        self._info_label = gui.Label('')
        
        self._panel.add_child(self._info_label)
        self._panel.add_fixed(2 * em)

        # Loading status label.
        self._loading_label = gui.Label('')
        
        self._panel.add_child(self._loading_label)
        self._panel.add_fixed(em)

        # Cache status label.
        self._cache_status_label = gui.Label('')
        
        self._panel.add_child(self._cache_status_label)
        
        self._update_cache_status()
        
        self._panel.add_fixed(4 * em)

        # Camera view buttons.
        self._panel.add_child(gui.Label('Camera View:'))

        camera_button_layout = gui.Horiz()
        camera_button_layout.add_stretch()

        view_button_layout_left = gui.Vert()
        view_button_layout_right = gui.Vert()

        self._bev_button = gui.Button('BEV')
        self._bev_button.set_on_clicked(self._on_bev_view)
        self._bev_button.horizontal_padding_em = 2.4
        
        view_button_layout_left.add_child(self._bev_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._tracker_button = gui.Button('Tracker')
        self._tracker_button.set_on_clicked(self._on_tracker_view)
        self._tracker_button.horizontal_padding_em = 1.6
        
        view_button_layout_right.add_child(self._tracker_button)
        view_button_layout_right.add_fixed(0.2 * em)

        self._left_button = gui.Button('Left')
        self._left_button.set_on_clicked(self._on_left_view)
        
        view_button_layout_left.add_child(self._left_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._right_button = gui.Button('Right')
        self._right_button.set_on_clicked(self._on_right_view)
        
        view_button_layout_right.add_child(self._right_button)
        view_button_layout_right.add_fixed(0.2 * em)

        self._front_button = gui.Button('Front')
        self._front_button.set_on_clicked(self._on_front_view)
        
        view_button_layout_left.add_child(self._front_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._back_button = gui.Button('Back')
        self._back_button.set_on_clicked(self._on_back_view)
        
        view_button_layout_right.add_child(self._back_button)
        view_button_layout_right.add_fixed(0.2 * em)

        camera_button_layout.add_child(view_button_layout_left)
        camera_button_layout.add_stretch()
        
        camera_button_layout.add_child(view_button_layout_right)
        camera_button_layout.add_stretch()

        self._panel.add_child(camera_button_layout)

    def _update_cache_status(self):
        '''Update cache status label.'''
        cache_info = self._data_loader.get_cache_info()
        
        cached = cache_info['cached_items']
        max_cached = cache_info['max_cached']

        # Show which items are cached.
        if cache_info['cached_keys']:
            labels = []
            
            for scene_idx, entity in cache_info['cached_keys']:
                scene_num = self._data_loader.get_scene_number(scene_idx)
                
                labels.append(f'{scene_num:04d}/{entity}')
            
            self._cache_status_label.text = (f'Cache: {cached}/{max_cached}\nLoaded: {", ".join(labels)}')
        else:
            self._cache_status_label.text = f'Cache: {cached}/{max_cached}'

    def _on_entity_changed(self, label: str, index: int):
        '''
        Handle entity combobox selection change.
        
        Args:
            label: selected label.
            index: selected index.
        '''
        if self._is_playing:
            self._stop_playback()

        self._current_entity = self._entity_list[index]

        self._load_and_display_scene(self._current_scene, False)
    
    def _update_frame(self):
        '''Update visualization of the current frame.'''
        if self._is_loading:
            return

        try:
            if self._sensor_type == 'voxels':
                sensor_data = self._data_loader.load_voxels(
                    self._current_scene,
                    self._current_entity,
                    self._current_frame
                )
            else:
                sensor_data = self._data_loader.get_frame(
                    self._current_scene,
                    self._current_frame,
                    self._current_entity,
                    self._sensor_type
                )
        except Exception as e:
            print(f'Unexpected error: {e}')
            
            return

        # Validate the data.
        if sensor_data is None or all(k not in sensor_data for k in ['points', 'centers']):
            return

        try:
            # Remove old geometry.
            if self._scene_widget.scene.has_geometry('point_cloud'):
                self._scene_widget.scene.remove_geometry('point_cloud')

            # For voxels, create cube meshes instead of point cloud.
            if self._sensor_type == 'voxels' and 'centers' in sensor_data:
                voxel_centers = sensor_data['centers']
                voxel_colors = sensor_data['colors'] ** 1.8
                voxel_size = sensor_data['voxel_size']

                # Create a combined mesh for all voxels.
                combined_mesh = o3d.geometry.TriangleMesh()

                for center, color in zip(voxel_centers, voxel_colors):
                    # Create a small cube at each voxel position.
                    cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
                    
                    # Translate to center the cube at the voxel position
                    cube.translate(center - voxel_size / 2)
                    
                    cube.paint_uniform_color(color)
                    
                    combined_mesh += cube

                mat = o3d.visualization.rendering.MaterialRecord()
                
                mat.shader = 'defaultLit'
                
                self._scene_widget.scene.add_geometry('point_cloud', combined_mesh, mat)
            else:
                # Standard point cloud for lidar, semantic-lidar, radar.
                pcd = o3d.geometry.PointCloud()
                
                pcd.points = o3d.utility.Vector3dVector(sensor_data['points'])

                if 'colors' in sensor_data and sensor_data['colors'] is not None:
                    pcd.colors = o3d.utility.Vector3dVector(sensor_data['colors'] ** 2.2)
                else:
                    colors = np.tile([0.8, 0.8, 0.8], (len(sensor_data['points']), 1))
                    pcd.colors = o3d.utility.Vector3dVector(colors ** 2.2)

                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = 'defaultUnlit'
                mat.point_size = self._point_size
                
                self._scene_widget.scene.add_geometry('point_cloud', pcd, mat)

            # Update bounding boxes.
            self._update_bboxes(sensor_data.get('bboxes', []))

        except Exception as e:
            print(f'Error updating geometry: {e}')
            
            return

        # Update labels.
        scene_number = self._data_loader.get_scene_number(self._current_scene)
        
        self._scene_label.text = (f'Scene: {scene_number:04d} ({self._current_scene + 1}/{self._max_scene + 1})')
        self._frame_label.text = f'Frame: {self._current_frame + 1}/{self._max_frame + 1}'

        if 'points' in sensor_data:
            self._info_label.text = (
                f'Entity: {self._format_entity_label(self._current_entity)}\n'
                f'Points: {len(sensor_data["points"])}\n'
                f'Bounding Boxes: {len(sensor_data.get("bboxes", []))}'
            )
        else:
            self._info_label.text = (
                f'Entity: {self._format_entity_label(self._current_entity)}\n'
                f'Voxels: {len(sensor_data["centers"])}\n'
                f'Bounding Boxes: {len(sensor_data.get("bboxes", []))}'
            )

        self._update_cache_status()


class MultiEntityInteractiveVisualizer(SingleEntityInteractiveVisualizer):
    '''
    Interactive Open3D visualizer with GUI controls that displays point cloud
    data for all entities in a scene on demand.

    Args:
        data_loader: V2XVizDataLoader instance.
        all_entities: list of all entity labels (e.g. ['vehicle_0', 'rsu_0']).
        title: window title.
        point_size: point cloud rendering size.
    '''
    def __init__(
        self,
        data_loader: V2XVizDataLoader,
        all_entities: list,
        title: str = 'SimBEV2X Interactive Viewer',
        point_size: float = 2.0
    ):
        # Per-entity display state needed before super().__init__().
        self._entity_colors = {entity: ENTITY_COLORS[i % len(ENTITY_COLORS)] for i, entity in enumerate(all_entities)}
        
        self._active_entities = {entity: True for entity in all_entities}

        vehicles = [e for e in all_entities if e.startswith('vehicle')]
        
        self._anchor_entity = vehicles[0] if vehicles else all_entities[0]

        super().__init__(data_loader, all_entities, title=title, point_size=point_size)

        self._num_sensor_types = 2

    def _load_and_display_scene(self, scene: int):
        '''
        Load the desired scene and display the first frame.
        
        Args:
            scene: scene index (0-based).
        '''
        self._data_loader.clear_cache()

        self._is_loading = True
        self._update_loading_label('Loading scene...')

        def progress_callback(current, total, message):
            self._load_progress = current
            self._load_total = total
            
            gui.Application.instance.post_to_main_thread(self._window, lambda: self._update_loading_label(message))

        def load_worker():
            success = self._data_loader.load_scene(scene, progress_callback)
            
            gui.Application.instance.post_to_main_thread(self._window, lambda: self._on_scene_loaded(success))

        threading.Thread(target=load_worker, daemon=True).start()

    def _create_control_panel(self):
        '''Create UI control panel.'''
        em = self._window.theme.font_size

        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Entity checkboxes.
        _vehicles = [e for e in self._entity_list if e.startswith('vehicle')]
        _rsus = [e for e in self._entity_list if not e.startswith('vehicle')]

        def make_callback(e):
            def callback(checked):
                self._on_entity_checkbox_changed(e, checked)
            
            return callback

        def make_swatch(color, size=32):
            r, g, b = [int(c * 255) for c in color]
            img_np = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
            
            return gui.ImageWidget(o3d.geometry.Image(img_np))

        entitys_layout = gui.Horiz(em)

        if _vehicles:
            vehicle_col = gui.Vert(int(0.25 * em))
            vehicle_col.add_child(gui.Label('Vehicles'))
            
            for entity in _vehicles:
                cb = gui.Checkbox(self._format_entity_label(entity))
                
                cb.checked = True
                
                cb.set_on_checked(make_callback(entity))
                
                row = gui.Horiz(int(0.5 * em))
                
                row.add_child(make_swatch(self._entity_colors[entity]))
                row.add_child(cb)
                
                vehicle_col.add_child(row)
            
            entitys_layout.add_child(vehicle_col)

        if _rsus:
            rsu_col = gui.Vert(int(0.25 * em))
            rsu_col.add_child(gui.Label('RSUs'))
            
            for entity in _rsus:
                cb = gui.Checkbox(self._format_entity_label(entity))
                
                cb.checked = True
                
                cb.set_on_checked(make_callback(entity))
                
                row = gui.Horiz(int(0.5 * em))
                
                row.add_child(make_swatch(self._entity_colors[entity]))
                row.add_child(cb)
                
                rsu_col.add_child(row)
            
            entitys_layout.add_child(rsu_col)

        self._panel.add_child(entitys_layout)
        self._panel.add_fixed(em)

        # Sensor type selection.
        self._panel.add_child(gui.Label('Sensor Type:'))

        self._sensor_radio = gui.RadioButton(gui.RadioButton.VERT)
        self._sensor_radio.set_items(['Lidar', 'Radar'])
        self._sensor_radio.selected_index = 0
        self._sensor_radio.set_on_selection_changed(self._on_sensor_changed)

        self._panel.add_child(self._sensor_radio)
        self._panel.add_fixed(em)

        # Scene number field.
        self._scene_label = gui.Label(f'Scene: {self._data_loader.get_scene_number(0):04d} (1/{self._max_scene + 1})')
        
        self._panel.add_child(self._scene_label)

        self._scene_number_field = gui.NumberEdit(gui.NumberEdit.INT)
        self._scene_number_field.set_limits(0, self._max_scene)
        self._scene_number_field.set_on_value_changed(self._on_scene_number_field_changed)
        
        self._panel.add_child(self._scene_number_field)
        self._panel.add_fixed(em)

        # Frame slider.
        self._frame_label = gui.Label(f'Frame: 1/{self._max_frame + 1}')
        
        self._panel.add_child(self._frame_label)

        self._frame_slider = gui.Slider(gui.Slider.INT)
        self._frame_slider.set_limits(0, self._max_frame)
        self._frame_slider.set_on_value_changed(self._on_frame_slider_changed)
        
        self._panel.add_child(self._frame_slider)
        self._panel.add_fixed(0.2 * em)

        # Frame navigation buttons.
        frame_button_layout = gui.Horiz()
        frame_button_layout.add_stretch()
        
        self._prev_frame_button = gui.Button('<')
        self._prev_frame_button.set_on_clicked(self._on_prev_frame_clicked)
        
        frame_button_layout.add_child(self._prev_frame_button)
        
        self._next_frame_button = gui.Button('>')
        self._next_frame_button.set_on_clicked(self._on_next_frame_clicked)
        
        frame_button_layout.add_child(self._next_frame_button)
        frame_button_layout.add_stretch()
        
        self._panel.add_child(frame_button_layout)
        self._panel.add_fixed(em)

        # Playback controls.
        playback_layout = gui.Horiz()
        
        self._play_button = gui.Button('Play')
        self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
        self._play_button.set_on_clicked(self._on_play_clicked)
        
        playback_layout.add_child(self._play_button)
        playback_layout.add_fixed(em)
        
        # Playback loop checkbox.
        self._loop_checkbox = gui.Checkbox('Loop Playback')
        self._loop_checkbox.checked = False
        self._loop_checkbox.set_on_checked(self._on_loop_toggle)
        
        playback_layout.add_child(self._loop_checkbox)
        
        self._panel.add_child(playback_layout)
        self._panel.add_fixed(em)

        # Playback speed slider.
        self._panel.add_child(gui.Label('Playback Speed (FPS):'))
        
        self._speed_slider = gui.Slider(gui.Slider.INT)
        self._speed_slider.set_limits(1, 30)
        self._speed_slider.int_value = self._play_speed
        self._speed_slider.set_on_value_changed(self._on_speed_changed)
        
        self._panel.add_child(self._speed_slider)
        self._panel.add_fixed(em)

        # Bounding box toggle.
        self._bbox_checkbox = gui.Checkbox('Show Bounding Boxes')
        self._bbox_checkbox.checked = True
        self._bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        
        self._panel.add_child(self._bbox_checkbox)
        self._panel.add_fixed(em)

        # Point size control.
        self._panel.add_child(gui.Label('Point Size:'))
        
        self._point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self._point_size_slider.set_limits(1.0, 20.0)
        self._point_size_slider.double_value = self._point_size
        self._point_size_slider.set_on_value_changed(self._on_point_size_slider_changed)
        
        self._panel.add_child(self._point_size_slider)
        self._panel.add_fixed(em)

        # Info label.
        self._info_label = gui.Label('')
        
        self._panel.add_child(self._info_label)
        self._panel.add_fixed(em)

        # Loading status label.
        self._loading_label = gui.Label('')
        
        self._panel.add_child(self._loading_label)
        self._panel.add_fixed(em)

        # Cache status label.
        self._cache_status_label = gui.Label('')
        
        self._panel.add_child(self._cache_status_label)
        
        self._update_cache_status()
        
        self._panel.add_fixed(4 * em)

        # Anchor entity selection.
        self._panel.add_child(gui.Label('Camera view anchor:'))

        self._entity_combo = gui.Combobox()
        
        for entity in self._entity_list:
            self._entity_combo.add_item(self._format_entity_label(entity))
        
        self._entity_combo.selected_index = 0
        self._entity_combo.set_on_selection_changed(self._on_entity_changed)

        self._panel.add_child(self._entity_combo)
        self._panel.add_fixed(em)

        # Camera view buttons.
        self._panel.add_child(gui.Label('Camera View:'))

        camera_button_layout = gui.Horiz()
        camera_button_layout.add_stretch()

        view_button_layout_left = gui.Vert()
        view_button_layout_right = gui.Vert()

        self._bev_button = gui.Button('BEV')
        self._bev_button.set_on_clicked(self._on_bev_view)
        self._bev_button.horizontal_padding_em = 2.4
        
        view_button_layout_left.add_child(self._bev_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._tracker_button = gui.Button('Tracker')
        self._tracker_button.set_on_clicked(self._on_tracker_view)
        self._tracker_button.horizontal_padding_em = 1.6
        
        view_button_layout_right.add_child(self._tracker_button)
        view_button_layout_right.add_fixed(0.2 * em)

        self._left_button = gui.Button('Left')
        self._left_button.set_on_clicked(self._on_left_view)
        
        view_button_layout_left.add_child(self._left_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._right_button = gui.Button('Right')
        self._right_button.set_on_clicked(self._on_right_view)
        
        view_button_layout_right.add_child(self._right_button)
        view_button_layout_right.add_fixed(0.2 * em)

        self._front_button = gui.Button('Front')
        self._front_button.set_on_clicked(self._on_front_view)
        
        view_button_layout_left.add_child(self._front_button)
        view_button_layout_left.add_fixed(0.2 * em)

        self._back_button = gui.Button('Back')
        self._back_button.set_on_clicked(self._on_back_view)
        
        view_button_layout_right.add_child(self._back_button)
        view_button_layout_right.add_fixed(0.2 * em)

        camera_button_layout.add_child(view_button_layout_left)
        camera_button_layout.add_stretch()
        
        camera_button_layout.add_child(view_button_layout_right)
        camera_button_layout.add_stretch()

        self._panel.add_child(camera_button_layout)

    def _on_sensor_changed(self, index: int):
        '''
        Handle sensor type radio button change.
        
        Args:
            index: selected index.
        '''
        self._sensor_type = 'lidar' if index == 0 else 'radar'
        
        # Update point size based on sensor.
        if self._sensor_type == 'lidar':
            self._point_size = self._lidar_point_size
            
            self._scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
        elif self._sensor_type == 'radar':
            self._point_size = self._radar_point_size

            self._scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])

        self._point_size_slider.double_value = self._point_size
        
        self._update_frame()

    def _on_entity_changed(self, label: str, index: int):
        '''
        Handle entity combobox selection change.
        
        Args:
            label: selected label.
            index: selected index.
        '''
        if self._is_playing:
            self._stop_playback()

        self._anchor_entity = self._entity_list[index]

    def _on_bev_view(self):
        '''Set camera to bird's-eye view anchored on the anchor entity.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')
            
            pos = lidar2global[:3, 3]
            forward = lidar2global[:3, 0]
            
            forward_xy = np.array([forward[0], forward[1], 0.0])
            
            norm = np.linalg.norm(forward_xy)
            
            forward_xy = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        bounds = self._scene_widget.scene.bounding_box
        
        extent = bounds.get_extent()
        
        max_extent = max(extent[0], extent[1])

        center = pos.tolist()
        
        eye = [pos[0], pos[1], pos[2] + max_extent]
        
        self._scene_widget.look_at(center, eye, forward_xy)

    def _on_tracker_view(self):
        '''Set camera to anchor entity tracker view.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')
            
            pos = lidar2global[:3, 3]
            forward = lidar2global[:3, 0]

            forward_xy = np.array([forward[0], forward[1], 0.0])

            norm = np.linalg.norm(forward_xy)

            forward_xy = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        distance = 8.0
        
        center = pos.tolist()
        
        # Place the eye behind and slightly above the anchor entity.
        eye = [
            pos[0] - forward_xy[0] * 2.0 * distance,
            pos[1] - forward_xy[1] * 2.0 * distance,
            pos[2] + 0.5 * distance
        ]
        
        self._scene_widget.look_at(center, eye, [0, 0, 1])

    def _on_left_view(self):
        '''Set camera to anchor entity left side view.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')

            pos = lidar2global[:3, 3]
            left = lidar2global[:3, 1]
            
            left_xy = np.array([left[0], left[1], 0.0])
            
            norm = np.linalg.norm(left_xy)
            
            left_xy = left_xy / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        center = pos.tolist()

        # Place the eye to see the left side of the anchor entity.
        eye = [pos[0] - left_xy[0] * 2.0, pos[1] - left_xy[1] * 2.0, pos[2]]

        self._scene_widget.look_at(center, eye, [0, 0, 1])

    def _on_right_view(self):
        '''Set camera to anchor entity right side view.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')

            pos = lidar2global[:3, 3]
            left = lidar2global[:3, 1]
            
            left_xy = np.array([left[0], left[1], 0.0])
            
            norm = np.linalg.norm(left_xy)
            
            left_xy = left_xy / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        center = pos.tolist()

        # Place the eye to see the right side of the anchor entity.
        eye = [pos[0] + left_xy[0] * 2.0, pos[1] + left_xy[1] * 2.0, pos[2]]

        self._scene_widget.look_at(center, eye, [0, 0, 1])
    
    def _on_front_view(self):
        '''Set camera to anchor entity front view.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')
            
            pos = lidar2global[:3, 3]
            forward = lidar2global[:3, 0]

            forward_xy = np.array([forward[0], forward[1], 0.0])

            norm = np.linalg.norm(forward_xy)

            forward_xy = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        center = pos.tolist()

        # Place the eye to see the front of the anchor entity.
        eye = [pos[0] - forward_xy[0] * 2.0, pos[1] - forward_xy[1] * 2.0, pos[2]]

        self._scene_widget.look_at(center, eye, [0, 0, 1])

    def _on_back_view(self):
        '''Set camera to anchor entity back view.'''
        try:
            lidar2global = self._data_loader.get_lidar2global(
                self._current_scene,
                self._current_frame,
                self._anchor_entity
            )

            if np.array_equal(lidar2global, np.eye(4, dtype=np.float32)):
                raise ValueError('Invalid transformation matrix.')
            
            pos = lidar2global[:3, 3]
            forward = lidar2global[:3, 0]

            forward_xy = np.array([forward[0], forward[1], 0.0])

            norm = np.linalg.norm(forward_xy)

            forward_xy = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        except Exception as e:
            print(f'Error anchoring camera to {self._format_entity_label(self._anchor_entity)}: {e}')

            return

        center = pos.tolist()

        # Place the eye to see the back of the anchor entity.
        eye = [pos[0] + forward_xy[0] * 2.0, pos[1] + forward_xy[1] * 2.0, pos[2]]

        self._scene_widget.look_at(center, eye, [0, 0, 1])

    def _on_entity_checkbox_changed(self, entity: str, checked: bool):
        '''
        Handle entity checkbox toggle.
        
        Args:
            entity: entity label.
            checked: whether the checkbox is now checked.
        '''
        self._active_entities[entity] = checked
        
        self._update_frame()

    def _update_cache_status(self):
        '''Update cache status label.'''
        cache_info = self._data_loader.get_cache_info()

        # Show which entities are cached.
        if cache_info['cached_keys']:
            labels = []
            
            for _, entity in cache_info['cached_keys']:
                labels.append(self._format_entity_label(entity))
            
            self._cache_status_label.text = (f'Loaded: {", ".join(labels)}')
        else:
            self._cache_status_label.text = ''
    
    def _update_frame(self):
        '''Update visualization of all entities in the current frame.'''
        if self._is_loading:
            return

        all_points = []
        all_colors = []
        
        total_points = 0

        for i, entity in enumerate(self._entity_list):
            if not self._active_entities.get(entity, False):
                continue

            if not self._data_loader.is_loaded(self._current_scene, entity):
                continue

            try:
                sensor_data = self._data_loader.get_frame(
                    self._current_scene,
                    self._current_frame,
                    entity,
                    self._sensor_type
                )
            except Exception as e:
                print(f'Error getting sensor data for {self._format_entity_label(entity)}: {e}')
                
                continue

            # Validate the data.
            if sensor_data is None or 'points' not in sensor_data:
                continue

            points = sensor_data['points']
            
            n = len(points)

            if n == 0:
                continue

            # Transform from the entity's lidar coordinate system to the
            # global coordinate system.
            try:
                lidar2global = self._data_loader.get_lidar2global(self._current_scene, self._current_frame, entity)
                
                points_h = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
                
                points_global = (lidar2global @ points_h.T)[:3].T
            
            except Exception as e:
                print(f'Error transforming points for {self._format_entity_label(entity)}: {e}')
                
                points_global = points

            color = ENTITY_COLORS[i % len(ENTITY_COLORS)] ** 2.2
            
            all_points.append(points_global)
            all_colors.append(np.tile(color, (n, 1)))
            
            total_points += n

        # Remove old geometry.
        if self._scene_widget.scene.has_geometry('point_cloud'):
            self._scene_widget.scene.remove_geometry('point_cloud')

        if all_points:
            pcd = o3d.geometry.PointCloud()
            
            pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = 'defaultUnlit'
            mat.point_size = self._point_size
            
            self._scene_widget.scene.add_geometry('point_cloud', pcd, mat)

        try:
            bboxes = self._data_loader.load_combined_bboxes(self._current_scene, self._current_frame)

        except Exception as e:
            print(f'Error loading combined bounding boxes: {e}')
            
            bboxes = []

        # Update bounding boxes.
        self._update_bboxes(bboxes)

        # Update labels.
        scene_number = self._data_loader.get_scene_number(self._current_scene)
        
        self._scene_label.text = (f'Scene: {scene_number:04d} ({self._current_scene + 1}/{self._max_scene + 1})')
        self._frame_label.text = f'Frame: {self._current_frame + 1}/{self._max_frame + 1}'

        info_lines = [f'Points: {total_points}', f'Bounding Boxes: {len(bboxes)}']
        
        self._info_label.text = '\n'.join(info_lines)

        self._update_cache_status()


def visualize_interactive_single(ctx):
    '''
    Unified interactive visualizer for all sensors with point cloud or voxel
    data for a single entity.

    Args:
        ctx: visualization context.
    '''
    metadata = None

    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev2x/infos/simbev2x_infos_{split}.json'

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                infos = json.load(f)

            metadata = infos['metadata']
            
            break

    if metadata is None:
        print('Error: Could not load metadata.')
        
        return

    all_entities = []

    for entity_type in ['vehicle', 'rsu']:
        i = 0
        
        while os.path.exists(f'{ctx.path}/simbev2x/ground-truth/{entity_type}-{i}'):
            all_entities.append(f'{entity_type}_{i}')
            
            i += 1

    if not all_entities:
        print('Error: No entity directories found.')
        
        return

    # Create data loader with cache.
    print('Initializing data loader...')

    data_loader = V2XVizDataLoader(
        ctx.path,
        metadata,
        ignore_valid_flag=ctx.ignore_valid_flag,
        max_workers=16,
        max_cached=5,
        filled_voxels=ctx.filled_voxels,
        trim_step = ctx.trim_step
    )

    scene_count = data_loader.get_scene_count()

    if scene_count == 0:
        print('No scenes found to visualize.')
        
        return

    print(f'Found {scene_count} scene(s), {len(all_entities)} entity(s): {", ".join(all_entities)}.')

    # Create and run the visualizer.
    visualizer = SingleEntityInteractiveVisualizer(
        data_loader=data_loader,
        all_entities=all_entities,
        title='SimBEV2X Interactive Viewer',
        point_size=2.0,
    )

    visualizer.run()


def visualize_interactive_multi(ctx):
    '''
    Unified interactive visualizer for all sensors of all entities with point
    cloud data.

    Args:
        ctx: visualization context.
    '''
    metadata = None

    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev2x/infos/simbev2x_infos_{split}.json'

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                infos = json.load(f)

            metadata = infos['metadata']
            
            break

    if metadata is None:
        print('Error: Could not load metadata.')
        
        return

    all_entities = []

    for entity_type in ['vehicle', 'rsu']:
        i = 0
        
        while os.path.exists(f'{ctx.path}/simbev2x/ground-truth/{entity_type}-{i}'):
            all_entities.append(f'{entity_type}_{i}')
            
            i += 1

    if not all_entities:
        print('Error: No entity directories found.')
        
        return

    # Create data loader with cache.
    print('Initializing data loader...')

    data_loader = V2XVizDataLoader(
        ctx.path,
        metadata,
        ignore_valid_flag=ctx.ignore_valid_flag,
        max_workers=16,
        max_cached=len(all_entities),
        filled_voxels=ctx.filled_voxels,
        trim_step = ctx.trim_step
    )

    scene_count = data_loader.get_scene_count()

    if scene_count == 0:
        print('No scenes found to visualize.')
        
        return

    print(f'Found {scene_count} scene(s), {len(all_entities)} entity(s): {", ".join(all_entities)}.')

    visualizer = MultiEntityInteractiveVisualizer(
        data_loader=data_loader,
        all_entities=all_entities,
        title='SimBEV2X Interactive Viewer',
        point_size=2.0,
    )

    visualizer.run()
