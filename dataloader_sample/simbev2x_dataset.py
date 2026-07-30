import random

from .simbev_dataset import *


RSU_CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']


@DATASETS.register_module()
class SimBEV2XDataset(SimBEVDataset):
    '''
    This class serves as the API for experiments on the SimBEV2X dataset.

    Args:
        comm_range: maximum communication range for cooperating entities (m).
        elevation_threshold: maximum elevation difference for valid
            cooperation (m).
        cooperation_modes: list of cooperation modes to evaluate. Can be a
            combination of 'none', 'x2v', 'x2i', 'x2x'.
        sample_mode: entity sampling mode. Can be 'all', 'single-first', or 'single-random'.
        **kwargs: additional arguments passed to SimBEVDataset.
    '''
    def __init__(
        self,
        comm_range: float = 100.0,
        elevation_threshold: float = 3.2,
        ego_type: str = 'vehicle',
        cooperation_modes: list[str] = None,
        sample_mode: str = 'all',
        max_vehicles: int = 8,
        max_rsus: int = 4,
        **kwargs
    ):
        self.comm_range = comm_range
        self.elevation_threshold = elevation_threshold
        self.ego_type = ego_type
        self.cooperation_modes = cooperation_modes or ['none', 'x2v', 'x2i', 'x2x']
        self.sample_mode = sample_mode
        self.max_vehicles = max_vehicles
        self.max_rsus = max_rsus

        super().__init__(**kwargs)

        self.data_info_default = {
            'scene': -1,
            'frame': -1,
            'ego_id': -1,
            'type': 'vehicle',
            'timestamp': -1,
            'gt_seg_path': '',
            'gt_det_path': '',
            'lidar_path': '',
            'sweeps_lidar_paths': [],
            'sweeps_ego2global': [],
            'ego2global': np.eye(4).astype(np.float32),
            'lidar2ego': np.eye(4).astype(np.float32),
            'image_paths': [],
            'camera_intrinsics': [np.eye(4).astype(np.float32)] * len(CAM_NAME),
            'camera2lidar': [np.eye(4).astype(np.float32)] * len(CAM_NAME),
            'lidar2camera': [np.eye(4).astype(np.float32)] * len(CAM_NAME),
            'lidar2image': [np.eye(4).astype(np.float32)] * len(CAM_NAME),
            'camera2ego': [np.eye(4).astype(np.float32)] * len(CAM_NAME),
            'cooperative': False
        }

    def get_cat_ids(self, index: int):
        '''
        Get category IDs of objects in the sample.

        Args:
            index: index of the sample in the dataset.

        Returns:
            cat_ids: list of category IDs of objects in the sample.
        '''
        info = self.data_infos[index]

        ego_id = info['ego_id']
        ego_type = info['ego_type']

        ego_info = info[f'{ego_type}_{ego_id}']

        if self.use_valid_flag:
            mask = ego_info['valid_flag']
            
            gt_names = set(ego_info['gt_names'][mask])
        else:
            gt_names = set(ego_info['gt_names'])

        cat_ids = []

        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        
        return cat_ids
    
    def load_annotations(self, ann_file: str):
        '''
        Load annotations from the annotation file.

        Args:
            ann_file: annotation file of the dataset.

        Returns:
            data_infos: list of data samples in the dataset.
        '''
        annotations = mmcv.load(ann_file)

        self.metadata = annotations['metadata']

        self.transformations = self._calculate_transformations()

        self.rsu_transformations = self._calculate_transformations(
            ['RSU-' + name for name in RSU_CAM_NAME],
            'RSU-LIDAR',
            'rsu_camera_intrinsics'
        )

        data_infos = []

        for scene_key in annotations['data']:
            data = annotations['data'][scene_key]
            
            scene_data = data['scene_data']

            # Identify vehicle and RSU entities.
            vehicle_ids = sorted([int(k.split('_')[1]) for k in scene_data.keys() if k.startswith('vehicle_')])
            rsu_ids = sorted([int(k.split('_')[1]) for k in scene_data.keys() if k.startswith('rsu_')])

            ego_ids = vehicle_ids if self.ego_type == 'vehicle' else rsu_ids
            
            if not ego_ids:
                continue

            # Get the number of frames from the first entity.
            num_frames = len(scene_data[f'{self.ego_type}_{ego_ids[0]}'])

            # Create samples.
            if self.sample_mode == 'all':
                id_list = ego_ids
            elif self.sample_mode == 'single-first':
                id_list = [ego_ids[0]]
            elif self.sample_mode == 'single-random':
                id_list = [random.choice(ego_ids)]
            else:
                raise ValueError(f'Invalid sample mode: {self.sample_mode}')
            
            for id in id_list:
                for i in range(num_frames):
                    sample = {
                        'ego_id': id,
                        'ego_type': self.ego_type,
                        'scene': int(scene_key.split('_')[1]),
                        'frame': i,
                        'vehicle_ids': vehicle_ids,
                        'rsu_ids': rsu_ids
                    }

                    for entity in scene_data.keys():
                        assert sample['scene'] == scene_data[entity][i]['scene']
                        assert sample['frame'] == scene_data[entity][i]['frame']

                        sample[entity] = scene_data[entity][i]
                    
                    data_infos.append(sample)
        
        self.full_infos = data_infos

        data_infos = data_infos[::self.load_interval]

        data_infos = self.load_gt_bboxes(data_infos)

        return data_infos

    def load_gt_bboxes(self, infos: list):
        '''
        Load ground truth bounding boxes from file into the list of data
        samples.

        Args:
            infos: list of data samples in the dataset.
        
        Returns:
            infos: list of data samples updated with ground truth bounding
                boxes.
        '''
        for sample in infos:
            vehicle_ids = ['vehicle_' + str(id) for id in sample['vehicle_ids']]
            rsu_ids = ['rsu_' + str(id) for id in sample['rsu_ids']]

            for entity in vehicle_ids + rsu_ids:
                info = sample[entity]
            
                self._get_box_info(info)

        return infos
    
    def get_data_info(self, index: int):
        '''
        Package information from a data sample.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            data: packaged information from the sample.
        '''
        info = self.data_infos[index]

        ego_id = info['ego_id']
        ego_type = info['ego_type']
        
        ego_info = info[f'{ego_type}_{ego_id}']

        data = self._collect_data_info(index, ego_id, ego_info, ego_type)

        data['cooperative_entities'] = {}

        for mode in self.cooperation_modes:
            data['cooperative_entities'][mode] = {}
            
            vehicle_list, rsu_list = self.get_cooperating_entities(info, mode=mode)

            for id in range(self.max_vehicles):
                if id in vehicle_list:
                    coop_info = info[f'vehicle_{id}']
                    
                    data['cooperative_entities'][mode][f'vehicle_{id}'] = self._collect_data_info(
                        index,
                        id,
                        coop_info
                    )
                else:
                    data['cooperative_entities'][mode][f'vehicle_{id}'] = self.data_info_default.copy()
            
            for id in range(self.max_rsus):
                if id in rsu_list:
                    coop_info = info[f'rsu_{id}']
                    
                    data['cooperative_entities'][mode][f'rsu_{id}'] = self._collect_data_info(
                        index,
                        id,
                        coop_info,
                        'rsu'
                    )
                else:
                    data['cooperative_entities'][mode][f'rsu_{id}'] = self.data_info_default.copy()
                    
                    data['cooperative_entities'][mode][f'rsu_{id}']['type'] = 'rsu'

                    data['cooperative_entities'][mode][f'rsu_{id}']['camera_intrinsics'] = \
                        [np.eye(4).astype(np.float32)] * len(RSU_CAM_NAME)
                    data['cooperative_entities'][mode][f'rsu_{id}']['camera2lidar'] = \
                        [np.eye(4).astype(np.float32)] * len(RSU_CAM_NAME)
                    data['cooperative_entities'][mode][f'rsu_{id}']['lidar2camera'] = \
                        [np.eye(4).astype(np.float32)] * len(RSU_CAM_NAME)
                    data['cooperative_entities'][mode][f'rsu_{id}']['lidar2image'] = \
                        [np.eye(4).astype(np.float32)] * len(RSU_CAM_NAME)
                    data['cooperative_entities'][mode][f'rsu_{id}']['camera2ego'] = \
                        [np.eye(4).astype(np.float32)] * len(RSU_CAM_NAME)
        
        vehicle_list, rsu_list = self.get_cooperating_entities(info, mode='x2x')

        data['ann_info'] = self.get_ann_info(index, vehicle_list, rsu_list)
        
        data['raw_sample_info'] = info

        return data
    
    def _collect_data_info(self, index: int, ego_id: int, ego_info: dict, entity: str = 'vehicle'):
        '''
        Collect information from the data sample.

        Args:
            index: index of the sample in the dataset.
            ego_id: ID of the ego vehicle.
            ego_info: ego vehicle information dictionary.
            entity: type of entity ('vehicle' or 'rsu').
        
        Returns:
            data: dictionary of data sample information.
        '''
        transformations = self.transformations if entity == 'vehicle' else self.rsu_transformations

        camera_names = CAM_NAME if entity == 'vehicle' else RSU_CAM_NAME

        data = dict(
            scene = ego_info['scene'],
            frame = ego_info['frame'],
            ego_id = ego_id,
            type = entity,
            cooperative = True,
            timestamp = ego_info['timestamp'],
            gt_seg_path = ego_info['GT_SEG'],
            gt_det_path = ego_info['GT_DET'],
            lidar_path = ego_info['LIDAR'],
            sweeps_lidar_paths = [],
            sweeps_ego2global = []
        )

        # Ego to global transformation.
        ego2global = np.eye(4).astype(np.float32)
        
        ego2global[:3, :3] = Q(ego_info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = ego_info['ego2global_translation']
        
        data['ego2global'] = ego2global

        # Lidar to ego transformation.
        data['lidar2ego'] = transformations['lidar2ego']

        # Load lidar sweeps for the ego vehicle.
        for i in range(self.max_num_sweeps):
            if ego_info['frame'] - (i + 1) >= 0:
                sweep_info = self.full_infos[self.load_interval * index - (i + 1)]

                ego_sweep_info = sweep_info[f'{entity}_{ego_id}']

                data['sweeps_lidar_paths'].append(ego_sweep_info['LIDAR'])

                ego2global = np.eye(4).astype(np.float32)
        
                ego2global[:3, :3] = Q(ego_sweep_info['ego2global_rotation']).rotation_matrix
                ego2global[:3, 3] = ego_sweep_info['ego2global_translation']

                data['sweeps_ego2global'].append(ego2global)

        if self.modality['use_camera']:
            data['image_paths'] = []
            data['camera_intrinsics'] = transformations['camera_intrinsics']
            data['camera2lidar'] = transformations['camera2lidar']
            data['lidar2camera'] = transformations['lidar2camera']
            data['lidar2image'] = transformations['lidar2image']
            data['camera2ego'] = transformations['camera2ego']

            for camera in camera_names:
                data['image_paths'].append(ego_info['RGB-' + camera])
        
        return data

    def get_cooperating_entities(self, info: dict, mode: str = 'x2x') -> tuple[list]:
        '''
        Find all cooperating entities within the communication range.

        Args:
            info: data sample information dictionary.
            mode: cooperation mode ('x2v', 'x2i', 'x2x').
        
        Returns:
            vehicle_list: list of cooperating vehicle indices.
            rsu_list: list of cooperating RSU indices.
        '''
        vehicle_list = []
        rsu_list = []

        ego_id = info['ego_id']
        ego_type = info['ego_type'] 

        ego_location = np.array(info[f'{ego_type}_{ego_id}']['ego2global_translation'])

        # Determine which vehicles can cooperate with the ego vehicle.
        if mode in ['x2v', 'x2x']:
            for id in info['vehicle_ids']:
                if id == ego_id and ego_type == 'vehicle':
                    continue

                coop_location = np.array(info[f'vehicle_{id}']['ego2global_translation'])

                distance = np.linalg.norm(ego_location - coop_location)

                if distance <= self.comm_range:
                    if abs(ego_location[2] - coop_location[2]) <= self.elevation_threshold:
                        vehicle_list.append(id)
        
        # Determine which RSUs can cooperate with the ego vehicle.
        if mode in ['x2i', 'x2x']:
            for id in info['rsu_ids']:
                if id == ego_id and ego_type == 'rsu':
                    continue

                coop_location = np.array(info[f'rsu_{id}']['ego2global_translation'])

                distance = np.linalg.norm(ego_location - coop_location)

                if distance <= self.comm_range:
                    if abs(ego_location[2] - coop_location[2]) <= self.elevation_threshold:
                        rsu_list.append(id)
        
        return vehicle_list, rsu_list

    def get_ann_info(self, index, vehicle_list: list = [], rsu_list: list = []):
        '''
        Get annotation information for a data sample.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            anns_results: annotation information from the sample.
        '''
        info = self.data_infos[index]

        ego_id = info['ego_id']
        ego_type = info['ego_type']
        
        ego_info = info[f'{ego_type}_{ego_id}']

        entity_list = ['vehicle_' + str(id) for id in vehicle_list] + ['rsu_' + str(id) for id in rsu_list]

        for entity in entity_list:
            entity_info = info[entity]

            for i, obj_id in enumerate(ego_info['gt_box_ids']):
                if obj_id in entity_info['gt_box_ids']:
                    j = np.argwhere(entity_info['gt_box_ids'] == obj_id)[0][0]

                    ego_info['valid_flag'][i] = ego_info['valid_flag'][i] or entity_info['valid_flag'][j]

        if self.use_valid_flag:
            mask = ego_info['valid_flag']
        else:
            mask = np.array([True] * len(ego_info['gt_box_ids']))
        
        gt_bboxes_3d = ego_info['gt_boxes'][mask]
        gt_names_3d = ego_info['gt_names'][mask]
        gt_distances_3d = ego_info['distance_to_ego'][mask]
        gt_angles_3d = ego_info['angle_to_ego'][mask]
        gt_difficulty_3d = ego_info['difficulty'][mask]

        gt_difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}

        gt_difficulty = np.array([gt_difficulty_map[d] for d in gt_difficulty_3d], dtype=np.int64)

        gt_angles_3d[np.isnan(gt_angles_3d)] = 0.0

        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = ego_info['gt_velocity'][mask]

            gt_velocity[np.isnan(gt_velocity[:, 0])] = [0.0, 0.0]
            
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_distances=gt_distances_3d,
            gt_angles=gt_angles_3d,
            gt_difficulty=gt_difficulty
        )

        return anns_results

    def evaluate(self, results, **kwargs):
        '''
        Evaluate model results.

        Args:
            results: list of results from the model.
        
        Returns:
            metrics: evaluation metrics for the results.
        '''
        metrics = {}

        for mode in self.cooperation_modes:
            print(f'\n{"=" * 40}')
            print(f'Evaluating cooperation mode: {mode.upper()}')
            print(f'{"=" * 40}')

            metrics[mode] = {}

            mode_results = [res[mode] for res in results if mode in res]

            # Evaluate BEV map segmentation results.
            if 'masks_bev' in mode_results[0]:
                metrics[mode].update(self.evaluate_map(mode_results))

            # Evaluate 3D object detection results.
            if 'boxes_3d' in mode_results[0]:
                simbev2x_eval = SimBEVDetectionEval(mode_results, self.object_classes, self.eval_mode, self.point_cloud_range)

                metrics[mode].update(simbev2x_eval.evaluate())
        
        return metrics
