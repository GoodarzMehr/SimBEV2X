# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that manages the calculations for obtaining the BEV ground truth, 3D
object bounding boxes, and HD map information.
'''

import cv2
import json
import carla
import logging

import numpy as np

from simbev.ground_truth_manager import GTManager

logger = logging.getLogger(__name__)


class GTManagerV2X(GTManager):
    '''
    The Ground Truth Manager V2X manages the calculations for obtaining the
    BEV ground truth, 3D object bounding boxes, and HD map information.

    Args:
        config: dictionary of configuration parameters.
        world: CARLA world.
        entity: data collection entity.
        sensor_manager: data collection entity's Sensor Manager.
        map_name: name of the CARLA map.
        idx: index of the entity.
        entity_type: type of the entity ('vehicle' or 'rsu').
    '''
    def __init__(
        self,
        config: dict,
        world: carla.World,
        vehicle: carla.Vehicle,
        sensor_manager,
        map_name: str,
        idx: int,
        entity_type: str = 'vehicle'
    ):
        super().__init__(config, world, vehicle, sensor_manager, map_name)

        self._idx = idx
        self._entity_type = entity_type
    
    def render(self):
        '''Render the BEV ground truth.'''
        name = 'Vehicle' if self._entity_type == 'vehicle' else 'RSU'
        
        cv2.imshow(f'{name} {self._idx} Ground Truth', self._canvas)
        cv2.waitKey(1)
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save the ground truth.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(
            f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/seg' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_SEG.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self._bev_gt)

        cv2.imwrite(
            f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/seg_viz' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_SEG_VIZ.jpg',
            self._canvas
        )
        
        actors = self.get_bounding_boxes()

        with open(
            f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/det' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-GT_DET.bin',
            'wb'
        ) as f:
            np.save(f, np.array(actors), allow_pickle=True)

        hd_map_info = self.get_hd_map_info()

        with open(
            f'{path}/simbev2x/ground-truth/{self._entity_type}-{self._idx}/hd_map' \
                f'/SimBEV2X-scene-{scene:04d}-frame-{frame:04d}-{self._entity_type}-{self._idx:04d}-HD_MAP.json',
            'w'
        ) as f:
            json.dump(hd_map_info, f, indent=4)