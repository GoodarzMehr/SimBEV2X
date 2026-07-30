# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import time
import argparse
import traceback

from tqdm import tqdm

from .visualization_handlers import *
from .visualization_interactive import *

from simbev_tools.visualization_handlers import (
    visualize_lidar,
    visualize_lidar3d,
    visualize_semantic_lidar,
    visualize_semantic_lidar3d
)


VISUALIZATION_MODES = {
    'rgb': {
        'handler': visualize_rgb,
        'output_dirs': [f'RGB-{camera}' for camera in CAM_NAME],
        'color': '#FF00FF'
    },
    'depth': {
        'handler': visualize_depth,
        'output_dirs': [f'DPT-{camera}' for camera in CAM_NAME],
        'color': '#FF7700'
    },
    'flow': {
        'handler': visualize_flow,
        'output_dirs': [f'FLW-{camera}' for camera in CAM_NAME],
        'color': '#FFFF00'
    },
    'lidar': {
        'handler': visualize_lidar,
        'output_dirs': [f'LIDAR-{distance}' for distance in VIEWS.keys()],
        'color': '#99FF00'
    },
    'lidar-with-bbox': {
        'handler': visualize_lidar_with_bbox,
        'output_dirs': [f'LIDARwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#55FF00'
    },
    'lidar3d': {
        'handler': visualize_lidar3d,
        'output_dirs': [f'LIDAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#00FF99'
    },
    'lidar3d-with-bbox': {
        'handler': visualize_lidar3d_with_bbox,
        'output_dirs': [f'LIDAR3DwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#00FF55'
    },
    'semantic-lidar': {
        'handler': visualize_semantic_lidar,
        'output_dirs': [f'SEG-LIDAR-{distance}' for distance in VIEWS.keys()],
        'color': '#0099FF'
    },
    'semantic-lidar3d': {
        'handler': visualize_semantic_lidar3d,
        'output_dirs': [f'SEG-LIDAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#0055FF'
    },
    'radar': {
        'handler': visualize_radar,
        'output_dirs': [f'RADAR-{distance}' for distance in VIEWS.keys()],
        'color': '#9900FF'
    },
    'radar-with-bbox': {
        'handler': visualize_radar_with_bbox,
        'output_dirs': [f'RADARwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#5500FF'
    },
    'radar3d': {
        'handler': visualize_radar3d,
        'output_dirs': [f'RADAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#FF0099'
    },
    'radar3d-with-bbox': {
        'handler': visualize_radar3d_with_bbox,
        'output_dirs': [f'RADAR3DwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#FF0055'
    },
    'voxel3d': {
        'handler': visualize_voxel3d,
        'output_dirs': [f'VOXEL3D-{distance}' for distance in VIEWS.keys()],
        'color': '#FF6600'
    },
    'interactive-single': {
        'handler': visualize_interactive_single,
        'output_dirs': [],
        'color': '#00AAFF'
    },
    'interactive-multi': {
        'handler': visualize_interactive_multi,
        'output_dirs': [],
        'color': '#009900'
    }
}


argparser = argparse.ArgumentParser(description='SimBEV visualization tool.')

argparser.add_argument(
    'mode',
    nargs='+',
    help='visualization mode (all, rgb, depth, flow, lidar, lidar3d, lidar-with-bbox, lidar3d-with-bbox, '
        'semantic-lidar, semantic-lidar3d, radar, radar3d, radar-with-bbox, radar3d-with-bbox)'
)
argparser.add_argument(
    '--path',
    default='/dataset',
    help='path to the dataset (default: /dataset)'
)
argparser.add_argument(
    '-s', '--scene',
    nargs='+',
    default=['-1'],
    help='scene number range(s) (e.g. 2 3 4-6 5 8-12) (default: -1, i.e. all scenes)'
)
argparser.add_argument(
    '-f', '--frame',
    nargs='+',
    default=['-1'],
    help='frame number range(s) (e.g. 2 3 4-6 5 8-12) (default: -1, i.e. all frames)'
)
argparser.add_argument(
    '--trim-step',
    type=int,
    default=1,
    help='step size for trimming point clouds (default: 1, i.e. no trimming)'
)
argparser.add_argument(
    '--ignore-valid-flag',
    action='store_true',
    help='ignore valid_flag when rendering bounding boxes'
)
argparser.add_argument(
    '--filled-voxels',
    action='store_true',
    help='use filled voxel grids instead of standard voxel grids'
)

args = argparser.parse_args()


def setup_output_directories(path: str, mode, max_num: dict):
    '''
    Create the output directories for the given mode.
    
    Args:
        path: root directory of the dataset.
        mode: visualization mode.
        max_num: maximum number of entities (vehicles and rsus).
    '''
    for entity in ['vehicle', 'rsu']:
        for i in range(max_num[entity]):
            if entity == 'rsu' and mode in ['rgb', 'depth', 'flow']:
                abbrev = 'RGB' if mode == 'rgb' else 'DPT' if mode == 'depth' else 'FLW'
                
                VISUALIZATION_MODES[mode]['output_dirs'] = [f'{abbrev}-{camera}' for camera in RSU_CAM_NAME]
            
            for name in VISUALIZATION_MODES[mode]['output_dirs']:
                os.makedirs(f'{path}/simbev2x/viz/{entity}-{i}/{name}', exist_ok=True)

def main(mode, path: str, max_num: dict):
    try:
        if mode not in VISUALIZATION_MODES:
            print(f'Warning: unknown mode "{mode}", skipping.')
            
            return
        
        if 'interactive' not in mode:
            setup_output_directories(path, mode, max_num)
        
        handler = VISUALIZATION_MODES[mode]['handler']
        
        if 'interactive' in mode:
            ctx = VisualizationContextV2X(
                path,
                scene_number=None,
                frame_number=None,
                entity_name=None,
                entity_num=None,
                frame_data=None,
                metadata=None,
                ignore_valid_flag=args.ignore_valid_flag,
                filled_voxels=args.filled_voxels,
                trim_step=args.trim_step
            )
            
            handler(ctx)
            
            return
        
        for split in ['train', 'val', 'test']:
            info_path = f'{path}/simbev2x/infos/simbev2x_infos_{split}.json'
            
            if not os.path.exists(info_path):
                continue
                
            with open(info_path, 'r') as f:
                infos = json.load(f)

            metadata = infos['metadata']

            # Get the list of scenes to visualize.
            if args.scene == ['-1']:
                scene_list = [int(scene.split('_')[1]) for scene in infos['data']]
            else:
                scene_list = parse_range_argument(args.scene)

            scene_pbar = tqdm(
                scene_list,
                desc=f'Visualizing {mode}',
                ncols=120,
                colour=VISUALIZATION_MODES[mode]['color']
            )
            
            for scene_number in scene_pbar:
                scene_key = f'scene_{scene_number:04d}'
                
                if scene_key not in infos['data']:
                    continue

                scene_data = infos['data'][scene_key]['scene_data']

                # Get the list of frames to visualize.
                if args.frame == ['-1']:
                    frame_list = list(range(len(scene_data['vehicle_0'])))
                else:
                    requested_frames = parse_range_argument(args.frame)

                    # Filter to find valid frame numbers.
                    frame_list = [f for f in requested_frames if 0 <= f < len(scene_data['vehicle_0'])]
                    
                    # Warn about invalid frames
                    invalid_frames = [f for f in requested_frames if f < 0 or f >= len(scene_data['vehicle_0'])]
                    
                    if invalid_frames:
                        print(f'Warning: Scene {scene_number} has only {len(scene_data["vehicle_0"])} frames. '
                              f'Skipping invalid frames: {invalid_frames}')
                
                entity_pbar = tqdm(
                    scene_data,
                    desc=f'{" " * (len(mode) + 2)}Scene {scene_number:04d}',
                    ncols=120,
                    colour="#AA88FF",
                    leave=False
                )

                for entity in entity_pbar:
                    entity_name = entity.split('_')[0]
                    entity_num = int(entity.split('_')[1])

                    name = 'RSU' if entity_name == 'rsu' else 'Vehicle'
                    margin = 4 if entity_name == 'rsu' else 0
                
                    frame_pbar = tqdm(
                        frame_list,
                        desc=f'{" " * (len(mode) + margin)}{name} {entity_num:04d}',
                        ncols=120,
                        colour='#0099FF',
                        leave=False
                    )

                    for frame_number in frame_pbar:
                        frame_data = scene_data[entity][frame_number]
                        
                        # Create the context.
                        ctx = VisualizationContextV2X(
                            path,
                            scene_number,
                            frame_number,
                            entity_name,
                            entity_num,
                            frame_data,
                            metadata,
                            args.ignore_valid_flag
                        )

                        # Call the handler.
                        handler(ctx)
    
    except Exception:
        print(traceback.format_exc())
        
        print('Killing the process...')
        
        time.sleep(3.0)


def entry():
    try:
        start = time.perf_counter()

        # Determine modes to process
        if 'all' in args.mode:
            mode_list = [mode for mode in VISUALIZATION_MODES.keys() if 'interactive' not in mode]
        else:
            mode_list = args.mode

        if not all('interactive' in mode for mode in mode_list):
            os.makedirs(f'{args.path}/simbev2x/viz', exist_ok=True)

        max_num = {'vehicle': 0, 'rsu': 0}

        for entity in ['vehicle', 'rsu']:
            while os.path.exists(f'{args.path}/simbev2x/ground-truth/{entity}-{max_num[entity]}'):
                max_num[entity] += 1

            if not all('interactive' in mode for mode in mode_list):
                for i in range(max_num[entity]):
                    os.makedirs(f'{args.path}/simbev2x/viz/{entity}-{i}', exist_ok=True)

        # Process each mode
        for mode in mode_list:
            main(mode, args.path, max_num)
        
        end = time.perf_counter()

        print(f'Visualization completed in {end - start:.3f} seconds.')
    
    except KeyboardInterrupt:
        print('Killing the process...')
        
        time.sleep(3.0)
    
    finally:
        print('Done.')


if __name__ == '__main__':
    entry()