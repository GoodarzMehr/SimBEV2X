# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


SIMBEV_PALETTE = {
    'road': (196, 80, 196),
    'car': (0, 128, 240),
    'truck': (128, 240, 64),
    'bus': (0, 144, 0),
    'motorcycle': (240, 240, 0),
    'bicycle': (0, 240, 240),
    'rider': (240, 144, 0),
    'pedestrian': (240, 0, 0),
    'traffic_light': (240, 160, 0),
    'traffic_sign': (240, 0, 128),
    'traffic_cone': (252, 180, 0),
    'barrier': (240, 128, 128)
}


argparser = argparse.ArgumentParser(description='SimBEV2X 3D bounding box points visualization script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--output', default='points', help='path to the output directory for plots (default: points)')
argparser.add_argument('--workers', type=int, default=16, help='number of parallel workers (default: 16)')


def _palette_color(cls):
    rgb = SIMBEV_PALETTE.get(cls, (128, 128, 128))
    
    return tuple(c / 255.0 for c in rgb)


def _process_file(task: tuple[str, str]):
    entity, gt_path = task

    if gt_path.endswith('.json'):
        with open(gt_path, 'r') as f:
            objects = json.load(f)
    else:
        objects = np.load(gt_path, allow_pickle=True)

    records = []

    for obj in objects:
        validity = obj['valid_flag']

        if validity:
            cls = obj['class']
            dist = obj['distance_to_ego']
            num_lidar_pts = obj['num_lidar_pts']
            num_radar_pts = obj['num_radar_pts']

            if cls is None or dist is None or num_lidar_pts is None or num_radar_pts is None:
                continue

            if num_lidar_pts + num_radar_pts == 0:
                continue

            records.append((cls, float(dist), int(num_lidar_pts), int(num_radar_pts)))

    return entity, records


def main():
    args = argparser.parse_args()

    os.makedirs(f'{args.path}/simbev2x/stats/{args.output}', exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
    })

    tasks = []
    data = defaultdict(list)

    for split in ['train', 'val', 'test']:
        info_path = f'{args.path}/simbev2x/infos/simbev2x_infos_{split}.json'

        if not os.path.exists(info_path):
            continue

        print(f'Loading {split} infos...')

        with open(info_path, 'r') as f:
            infos = json.load(f)

        for scene in infos['data']:
            for entity in infos['data'][scene]['scene_data']:
                for info in infos['data'][scene]['scene_data'][entity]:
                    if 'GT_DET' not in info:
                        continue

                    gt_path = info['GT_DET']

                    if not os.path.exists(gt_path):
                        print(f'Warning: file not found: {gt_path}')

                        continue

                    tasks.append((entity, gt_path))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing GT files', ncols=120):
            entity, records = future.result()

            for cls, dist, num_lidar_pts, num_radar_pts in records:
                data['class'].append(cls)
                data['distance'].append(dist)
                data['num_lidar_pts'].append(num_lidar_pts)
                data['num_radar_pts'].append(num_radar_pts)

    class_order = [
        'bus',
        'truck',
        'car',
        'pedestrian',
        'motorcycle',
        'traffic_sign',
        'bicycle',
        'traffic_light',
        'barrier',
        'traffic_cone',
    ]

    alpha = [1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    classes = np.array(data['class'])
    distances = np.array(data['distance'])
    
    for sensor in ['lidar', 'radar']:
        points = np.array(data[f'num_{sensor}_pts'])

        fig, ax = plt.subplots(figsize=(8, 8))

        sensor_mask = points > 0
        
        sensor_classes = classes[sensor_mask]
        sensor_distances = distances[sensor_mask]
        sensor_points = points[sensor_mask]

        for i, cls in enumerate(class_order):
            mask = sensor_classes == cls

            ax.scatter(
                sensor_distances[mask],
                sensor_points[mask],
                color=_palette_color(cls),
                s=0.05,
                linewidth=0,
                alpha=alpha[i],
                label=cls
            )
        
        ax.set_box_aspect(1)
        
        ax.set_xlim(0, 120)
        ax.set_ylim(1, 10000) if sensor == 'radar' else ax.set_ylim(1, 100000)
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.yaxis.set_tick_params(which='both', length=0)
        
        out_path = os.path.join(f'{args.path}/simbev2x/stats/{args.output}', f'{sensor}.png')
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f'Saved: {out_path}')

    print(f'\nAll plots saved to {args.output}/')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    main()
