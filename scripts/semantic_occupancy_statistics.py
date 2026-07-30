# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


VOXEL_CLASSES = [
    'None',
    'Road',
    'Sidewalk',
    'Building',
    'Wall',
    'Fence',
    'Pole',
    'TrafficLight',
    'TrafficSign',
    'Vegetation',
    'Terrain',
    'Sky',
    'Pedestrian',
    'Rider',
    'Car',
    'Truck',
    'Bus',
    'Train',
    'Motorcycle',
    'Bicycle',
    'Static',
    'Dynamic',
    'Other',
    'Water',
    'RoadLine',
    'Ground',
    'Bridge',
    'RailTrack',
    'GuardRail',
    'Rock',
    'TrafficCone',
    'Barrier'
]


argparser = argparse.ArgumentParser(description='SimBEV2X semantic occupancy statistics script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--workers', type=int, default=24, help='number of parallel workers (default: 24)')

args = argparser.parse_args()


def _process_file(gt_path: str):
    grid = np.load(gt_path)['data']

    return np.bincount(grid.ravel(), minlength=len(VOXEL_CLASSES)).astype(np.int64)

def _count_voxels(infos: dict, key: str, desc: str = 'Processing'):
    class_counts = np.zeros(len(VOXEL_CLASSES), dtype=np.int64)

    paths = []

    for scene in infos['data']:
        for entity in infos['data'][scene]['scene_data']:
            for info in infos['data'][scene]['scene_data'][entity]:
                if key not in info:
                    continue

                gt_path = info[key]

                if not os.path.exists(gt_path):
                    print(f'Warning: file not found: {gt_path}')

                    continue

                paths.append(gt_path)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, p): p for p in paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, ncols=120):
            counts = future.result()
            class_counts += counts[:len(VOXEL_CLASSES)]

    return class_counts

def _print_counts(title: str, class_counts: np.ndarray):
    print(f'\n--- {title} ---')
    print(f'{"Class":<20} {"Count":>15}')
    print('-' * 38)

    for i, cls in enumerate(VOXEL_CLASSES):
        if class_counts[i] > 0:
            print(f'{cls:<20} {class_counts[i]:>15}')

    print('-' * 38)
    print(f'{"Total":<20} {class_counts.sum():>15}')


def main():
    voxel_counts = np.zeros(len(VOXEL_CLASSES), dtype=np.int64)
    filled_counts = np.zeros(len(VOXEL_CLASSES), dtype=np.int64)

    for split in ['train', 'val', 'test']:
        info_path = f'{args.path}/simbev2x/infos/simbev2x_infos_{split}.json'

        if not os.path.exists(info_path):
            continue

        print(f'Loading {split} infos...')

        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        voxel_counts += _count_voxels(infos, 'VOXEL-GRID', desc=f'Processing {split} voxel grids')
        filled_counts += _count_voxels(infos, 'VOXEL-GRID-FILLED', desc=f'Processing {split} filled voxel grids')

    _print_counts('Semantic Occupancy Statistics (VOXEL-GRID)', voxel_counts)
    _print_counts('Semantic Occupancy Statistics (VOXEL-GRID-FILLED)', filled_counts)


if __name__ == '__main__':
    main()
