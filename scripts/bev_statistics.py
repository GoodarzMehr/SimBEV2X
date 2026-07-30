# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


BEV_CLASSES = [
    'road',
    'hazard',
    'road_line',
    'sidewalk',
    'crosswalk',
    'traffic_cone',
    'barrier',
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'rider',
    'pedestrian'
]


argparser = argparse.ArgumentParser(description='SimBEV2X BEV statistics script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--workers', type=int, default=24, help='number of parallel workers (default: 24)')


def _process_file(path: str):
    grid = np.load(path)['data']

    return grid.sum(axis=(1, 2)).astype(np.int64)


def main():
    args = argparser.parse_args()

    class_counts = np.zeros(len(BEV_CLASSES), dtype=np.int64)

    paths = []

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
                    if 'GT_SEG' not in info:
                        continue

                    gt_path = info['GT_SEG']

                    if not os.path.exists(gt_path):
                        print(f'Warning: file not found: {gt_path}')

                        continue
                    
                    paths.append(gt_path)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, p): p for p in paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing BEV grids', ncols=120):
            class_counts += future.result()

    print('\n--- BEV Label Statistics ---')
    print(f'{"Class":<20} {"Count":>15}')
    print('-' * 38)

    for i, cls in enumerate(BEV_CLASSES):
        print(f'{cls:<20} {class_counts[i]:>15}')

    print('-' * 38)
    print(f'{"Total":<20} {class_counts.sum():>15}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    main()
