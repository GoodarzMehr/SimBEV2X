# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


argparser = argparse.ArgumentParser(description='SimBEV2X 3D bounding box wheelchair statistics script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--workers', type=int, default=12, help='number of parallel workers (default: 12)')


def _process_file(gt_path: str):
    if gt_path.endswith('.json'):
        with open(gt_path, 'r') as f:
            objects = json.load(f)
    else:
        objects = np.load(gt_path, allow_pickle=True)

    records = []

    for obj in objects:
        if obj['class'] == 'pedestrian':
            records.append((obj['valid_flag'], False if obj['attributes']['use_wheelchair'] == 'false' else True))

    return gt_path, records


def main():
    args = argparser.parse_args()
    
    seen_files = set()

    num_pedestrians = 0
    num_wheelchair_users = 0
    num_valid_pedestrians = 0
    num_valid_wheelchair_users = 0

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
                    if 'GT_DET_COMBINED' not in info:
                        continue

                    gt_path = info['GT_DET_COMBINED']

                    if gt_path in seen_files:
                        continue

                    if not os.path.exists(gt_path):
                        print(f'Warning: file not found: {gt_path}')
                        
                        continue
                    
                    seen_files.add(gt_path)
        
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, p): p for p in seen_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing GT files', ncols=120):
            gt_path, records = future.result()

            for validity, use_wheelchair in records:
                num_pedestrians += 1

                if use_wheelchair:
                    num_wheelchair_users += 1

                if validity:
                    num_valid_pedestrians += 1

                    if use_wheelchair:
                        num_valid_wheelchair_users += 1
    
    print(f'Total pedestrians: {num_pedestrians}')
    print(f'Total wheelchair users: {num_wheelchair_users}')
    print(f'Total valid pedestrians: {num_valid_pedestrians}')
    print(f'Total valid wheelchair users: {num_valid_wheelchair_users}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    main()
