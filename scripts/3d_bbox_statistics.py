# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


argparser = argparse.ArgumentParser(description='SimBEV2X 3D bounding box statistics script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--workers', type=int, default=12, help='number of parallel workers (default: 12)')


def _process_file(gt_path: str):
    if gt_path.endswith('.json'):
        with open(gt_path, 'r') as f:
            objects = json.load(f)
    else:
        objects = np.load(gt_path, allow_pickle=True)

    records = []

    per_frame_valid = defaultdict(int)

    for obj in objects:
        cls = obj['class']
        validity = obj['valid_flag']
        
        if cls is None:
            continue

        yaw = None
        speed = None
        yaw_rate = None
        
        if validity:
            per_frame_valid[cls] += 1
            
            yaw = obj['rotation'][2] if obj['rotation'] is not None else None
            speed = np.linalg.norm(obj['linear_velocity']) if obj['linear_velocity'] is not None else None
            yaw_rate = obj['angular_velocity'][2] if obj['angular_velocity'] is not None else None

        records.append((cls, validity, yaw, speed, yaw_rate))

    return gt_path, records, per_frame_valid


def _print_histogram(
        values: list,
        interval: int,
        max_val: int,
        min_val: int = 0,
        overflow_label: str = None,
        underflow_label: str = None
    ):
    bins = [0] * ((max_val - min_val) // interval)
    
    overflow = 0
    underflow = 0

    for v in values:
        if v < min_val:
            underflow += 1
        elif v >= max_val:
            overflow += 1
        else:
            bins[int((v - min_val) // interval)] += 1
    
    if underflow_label is not None:
        print(f' {underflow_label:>11}   : {underflow}')
    
    for i, b in enumerate(bins):
        low = min_val + i * interval
        high = low + interval - 1
        
        print(f'    {low:>5}-{high:<5}: {b}')
    
    if overflow_label is not None:
        print(f' {overflow_label:>11}   : {overflow}')


def main():
    args = argparser.parse_args()
    
    total = 0
    valid_total = 0
    seen_files = set()
    class_counts = defaultdict(int)
    valid_class_counts = defaultdict(int)
    frame_valid_counts = defaultdict(lambda: defaultdict(int))
    yaw_values = defaultdict(list)
    speed_values = defaultdict(list)
    yaw_rate_values = defaultdict(list)

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
            gt_path, records, per_frame_valid = future.result()

            for cls, validity, yaw, speed, yaw_rate in records:
                class_counts[cls] += 1
                total += 1

                if validity:
                    valid_class_counts[cls] += 1
                    valid_total += 1

                    if yaw is not None:
                        yaw_values[cls].append(yaw)

                    if speed is not None:
                        speed_values[cls].append(speed)

                    if yaw_rate is not None:
                        yaw_rate_values[cls].append(yaw_rate)

            for cls, count in per_frame_valid.items():
                frame_valid_counts[cls][gt_path] = count

            for cls in class_counts:
                if cls not in per_frame_valid:
                    frame_valid_counts[cls][gt_path] = 0

    print('\n--- 3D Bounding Box Statistics ---')
    print(f'{"Class":<20} {"Count":>10} {"Valid Count":>10}')
    print('-' * 45)

    for cls in sorted(class_counts):
        print(f'{cls:<20} {class_counts[cls]:>10} {valid_class_counts[cls]:>10}')

    print('-' * 45)
    print(f'{"Total":<20} {total:>10} {valid_total:>10}')

    print('\n--- Valid Object Distribution per Frame ---')

    for cls in sorted(frame_valid_counts):
        counts = list(frame_valid_counts[cls].values())

        if not counts:
            continue

        print(f'\n  {cls}:')
        
        _print_histogram(counts, 5, 100, 0, '100+')

    print('\n--- Yaw Distribution (degrees) ---')

    for cls in sorted(yaw_values):
        vals = yaw_values[cls]

        if not vals:
            continue
        
        print(f'\n  {cls}:')

        _print_histogram(vals, 10, 180, -180)

    moving_objects = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
    
    print('\n--- Speed Distribution (m/s) ---')

    for cls in sorted(speed_values):
        if cls in moving_objects:
            vals = speed_values[cls]

            if not vals:
                continue

            print(f'\n  {cls}:')

            _print_histogram(vals, 1, 40, 0, '40+')

    print('\n--- Yaw Rate Distribution (deg/s) ---')

    for cls in sorted(yaw_rate_values):
        if cls in moving_objects:
            vals = yaw_rate_values[cls]

            if not vals:
                continue

            print(f'\n  {cls}:')

            _print_histogram(vals, 5, 50, -50, '50+', '-50-')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    main()
