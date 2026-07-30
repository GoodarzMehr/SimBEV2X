# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.colors import LogNorm
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


argparser = argparse.ArgumentParser(description='SimBEV2X BEV heatmap visualization script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--output', default='bev', help='path to the output directory for plots (default: bev)')
argparser.add_argument('--workers', type=int, default=16, help='number of parallel workers (default: 16)')


def _process_batch(paths):
    result = None

    for path in paths:
        grid = np.load(path)['data'].astype(np.uint8)
        
        result = grid if result is None else result + grid

    return result


def main():
    args = argparser.parse_args()

    os.makedirs(f'{args.path}/simbev2x/stats/{args.output}', exist_ok=True)

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

    aggregate = None
    batch_size = 64
    batches = [paths[i:min(i + batch_size, len(paths) - 1)] for i in range(0, len(paths), batch_size)]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_batch, batch): batch for batch in batches}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing BEV grids', ncols=120):
            aggregate = future.result().astype(np.uint64) if aggregate is None else aggregate + future.result()

    for i, cls in enumerate(BEV_CLASSES):
        heatmap = aggregate[i]

        if heatmap.sum() == 0:
            continue

        heatmap[heatmap == 0] = 1

        if cls == 'car':
            heatmap[193:207, 197:203] = 1
        
        fig, ax = plt.subplots(figsize=(1, 1))

        im = ax.imshow(heatmap, origin='upper', norm=LogNorm(vmin=heatmap.min(), vmax=heatmap.max()), cmap='rainbow')

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        out_path = os.path.join(f'{args.path}/simbev2x/stats/{args.output}', f'{cls}.png')
        
        plt.savefig(out_path, dpi=520, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f'Saved: {out_path}')

    print(f'\nAll plots saved to {args.path}/simbev2x/stats/{args.output}/')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    main()
