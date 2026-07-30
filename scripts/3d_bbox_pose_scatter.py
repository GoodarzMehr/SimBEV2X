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


argparser = argparse.ArgumentParser(description='SimBEV2X 3D bounding box pose visualization script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')
argparser.add_argument('--output', default='pose', help='path to the output directory for plots (default: pose)')
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
            angle = obj['angle_to_ego']

            if cls is None or dist is None or angle is None:
                continue

            records.append((cls, float(dist), float(angle)))

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

    entity_data = defaultdict(lambda: {'class': [], 'distance': [], 'angle': []})

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing GT files', ncols=120):
            entity, records = future.result()

            for cls, dist, angle in records:
                entity_data[entity]['class'].append(cls)
                entity_data[entity]['distance'].append(dist)
                entity_data[entity]['angle'].append(angle)

    degree_labels = [
        '$0^\\circ$',
        '$45^\\circ$',
        '$90^\\circ$',
        '$135^\\circ$  ',
        '$180^\\circ$',
        ' $225^\\circ$',
        ' $270^\\circ$',
        '$315^\\circ$'
    ]
    
    class_order = [
        'pedestrian',
        'car',
        'truck',
        'motorcycle',
        'bicycle',
        'bus',
        'traffic_cone',
        'barrier',
        'traffic_sign',
        'traffic_light'
    ]

    alpha = [1.0, 0.8, 0.4, 1.0, 1.0, 1.0, 0.4, 1.0, 0.2, 0.2]
    
    for entity, data in entity_data.items():
        classes = np.array(data['class'])
        distances = np.array(data['distance'])
        angles = np.array(data['angle'])

        if len(classes) == 0:
            continue

        # Convert degrees to radians if needed.
        if np.any(np.abs(angles) > 2 * np.pi):
            angles = np.deg2rad(angles)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        ax.set_theta_zero_location('N')
        
        ax.set_rmax(200)
        ax.set_rlim(0, 200)
        ax.set_rlabel_position(247.5)
        ax.set_rticks([50, 100, 150, 200])
        
        ax.set_yticklabels([' $50$ m', ' $100$ m', ' $150$ m', ' $200$ m'])
        
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=degree_labels)

        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.grid(True, color='dimgray', linewidth=0.8, alpha=1.0)

        for i, cls in enumerate(class_order):
            mask = classes == cls

            ax.scatter(
                angles[mask],
                distances[mask],
                color=_palette_color(cls),
                s=0.05,
                linewidth=0,
                alpha=alpha[i],
                label=cls
            )

        safe_name = entity.replace('/', '_').replace(' ', '_').replace(':', '_')
        out_path = os.path.join(f'{args.path}/simbev2x/stats/{args.output}', f'{safe_name}.png')
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        print(f'Saved: {out_path}')

        for i, cls in enumerate(class_order):
            mask = classes == cls

            if mask.sum() == 0:
                continue

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

            ax.set_theta_zero_location('N')
            
            ax.set_rmax(200)
            ax.set_rlim(0, 200)
            ax.set_rlabel_position(247.5)
            ax.set_rticks([50, 100, 150, 200])

            ax.set_yticklabels([' $50$ m', ' $100$ m', ' $150$ m', ' $200$ m'])
            
            ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=degree_labels)

            ax.tick_params(axis='both', which='major', labelsize=16)
        
            ax.grid(True, color='dimgray', linewidth=0.8, alpha=1.0)

            ax.scatter(angles[mask], distances[mask], color=_palette_color(cls), s=0.2, linewidth=0, label=cls)

            safe_name = entity.replace('/', '_').replace(' ', '_').replace(':', '_')
            out_path = os.path.join(f'{args.path}/simbev2x/stats/{args.output}', f'{safe_name}_{cls}.png')
            
            plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)

            print(f'Saved: {out_path}')

    print(f'\nAll plots saved to {args.output}/')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    main()
