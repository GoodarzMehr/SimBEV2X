# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import argparse

import numpy as np

from tqdm import tqdm


argparser = argparse.ArgumentParser(description='SimBEV2X infos statistics script.')

argparser.add_argument('--path', default='/dataset', help='path to the dataset (default: /dataset)')

args = argparser.parse_args()


def _weather_bin(value: float):
    if value < 10:
        return 0
    elif value < 40:
        return 1
    elif value < 70:
        return 2
    else:
        return 3

def _sun_bin(value: float):
    if value < 0:
        return 0
    elif value < 6:
        return 1
    else:
        return 2

def _print_distribution(values: list, interval: int = 10):
    if not values:
        print('  No data.')
        
        return

    max_val = max(values)
    min_val = min(values)
    
    num_bins = ((max_val - min_val) // interval) + 1
    
    bins = [0] * num_bins

    for v in values:
        bins[(v - min_val) // interval] += 1

    for i, b in enumerate(bins):
        lo = i * interval + min_val
        hi = lo + interval - 1
        
        print(f'  {lo:>6}-{hi:<6}: {b}')

def _print_matrix(matrix: np.ndarray, row_labels: list, col_labels: list):
    col_width = max(len(l) for l in col_labels + row_labels) + 2
    
    header = ' ' * col_width + ''.join(f'{l:>{col_width}}' for l in col_labels)
    
    print(header)

    for i, row_label in enumerate(row_labels):
        row = f'{row_label:<{col_width}}' + ''.join(f'{matrix[i][j]:>{col_width}.4f}' for j in range(len(col_labels)))
        
        print(row)

def main():
    vehicles_per_scene = []
    rsus_per_scene = []
    scene_durations = []
    spawned_vehicles = []
    spawned_walkers = []
    accident_hazards = []
    road_work_hazards = []

    weather_params = ['cloudiness', 'precipitation', 'fog_density']
    weather_labels = ['none (<10)', 'low (10-40)', 'moderate (40-70)', 'heavy (70-100)']
    weather_matrices = {p: np.zeros((4, 4)) for p in weather_params}

    sun_labels = ['night (-90-0)', 'dawn/dusk (0-6)', 'day (6-90)']
    sun_matrix = np.zeros((3, 3))

    for split in ['train', 'val', 'test']:
        info_path = f'{args.path}/simbev2x/infos/simbev2x_infos_{split}.json'

        if not os.path.exists(info_path):
            continue

        print(f'Loading {split} infos...')

        with open(info_path, 'r') as f:
            infos = json.load(f)

        scene_pbar = tqdm(infos['data'], desc=f'Processing {split} infos', ncols=120)

        for scene in scene_pbar:
            scene_info = infos['data'][scene].get('scene_info', {})

            n_vehicle = sum(1 for k in scene_info if k.startswith('vehicle_'))
            n_rsu = sum(1 for k in scene_info if k.startswith('rsu_'))
            
            vehicles_per_scene.append(n_vehicle)
            rsus_per_scene.append(n_rsu)

            if 'scene_duration' in scene_info:
                scene_durations.append(scene_info['scene_duration'])

            if 'n_vehicles' in scene_info:
                spawned_vehicles.append(scene_info['n_vehicles'])

            if 'n_walkers' in scene_info:
                spawned_walkers.append(scene_info['n_walkers'])

            if 'n_accident_hazards' in scene_info:
                accident_hazards.append(scene_info['n_accident_hazards'])

            if 'n_road_work_hazards' in scene_info:
                road_work_hazards.append(scene_info['n_road_work_hazards'])

            initial_wp = scene_info.get('initial_weather_parameters', {})
            final_wp = scene_info.get('final_weather_parameters', initial_wp)

            for p in weather_params:
                if p in initial_wp and p in final_wp:
                    weather_matrices[p][_weather_bin(initial_wp[p])][_weather_bin(final_wp[p])] += 1

            if 'sun_altitude_angle' in initial_wp and 'sun_altitude_angle' in final_wp:
                sun_matrix[_sun_bin(initial_wp['sun_altitude_angle'])][_sun_bin(final_wp['sun_altitude_angle'])] += 1

    print('\n--- Data Collection Vehicles per Scene ---')
    _print_distribution(vehicles_per_scene, interval=1)

    print('\n--- RSUs per Scene ---')
    _print_distribution(rsus_per_scene, interval=1)

    print('\n--- Scene Duration Distribution (seconds) ---')
    _print_distribution([int(d) for d in scene_durations], interval=1)

    print('\n--- Spawned Vehicles per Scene ---')
    _print_distribution(spawned_vehicles, interval=20)

    print('\n--- Spawned Pedestrians per Scene ---')
    _print_distribution(spawned_walkers, interval=20)

    print('\n--- Accident Hazards per Scene ---')
    _print_distribution(accident_hazards, interval=1)

    print('\n--- Road Work Hazards per Scene ---')
    _print_distribution(road_work_hazards, interval=1)

    for p in weather_params:
        print(f'\n--- Weather Transition Matrix: {p} ---')
        
        _print_matrix(weather_matrices[p] / weather_matrices[p].sum(), weather_labels, weather_labels)

    print('\n--- Weather Transition Matrix: sun_altitude_angle ---')
    
    _print_matrix(sun_matrix / sun_matrix.sum(), sun_labels, sun_labels)


if __name__ == '__main__':
    main()