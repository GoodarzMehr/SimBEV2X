# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that manages the CARLA world, performing functions such as loading the
map, controlling the Scenario and Vehicle Managers, and stepping through the
simulation.
'''

import time
import carla
import random
import logging

import numpy as np

from simbev.utils import is_used, kill_all_servers
from simbev.world_manager import WorldManager

try:
    from .rsu_manager import RSUManagerV2X
    from .vehicle_manager import VehicleManagerV2X
    from .scenario_manager import ScenarioManagerV2X

except ImportError:
    from rsu_manager import RSUManagerV2X
    from vehicle_manager import VehicleManagerV2X
    from scenario_manager import ScenarioManagerV2X


logger = logging.getLogger(__name__)


class WorldManagerV2X(WorldManager):
    '''
    The World Manager V2X manages the CARLA world, performing functions such
    as loading the map, controlling the Scenario and Vehicle Managers, and
    stepping through the simulation.

    Args:
        config: dictionary of configuration parameters.
        client: CARLA client.
        server_port: port number of the CARLA server.
    '''
    def __init__(self, config: dict, client: carla.Client, server_port: int):
        super().__init__(config, client, server_port)
    
    def set_scene_counter(self, counter: int):
        '''
        Set scene counter.

        Args:
            counter: scene counter.
        '''
        self._scene_counter = counter
    
    def set_path(self, path: str):
        '''
        Set dataset root directory.

        Args:
            path: dataset root directory.
        '''
        self._dataset_path = path
    
    def load_map(self, map_name: str):
        '''
        Load the desired map and apply the appropriate settings.

        Args:
            map_name: name of the map to load.
        '''
        logger.info(f'Loading {map_name}...')

        self._map_name = map_name
        
        if map_name == 'Town10HD':
            self._client.load_world('Town10HD_Opt')
        else:
            self._client.load_world(map_name)

        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._spectator = self._world.get_spectator()

        settings = self._world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._config['timestep']

        # If the selected map is Town12 or Town13 (large maps), limit tile
        # stream distance and actor active distance. If Town13 or Town15 are
        # selected, set max culling distance to 100.0, then revert back to 0.
        # This ensures faraway objects are rendered correctly.
        if map_name in ['Town12', 'Town13']:
            settings.tile_stream_distance = self._config['tile_stream_distance']
            settings.actor_active_distance = self._config['actor_active_distance']

        if map_name in ['Town13', 'Town15']:
            settings.max_culling_distance = 100.0

            self._world.apply_settings(settings)

            time.sleep(3.0)
        
        settings.max_culling_distance = 0.0

        self._world.apply_settings(settings)

        self._world.set_annotations_traverse_translucency(True)

        # Set up the Traffic Manager.
        logger.debug('Setting up the Traffic Manager...')

        self._tm_port = self._server_port // 10 + self._server_port % 10

        while is_used(self._tm_port):
            logger.warning(f'Traffic Manager port {self._tm_port} is already being used. Checking the next one...')
            self._tm_port += 1
        
        self._traffic_manager = self._client.get_trafficmanager(self._tm_port)
        self._traffic_manager.set_synchronous_mode(True)

        logger.debug(f'Traffic Manager is connected to port {self._tm_port}.')

        self._world.tick()

        logger.info(f'{map_name} loaded.')
    
        # Some objects obstruct the overhead or bottom-up view that is
        # necessary for the collection of accurate ground truth data, so they
        # are removed from the map.
        logger.debug(f'Removing objects obstructing the overhead or bottom-up view from {map_name}...')

        if map_name == 'Town02':
            obstructing = [
                'Floor_',
                'Vh_Car_AudiA2_'
            ]
        elif map_name == 'Town03':
            obstructing = [
                'SM_GasStation01',
                'SM_Mansion02',
                'Sassafras_04_LOD27',
                'Custom_pine_beech_02_LOD1',
                'Veg_Tree_AcerSaccharum_v19',
                'Veg_Tree_AcerSaccharum_v20',
                'Japanese_Maple_01_LOD10',
                'Japanese_Maple_01_LOD11',
                'Japanese_Maple_01_LOD14',
                'SM_T03_RailTrain02',
                'BP_RepSpline5_Inst_0_0',
                'BP_RepSpline5_Inst_2_2',
                'BP_RepSpline6_',
                'Road_Road_Town03_1_'
            ]
        elif map_name == 'Town04':
            obstructing = [
                'SideWalkCube',
                'SM_GasStation01'
            ]
        elif map_name == 'Town05':
            obstructing = [
                'Plane',
                'SM_Awning117'
            ]
        elif map_name == 'Town07':
            obstructing = [
                'Cube'
            ]
        elif map_name == 'Town10HD':
            # The first ones are for Town10HD, the second ones are for
            # Town10HD_Opt.
            # obstructing = [
            #     'SM_Tesla2',
            #     'SM_Tesla_2502',
            #     'SM_Mustang_prop2',
            #     'SM_Patrol2021Parked2',
            #     'SM_mercedescccParked2',
            #     'SM_LincolnMkz2017_prop',
            #     'Vh_Car_ToyotaPrius_NOrig',
            #     'InstancedFoliageActor_0_Inst_235_0',
            #     'InstancedFoliageActor_0_Inst_239_4',
            #     'InstancedFoliageActor_0_Inst_245_10',
            #     'InstancedFoliageActor_0_Inst_246_11',
            #     'InstancedFoliageActor_0_Inst_249_14',
            #     'InstancedFoliageActor_0_Inst_250_15',
            #     'InstancedFoliageActor_0_Inst_251_16',
            #     'InstancedFoliageActor_0_Inst_252_17',
            #     'InstancedFoliageActor_0_Inst_253_18',
            #     'InstancedFoliageActor_0_Inst_254_19',
            #     'InstancedFoliageActor_0_Inst_255_20',
            #     'InstancedFoliageActor_0_Inst_256_21',
            #     'InstancedFoliageActor_0_Inst_257_22',
            #     'InstancedFoliageActor_0_Inst_258_23',
            #     'InstancedFoliageActor_0_Inst_259_24',
            #     'InstancedFoliageActor_0_Inst_260_25',
            #     'InstancedFoliageActor_0_Inst_261_26',
            #     'InstancedFoliageActor_0_Inst_276_41',
            #     'InstancedFoliageActor_0_Inst_277_42'
            # ]
            obstructing = [
                'SM_Tesla2',
                'SM_Tesla_2502',
                'SM_Mustang_prop2',
                'SM_Patrol2021Parked2',
                'SM_mercedescccParked2',
                'SM_LincolnMkz2017_prop',
                'Vh_Car_ToyotaPrius_NOrig',
                'InstancedFoliageActor_0_Inst_12961_0',
                'InstancedFoliageActor_0_Inst_12965_4',
                'InstancedFoliageActor_0_Inst_12971_10',
                'InstancedFoliageActor_0_Inst_12972_11',
                'InstancedFoliageActor_0_Inst_12980_19',
                'InstancedFoliageActor_0_Inst_12981_20',
                'InstancedFoliageActor_0_Inst_12982_21',
                'InstancedFoliageActor_0_Inst_12983_22',
                'InstancedFoliageActor_0_Inst_13002_41',
                'InstancedFoliageActor_0_Inst_13003_42'
            ]
        else:
            obstructing = []

        self._bad_crosswalks = [
            'Road_Crosswalk_Town03_59_',
            'Road_Crosswalk_Town04_28_',
            'Road_Crosswalk_Town04_29_',
            'Road_Crosswalk_Town04_30_',
            'Road_Crosswalk_Town07_5_',
            'Road_Crosswalk_Town07_8_',
            'Road_Crosswalk_Town07_9_'
        ]
        
        self._objects = self._world.get_environment_objects()

        to_remove = [obj.id for obj in self._objects if any(x in obj.name for x in obstructing)]

        self._world.enable_environment_objects(set(to_remove), False)

        self._world.tick()

        logger.debug(f'Objects obstructing the overhead or bottom-up view were removed from {map_name}.')
        logger.debug('Generating waypoints...')

        self._waypoints = self._map.generate_waypoints(self._config['waypoint_distance'])

        self._crosswalks = self._map.get_crosswalks()

        self._world.tick()

        logger.debug('Waypoints generated.')
        logger.debug('Getting the Light Manager...')

        self._light_manager = self._world.get_lightmanager()

        logger.debug('Got the Light Manager.')
        logger.debug('Creating the Scenario Manager...')

        self._scenario_manager = ScenarioManagerV2X(
            self._config,
            self._client,
            self._world,
            self._traffic_manager,
            self._light_manager,
            map_name
        )

        logger.debug('Scenario Manager created.')
        logger.debug('Creating Vehicle Managers...')

        self._vehicle_managers = []

        for i in range(self._config['max_vehicles']):
            self._vehicle_managers.append(
                VehicleManagerV2X(self._config, self._world, self._traffic_manager, map_name, i)
            )

        logger.debug('Vehicle Managers created.')
        logger.debug('Creating RSU Managers...')

        self._rsu_managers = []

        for i in range(self._config['max_rsus']):
            self._rsu_managers.append(
                RSUManagerV2X(self._config, self._world, map_name, i)
            )

        logger.debug('RSU Managers created.')
        logger.debug('Getting vehicle spawn points...')

        wp_all = self._map.generate_waypoints(self._config['spawn_point_separation_distance'])

        # Filter out spawn points that are in a junction or within 4 meters of
        # a junction.
        wp_mid = [wp for wp in wp_all if wp.next(4.0) != []]

        sp_non_junction = [
            wp.transform for wp in wp_mid if (wp.is_junction is False and wp.next(4.0)[0].is_junction is False)
        ]

        self._available_spawn_points = []

        for sp in sp_non_junction:
            if all(sp.location.distance(osp.location) > 2.0 for osp in self._available_spawn_points):
                self._available_spawn_points.append(sp)

        for sp in self._available_spawn_points:
            sp.location.z += 0.4

        self._spawn_points_copy = self._available_spawn_points
        
        if 'data_collection_vehicle_spawn_points' in self._config:
            if self._map_name in self._config['data_collection_vehicle_spawn_points']:
                spawn_point_list = self._config['data_collection_vehicle_spawn_points'][self._map_name]

                self._available_spawn_points = []

                for sp in spawn_point_list:
                    location = carla.Location(x=sp[0][0], y=sp[0][1], z=sp[0][2])
                    rotation = carla.Rotation(roll=sp[1][0], pitch=sp[1][1], yaw=sp[1][2])

                    self._available_spawn_points.append(carla.Transform(location, rotation))
        
        logger.debug(f'{len(self._available_spawn_points)} vehicle spawn points available.')
        logger.debug(f'Getting map junctions with traffic lights...')

        self._junctions = []
        self._junction_ids = []
        
        wp_close_all = self._map.generate_waypoints(4.0)

        for wp in wp_close_all:
            if wp.is_junction and len(self._world.get_traffic_lights_in_junction(wp.junction_id)) > 0:
                if wp.junction_id not in self._junction_ids:
                    self._junctions.append(wp.get_junction())
                    self._junction_ids.append(wp.junction_id)
        
        logger.debug(f'Found {len(self._junctions)} junctions with traffic lights.')
    
    def spawn_vehicles(self):
        '''
        Prepare to spawn data collection vehicles at random spawn points.
        '''
        self._num_vehicles = random.randint(self._config['min_vehicles'], self._config['max_vehicles'])
        
        logger.info(f'Spawning {self._num_vehicles} data collection vehicles...')
        
        # Get data collection vehicle blueprints and spawn points.
        logger.debug('Getting vehicle blueprints...')

        bps = []

        for i in range(self._num_vehicles):
            bp = self._world.get_blueprint_library().filter(
                self._config['vehicle_list'][i % len(self._config['vehicle_list'])]
            )[0]

            if i == 0:
                bp.set_attribute('role_name', 'hero')
            else:
                bp.set_attribute('role_name', f'vehicle_{i}')
            
            bp.set_attribute(
                'color',
                self._config['vehicle_color_list'][i % len(self._config['vehicle_color_list'])]
            )

            bps.append(bp)
        
        logger.debug('Got vehicle blueprints.')
        
        if len(self._available_spawn_points) < self._num_vehicles:
            logger.warning(
                f'Only {len(self._available_spawn_points)} spawn points available, but {self._num_vehicles} vehicles'
                f' requested. Reducing the number of vehicles to {len(self._available_spawn_points)}.'
            )

            self._num_vehicles = len(self._available_spawn_points)

        self._center = None
        
        # Determine spawn points based on the selected spawn mode.
        if self._config['vehicle_spawn_mode'] == 'both':
            if random.random() < 0.5:
                self._num_vehicles, self._spawn_points, self._center = self._determine_clustered_spawn_points(
                    self._num_vehicles,
                    self._available_spawn_points
                )

                self._spawn_mode = 'clustered'
            else:
                self._num_vehicles, self._spawn_points = self._determine_chained_spawn_points(
                    self._num_vehicles,
                    self._available_spawn_points
                )

                self._spawn_mode = 'chained'
        elif self._config['vehicle_spawn_mode'] == 'clustered':
            self._num_vehicles, self._spawn_points, self._center = self._determine_clustered_spawn_points(
                self._num_vehicles,
                self._available_spawn_points
            )

            self._spawn_mode = 'clustered'
        elif self._config['vehicle_spawn_mode'] == 'chained':
            self._num_vehicles, self._spawn_points = self._determine_chained_spawn_points(
                self._num_vehicles,
                self._available_spawn_points
            )

            self._spawn_mode = 'chained'
        else:
            logger.error(
                f'Unknown vehicle spawn mode: {self._config["vehicle_spawn_mode"]}. Defaulting to "clustered".'
            )

            self._num_vehicles, self._spawn_points, self._center = self._determine_clustered_spawn_points(
                self._num_vehicles,
                self._available_spawn_points
            )

            self._spawn_mode = 'clustered'
        
        scene_info = {}

        scene_info['map'] = self._map_name

        for i in range(self._num_vehicles):
            scene_info[f'vehicle_{i}'] = self._vehicle_managers[i].spawn_vehicle(
                bps[i],
                self._spawn_points[i],
                self._tm_port
            )
        
        time.sleep(3.0)
        
        self._scenario_manager.scene_info = scene_info
    
    def _determine_clustered_spawn_points(
            self,
            num_vehicles: int,
            spawn_points: list[carla.Transform]
        ) -> tuple[int, list[carla.Transform]]:
        '''
        Determine clustered spawn points for the vehicles.

        Args:
            num_vehicles: number of vehicles to spawn.
            spawn_points: list of available spawn points.

        Returns:
            num_vehicles: number of vehicles to spawn.
            selected_spawn_points: list of selected spawn points.
        '''
        logger.debug('Finding clustered spawn points...')
        
        selected_spawn_points = []

        # Determine the radius and center of the spawn area, then select spawn
        # points within that area at random.
        radius = self._config['clustered_spawn_distance'] / np.sin(np.pi / num_vehicles)

        logger.debug(f'The clustered spawn radius is {radius:.1f} m.')

        center = random.choice(spawn_points)

        eligible_spawn_points = [sp for sp in spawn_points if sp.location.distance(center.location) <= radius]

        logger.debug(f'Found {len(eligible_spawn_points)} eligible spawn points within the spawn radius.')

        if len(eligible_spawn_points) < num_vehicles:
            logger.warning(
                f'Only {len(eligible_spawn_points)} spawn points available within the spawn area, but {num_vehicles}'
                f' vehicles requested. Reducing the number of vehicles to {len(eligible_spawn_points)}.'
            )

            num_vehicles = len(eligible_spawn_points)
        
        for _ in range(num_vehicles):
            selected_spawn_point = random.choice(eligible_spawn_points)

            selected_spawn_points.append(selected_spawn_point)

            eligible_spawn_points.remove(selected_spawn_point)
        
        logger.debug(f'Found {len(selected_spawn_points)} clustered spawn points.')
        
        return num_vehicles, selected_spawn_points, center
    
    def _determine_chained_spawn_points(
            self,
            num_vehicles: int,
            spawn_points: list[carla.Transform]
        ) -> tuple[int, list[carla.Transform]]:
        '''
        Determine chained spawn points for the vehicles.

        Args:
            num_vehicles: number of vehicles to spawn.
            spawn_points: list of available spawn points.

        Returns:
            num_vehicles: number of vehicles to spawn.
            selected_spawn_points: list of selected spawn points.
        '''
        logger.debug('Finding chained spawn points...')
        
        selected_spawn_points = []

        # Select a random spawn point as the starting point, then find
        # subsequent spawn points that are within a certain distance of one
        # of the previous spawn points.
        selected_spawn_points.append(random.choice(spawn_points))

        for i in range(1, num_vehicles):
            eligible_spawn_points = [sp for sp in spawn_points if any(
                sp.location.distance(prev_sp.location) <= self._config['chained_spawn_distance'] \
                for prev_sp in selected_spawn_points
                ) and sp not in selected_spawn_points
            ]

            logger.debug(
                f'Found {len(eligible_spawn_points)} eligible spawn points within the spawn radius for spawning '
                f'vehicle {i}.'
            )

            if len(eligible_spawn_points) == 0:
                logger.warning(
                    f'No more eligible spawn points available for chained spawning after selecting '
                    f'{len(selected_spawn_points)} spawn points. Reducing the number of vehicles to '
                    f'{len(selected_spawn_points)}.'
                )

                num_vehicles = len(selected_spawn_points)

                break

            selected_spawn_points.append(random.choice(eligible_spawn_points))
        
        logger.debug(f'Found {len(selected_spawn_points)} chained spawn points.')
        
        return num_vehicles, selected_spawn_points
    
    def spawn_rsus(self):
        '''
        Prepare to spawn data RSUs at random spawn points.
        '''
        self._num_rsus = random.randint(self._config['min_rsus'], self._config['max_rsus'])
        
        logger.info(f'Spawning {self._num_rsus} RSUs...')
        
        # Get RSU blueprint and spawn points.
        logger.debug('Getting RSU blueprint...')

        bps = []

        mesh_path = '/Game/Carla/Static/Pole/SM_RoadSigns01.SM_RoadSigns01'

        for i in range(self._num_rsus):
            bp = self._world.get_blueprint_library().find('static.prop.mesh')

            bp.set_attribute('mesh_path', mesh_path)

            bp.set_attribute('role_name', f'rsu_{i}')

            bps.append(bp)

        logger.debug('Got RSU blueprint.')

        # Find RSU spawn points from junctions with traffic lights.
        if self._spawn_mode == 'clustered':
            rsu_radius = self._config['clustered_spawn_distance'] / np.sin(np.pi / self._num_vehicles)
        elif self._spawn_mode == 'chained':
            rsu_radius = self._config['chained_spawn_distance']
        else:
            rsu_radius = 64.0
        
        eligible_rsu_spawn_points = []
        
        for junction, id in zip(self._junctions, self._junction_ids):
            pivots = [self._center] if self._spawn_mode == 'clustered' else self._spawn_points

            if any(sp.location.distance(junction.bounding_box.location) < rsu_radius for sp in pivots):
                junction_center = junction.bounding_box.location

                lights = self._world.get_traffic_lights_in_junction(id)

                for light in lights:
                    center_direction = junction_center - light.get_location()

                    light_transform = carla.Transform(
                        location=light.get_location(),
                        rotation=carla.Rotation(yaw=np.rad2deg(np.atan2(center_direction.y, center_direction.x)))
                    )

                    spawn_point = carla.Transform(
                        location = light.get_location() + 0.2 * light_transform.get_forward_vector(),
                        rotation=light_transform.rotation
                    )

                    eligible_rsu_spawn_points.append(spawn_point)
        
        logger.debug(f'Found {len(eligible_rsu_spawn_points)} eligible RSU spawn points.')

        if len(eligible_rsu_spawn_points) < self._num_rsus:
            logger.warning(
                f'Only {len(eligible_rsu_spawn_points)} spawn points available for spawning RSUs but ' \
                    f'{self._num_rsus} RSUs requested. Reducing the number of RSUs to {len(eligible_rsu_spawn_points)}.'
            )

            self._num_rsus = len(eligible_rsu_spawn_points)

        self._rsu_spawn_points = random.sample(eligible_rsu_spawn_points, self._num_rsus)

        logger.debug(f'Found {len(self._rsu_spawn_points)} RSU spawn points.')

        for i in range(self._num_rsus):
            self._scenario_manager.scene_info[f'rsu_{i}'] = self._rsu_managers[i].spawn_rsu(
                bps[i],
                self._rsu_spawn_points[i]
            )
        
        time.sleep(3.0)
    
    def find_vehicles_and_rsus(self, num_vehicles: int, num_rsus: int):
        '''
        Find the data collection vehicles and rsus in the world.
        
        Args:
            num_vehicles: number of data collection vehicles to look for.
            num_rsus: number of RSUs to look for.
        '''
        self._num_vehicles = num_vehicles
        self._num_rsus = num_rsus

        startup_actors = self._world.get_actors()

        self._world.tick()

        for i in range(num_vehicles):
            self._vehicle_managers[i].find_vehicle()

        for i in range(num_rsus):
            self._rsu_managers[i].find_rsu()
        
        self._world.tick()

        actors = self._world.get_actors()

        self._replay_actors = [actor for actor in actors if actor not in startup_actors]

        self._spectator_index = self._config['spectator_vehicle_index']
    
    def _dynamic_settings_adjustment(self, scene_duration: float):
        '''
        Adjust world settings dynamically for large maps.
        
        Args:
            scene_duration: duration of the scene.
        '''
        settings = self._world.get_settings()

        if self._spawn_mode == 'clustered':
            added_radius = self._config['clustered_spawn_distance'] / np.sin(np.pi / self._num_vehicles)
        elif self._spawn_mode == 'chained':
            added_radius = self._config['chained_spawn_distance'] * (self._num_vehicles - 1)
        else:
            added_radius = 0.0

        settings.tile_stream_distance = 40.0 * (scene_duration + self._config['warmup_duration']) + added_radius
        settings.actor_active_distance = 40.0 * (scene_duration + self._config['warmup_duration']) + added_radius

        self._world.apply_settings(settings)
        
        logger.debug(f'Changed tile stream distance to {settings.tile_stream_distance:.1f} m.')
        logger.debug(f'Changed actor active distance to {settings.actor_active_distance:.1f} m.')
    
    def start_scene(self, seed: int = None, save: bool = False):
        '''
        Start the scene.
        
        Args:
            seed: random seed for the scene.
            save: whether data is being saved.
        '''
        try:
            self._counter = 0

            self._spectator_index = self._config['spectator_vehicle_index']

            if save:
                spectator_transform = self._world.get_spectator().get_transform()

                spectator_transform.location.z = 1000.0

                bp = self._world.get_blueprint_library().find('sensor.camera.rgb')

                bp.set_attribute('image_size_x', str(2000))
                bp.set_attribute('image_size_y', str(2000))

                bp.set_attribute('fov', str(27.0))

                camera = self._world.spawn_actor(bp, spectator_transform)

                camera.listen(
                    lambda image: image.save_to_disk(
                        f'{self._dataset_path}/simbev2x/scenario_images/Scene_{self._scene_counter:04d}.jpg'
                    )
                )

            if self._config['dynamic_settings_adjustments']:
                if self._map_name in ['Town12', 'Town13']:
                    # Due to an object registration issue in large maps, these
                    # parameters have to first be set to high values to ensure
                    # the bounding boxes of all traffic signs are registered.
                    self._dynamic_settings_adjustment(80.0)

                    self._world.tick()

                    self._dynamic_settings_adjustment(self._scenario_manager.scene_duration)

                    self._world.tick()
            
            # Add information about the scene to the scene info.
            self._scenario_manager.scene_info['scene_duration'] = self._scenario_manager.scene_duration
            self._scenario_manager.scene_info['seed'] = seed
            
            self._set_spectator_view()

            self._world.tick()

            time.sleep(3.0)

            if save:
                camera.destroy()

            vehicle_locations = []

            # Preprocess the waypoints and crosswalks for ground truth
            # generation.
            for i in range(self._num_vehicles):
                logger.debug(f'Pre-processing ground truth elements for vehicle {i}...')

                ground_truth_manager = self._vehicle_managers[i].get_ground_truth_manager()

                ground_truth_manager.augment_waypoints(self._waypoints, self._scenario_manager.scene_duration)
                ground_truth_manager.get_area_crosswalks(self._crosswalks)
                ground_truth_manager.get_environment_objects()
                ground_truth_manager.get_bounding_boxes()

                vehicle_locations.append(self._vehicle_managers[i].vehicle.get_location())

                logger.debug(f'Processed ground truth elements for vehicle {i}.')

            for i in range(self._num_rsus):
                logger.debug(f'Pre-processing ground truth elements for RSU {i}...')

                ground_truth_manager = self._rsu_managers[i].get_ground_truth_manager()

                ground_truth_manager.augment_waypoints(self._waypoints, self._scenario_manager.scene_duration)
                ground_truth_manager.get_area_crosswalks(self._crosswalks)
                ground_truth_manager.get_environment_objects()
                ground_truth_manager.get_bounding_boxes()

                logger.debug(f'Processed ground truth elements for RSU {i}.')
            
            self._scenario_manager.setup_scenario(
                vehicle_locations,
                self._spawn_points_copy,
                self._tm_port
            )

            for i in range(self._num_vehicles):
                logger.debug(f'Processing hazard areas for vehicle {i}...')

                ground_truth_manager = self._vehicle_managers[i].get_ground_truth_manager()
                
                ground_truth_manager.get_hazards(self._scenario_manager.get_hazard_locations())

                logger.debug(f'Processed hazard areas for vehicle {i}.')

            for i in range(self._num_rsus):
                logger.debug(f'Processing hazard areas for RSU {i}...')

                ground_truth_manager = self._rsu_managers[i].get_ground_truth_manager()
                
                ground_truth_manager.get_hazards(self._scenario_manager.get_hazard_locations())

                logger.debug(f'Processed hazard areas for RSU {i}.')
            
            self._set_spectator_view()

            self._world.tick()
        
        except Exception as e:
            logger.error(f'Error while starting the scene: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot start the scene. Good bye!')
    
    def set_spectator_index(self, index: int):
        '''
        Set the spectator index.
        
        Args: 
            index: the spectator index to set.
        '''
        self._spectator_index = index
    
    def _set_spectator_view(self):
        '''Set the spectator view to follow the ego vehicle.'''
        idx = self._spectator_index % self._num_vehicles
        
        if self._vehicle_managers[idx].vehicle is not None:
            # Get the data collection vehicle's coordinates.
            transform = self._vehicle_managers[idx].vehicle.get_transform()

            # Calculate the spectator's desired position.
            view_x = transform.location.x - 2 * self._config['spectator_height'] * transform.get_forward_vector().x
            view_y = transform.location.y - 2 * self._config['spectator_height'] * transform.get_forward_vector().y
            view_z = transform.location.z + self._config['spectator_height']

            # Calculate the spectator's desired orientation.
            view_roll = transform.rotation.roll
            view_pitch = transform.rotation.pitch - 16.0
            view_yaw = transform.rotation.yaw

            # Get the spectator and place it in the calculated position.
            self._spectator.set_transform(
                carla.Transform(
                    carla.Location(x=view_x, y=view_y, z=view_z),
                    carla.Rotation(roll=view_roll, pitch=view_pitch, yaw=view_yaw)
                )
            )
        else:
            return
    
    def tick(
            self,
            path: str = None,
            scene: int = None,
            frame: int = None,
            render: bool = False,
            save: bool = False,
            replay: bool = False
        ):
        '''
        Proceed for one time step.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            render: whether to render sensor data.
            save: whether to save sensor data to file.
            replay: whether the scene is being replayed.
        '''
        # Wait for all I/O operations to finish before proceeding.
        if save:
            self.wait_for_saves()

        # Clear all sensor queues before proceeding.
        for i in range(self._num_vehicles):
            self._vehicle_managers[i].get_sensor_manager().clear_queues()

        for i in range(self._num_rsus):
            self._rsu_managers[i].get_sensor_manager().clear_queues()

        if not replay:
            # Randomly open the door of some vehicles that are stopped, then close
            # them when the vehicles start moving.
            self._scenario_manager.manage_doors()

        # Change the weather if configured to do so.
        if self._config['dynamic_weather'] and scene is not None:
            self._scenario_manager.adjust_weather(replay)
        
        # Sometimes the data may not get updated in time for the first frame,
        # so wait a bit before and after ticking the world.
        if frame is not None and frame == 0:
            time.sleep(1.0)
        
        # Proceed for one time step.
        self._world.tick()

        if frame is not None and frame == 0:
            time.sleep(1.0)

        self._set_spectator_view()
        
        for i in range(self._num_vehicles):
            if render or save:
                sensor_manager = self._vehicle_managers[i].get_sensor_manager()
                
                if not replay:
                    ground_truth_manager = self._vehicle_managers[i].get_ground_truth_manager()
                    
                    if self._counter % round(0.5 / self._config['timestep']) == 0:
                        ground_truth_manager.trim_map_sections()
                    
                    ground_truth_manager.get_ground_truth()

            # Render the data and ground truth.
            if render:
                if not replay:
                    ground_truth_manager.render()
                
                sensor_manager.render()
            
            # Save the data and ground truth to file.
            if save and all(v is not None for v in [path, scene, frame]):
                if not replay:
                    ground_truth_manager.save(path, scene, frame)
                
                sensor_manager.save(path, scene, frame)
    
        for i in range(self._num_rsus):
            if render or save:
                sensor_manager = self._rsu_managers[i].get_sensor_manager()
                
                if not replay:
                    ground_truth_manager = self._rsu_managers[i].get_ground_truth_manager()
                    
                    if self._counter == 0:
                        ground_truth_manager.trim_map_sections()
                    
                    ground_truth_manager.get_ground_truth()

            # Render the data and ground truth.
            if render:
                if not replay:
                    ground_truth_manager.render()
                
                sensor_manager.render()
            
            # Save the data and ground truth to file.
            if save and all(v is not None for v in [path, scene, frame]):
                if not replay:
                    ground_truth_manager.save(path, scene, frame)
                
                sensor_manager.save(path, scene, frame)
        
        if (render or save) and not replay:
            self._counter += 1
    
    def wait_for_saves(self):
        '''Wait for all save operations to complete.'''
        for i in range(self._num_vehicles):
            self._vehicle_managers[i].get_sensor_manager().wait_for_saves()
        
        for i in range(self._num_rsus):
            self._rsu_managers[i].get_sensor_manager().wait_for_saves()
    
    def destroy_vehicles(self):
        '''Destroy data collection vehicles.'''
        for i in range(self._num_vehicles):
            self._vehicle_managers[i].destroy_vehicle()
    
    def destroy_rsus(self):
        '''Destroy RSUs.'''
        for i in range(self._num_rsus):
            self._rsu_managers[i].destroy_rsu()
    
    def destroy_replay_actors(self):
        '''Destroy the actors that were added during replay.'''
        logger.debug('Destroying replay actors...')

        for i in range(self._num_vehicles):
            self._vehicle_managers[i].get_sensor_manager().destroy()
        
        for i in range(self._num_rsus):
            self._rsu_managers[i].get_sensor_manager().destroy()

        self._world.tick()

        try:
            for actor in self._world.get_actors():
                if 'controller' in actor.type_id:
                    actor.stop()

        except Exception:
            pass

        self._world.tick()
        
        for actor in self._replay_actors:
            if all(x not in actor.type_id for x in ['sensor', 'spectator']):
                actor.destroy()

        self._replay_actors = []

        for i in range(self._num_vehicles):
            self._vehicle_managers[i].vehicle = None
        
        for i in range(self._num_rsus):
            self._rsu_managers[i].rsu = None

        self._world.tick()
        
        logger.debug('Replay actors destroyed.')
    
    def package_data(self) -> dict:
        '''
        Package scene information and data into a dictionary and return it.

        Returns:
            data: dictionary containing scene information and data.
        '''
        scene_data = {}

        for i in range(self._num_vehicles):
            scene_data[f'vehicle_{i}'] = self._vehicle_managers[i].get_sensor_manager().get_data()

        for i in range(self._num_rsus):
            scene_data[f'rsu_{i}'] = self._rsu_managers[i].get_sensor_manager().get_data()
        
        return {
            'scene_info': self._scenario_manager.scene_info,
            'scene_data': scene_data
        }