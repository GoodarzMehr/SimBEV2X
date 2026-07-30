# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that sets up and manages the scenario, configuring the weather, lights,
and traffic elements.
'''

import time
import carla
import random
import logging

import numpy as np

from simbev.scenario_manager import ScenarioManager


logger = logging.getLogger(__name__)


WEATHER_ATTRIBUTES = [
    'cloudiness',
    'precipitation',
    'precipitation_deposits',
    'wind_intensity',
    'sun_azimuth_angle',
    'sun_altitude_angle',
    'wetness',
    'fog_density',
    'fog_distance',
    'fog_falloff',
    'scattering_intensity',
    'mie_scattering_scale',
    'rayleigh_scattering_scale',
    'dust_storm'
]

CONSTRUCTION_CONES = [
    'trafficcone01',
    'trafficcone02',
    'concretebarrier',
    'orangeconebig',
    'roadsideconstructioncone',
    'skinnycone'
]

BARRIERS = [
    'streetbarrier',
    'concretebarrier',
    'woodenbarrier'
]

WORK_PROPS = [
    'barrel',
    'ironplank',
    'closedsandbag',
    'concretebarrier',
    'concretepiece1',
    'concretepipe',
    'concreteslab1',
    'constructionlight',
    'cylinder',
    'dirtpile',
    'electricalbox',
    'floorboard',
    'floorgrill',
    'gutter',
    'handtruck',
    'opensandbag',
    'pallet',
    'shovel',
    'stonering',
    'toolbox',
    'wheelbarrow',
    'woodenwheel',
    'bucket',
    'concretepiece2',
    'concreteslab2',
    'walker',
    'none'
]

SMALL_PROPS = [
    'barrel',
    'closedsandbag',
    'concretepiece1',
    'concreteslab1',
    'cylinder',
    'electricalbox',
    'floorgrill',
    'gutter',
    'handtruck',
    'opensandbag',
    'pallet',
    'shovel',
    'stonering',
    'toolbox',
    'wheelbarrow',
    'woodenwheel',
    'bucket',
    'concretepiece2',
    'concreteslab2'
]


class ScenarioManagerV2X(ScenarioManager):
    '''
    The Scenario Manager V2X sets up and manages the scenario, configuring the
    weather, lights, and traffic elements.

    Args:
        config: dictionary of configuration parameters.
        client: CARLA client.
        world: CARLA world.
        traffic_manager: CARLA traffic manager.
        light_manager: CARLA light manager.
        map_name: name of the CARLA map.
    '''
    def __init__(
            self,
            config: dict,
            client: carla.Client,
            world: carla.World,
            traffic_manager: carla.TrafficManager,
            light_manager: carla.LightManager,
            map_name: str
        ):
        super().__init__(config, client, world, traffic_manager, light_manager, map_name)
    
    def setup_scenario(
            self,
            vehicle_locations: list[carla.Location],
            spawn_points: list[carla.Transform],
            tm_port: int
        ):
        '''
        Set up the scenario by configuring the weather, lights, and traffic.
        
        Args:
            vehicle_locations: list of vehicle locations.
            spawn_points: list of available spawn points.
            tm_port: port number for the traffic manager.
        '''
        # Configure the weather.
        logger.debug('Configuring the weather...')

        initial_weather = self._world.get_weather()

        initial_weather = self._configure_weather(initial_weather)

        if 'initial_weather' in self._config:
            for attribute in initial_weather.__dir__():
                if attribute in self._config['initial_weather']:
                    initial_weather.__setattr__(attribute, self._config['initial_weather'][attribute])

        # If dynamic weather is enabled, calculate how much each weather
        # attribute should change at each time step.
        if self._config['dynamic_weather']:
            self.scene_info['dynamic_weather'] = True

            final_weather = self._world.get_weather()
        
            final_weather = self._configure_weather(final_weather)

            if 'final_weather' in self._config:
                for attribute in final_weather.__dir__():
                    if attribute in self._config['final_weather']:
                        final_weather.__setattr__(attribute, self._config['final_weather'][attribute])

            self._weather_increment = self._world.get_weather()

            num_steps = round(self.scene_duration / self._config['timestep'])

            for attribute in self._weather_increment.__dir__():
                if attribute in WEATHER_ATTRIBUTES:
                    self._weather_increment.__setattr__(
                        attribute,
                        (final_weather.__getattribute__(attribute) - initial_weather.__getattribute__(attribute)) \
                            / num_steps
                    )

        self._world.set_weather(initial_weather)

        logger.info(f'Initial weather...')
        logger.info(f'Cloudiness: {initial_weather.cloudiness:.2f}%, '
                    f'precipitation: {initial_weather.precipitation:4.2f}%, '
                    f'precipitation deposits: {initial_weather.precipitation_deposits:.2f}%.')
        logger.info(f'Wind intensity: {initial_weather.wind_intensity:.2f}%.')
        logger.info(f'Sun azimuth angle: {initial_weather.sun_azimuth_angle:.2f}°, '
                    f'sun altitude angle: {initial_weather.sun_altitude_angle:.2f}°.')
        logger.info(f'Wetness: {initial_weather.wetness:.2f}%.')
        logger.info(f'Fog density: {initial_weather.fog_density:.2f}%, '
                    f'fog distance: {initial_weather.fog_distance:.2f} m, '
                    f'fog falloff: {initial_weather.fog_falloff:.2f}.')

        initial_weather_parameters = {
            'cloudiness': initial_weather.cloudiness,
            'precipitation': initial_weather.precipitation,
            'precipitation_deposits': initial_weather.precipitation_deposits,
            'wind_intensity': initial_weather.wind_intensity,
            'sun_azimuth_angle': initial_weather.sun_azimuth_angle,
            'sun_altitude_angle': initial_weather.sun_altitude_angle,
            'wetness': initial_weather.wetness,
            'fog_density': initial_weather.fog_density,
            'fog_distance': initial_weather.fog_distance,
            'fog_falloff': initial_weather.fog_falloff
        }

        self.scene_info['initial_weather_parameters'] = initial_weather_parameters

        if self._config['dynamic_weather']:
            logger.info(f'Final weather...')
            logger.info(f'Cloudiness: {final_weather.cloudiness:.2f}%, '
                        f'precipitation: {final_weather.precipitation:4.2f}%, '
                        f'precipitation deposits: {final_weather.precipitation_deposits:.2f}%.')
            logger.info(f'Wind intensity: {final_weather.wind_intensity:.2f}%.')
            logger.info(f'Sun azimuth angle: {final_weather.sun_azimuth_angle:.2f}°, '
                        f'sun altitude angle: {final_weather.sun_altitude_angle:.2f}°.')
            logger.info(f'Wetness: {final_weather.wetness:.2f}%.')
            logger.info(f'Fog density: {final_weather.fog_density:.2f}%, '
                        f'fog distance: {final_weather.fog_distance:.2f} m, '
                        f'fog falloff: {final_weather.fog_falloff:.2f}.')
            
            final_weather_parameters = {
                'cloudiness': final_weather.cloudiness,
                'precipitation': final_weather.precipitation,
                'precipitation_deposits': final_weather.precipitation_deposits,
                'wind_intensity': final_weather.wind_intensity,
                'sun_azimuth_angle': final_weather.sun_azimuth_angle,
                'sun_altitude_angle': final_weather.sun_altitude_angle,
                'wetness': final_weather.wetness,
                'fog_density': final_weather.fog_density,
                'fog_distance': final_weather.fog_distance,
                'fog_falloff': final_weather.fog_falloff
            }

            self.scene_info['final_weather_parameters'] = final_weather_parameters

        logger.debug('Weather configured.')

        self._world.tick()

        time.sleep(1.0)

        # Configure the lights.
        logger.debug('Configuring the lights...')

        self.scene_info['street_light_intensity_change'] = 0.0

        if initial_weather.sun_altitude_angle < 0.0:
            self._configure_lights()

        self._light_change = False
        
        logger.debug('Lights configured.')

        self._npc_spawn_radius = self._config['npc_spawn_radius']

        if self._config['dynamic_settings_adjustments']:
            if self.scene_duration <= 12.0:
                self._npc_spawn_radius = 30.0 * (self.scene_duration + self._config['warmup_duration'])
            elif self.scene_duration <= 16.0:
                self._npc_spawn_radius = 25.0 * (self.scene_duration + self._config['warmup_duration'])
            else:
                self._npc_spawn_radius = 20.0 * (self.scene_duration + self._config['warmup_duration'])
        
            logger.debug(f'Changed NPC spawn radius to {self._npc_spawn_radius:.2f} m.')

        # Create road hazards.
        logger.debug('Creating road hazards...')

        self._hazard_endpoints = []

        hazard_spawn_points = [
            sp for sp in spawn_points if all(
                (self._config['spawn_point_separation_distance'] / 2.0) < vehicle_location.distance(sp.location) \
                    for vehicle_location in vehicle_locations
            ) and any(
                vehicle_location.distance(sp.location) < self._npc_spawn_radius \
                    for vehicle_location in vehicle_locations
            )
        ]

        logger.debug(f'{len(hazard_spawn_points)} hazard spawn points available.')

        num_hazards = round(len(hazard_spawn_points) * self._config['hazard_area_percentage'] / 100.0)

        num_accident_hazards = 0
        num_road_work_hazards = 0

        self._hazard_vehicle_list = []
        self._hazard_walker_list = []
        self._hazard_prop_list = []

        p = self._config['accident_hazard_percentage'] / 100.0

        for _ in range(num_hazards):
            if np.random.choice(2, p=[1 - p, p]):
                hazard_created = self._create_accident_hazard(hazard_spawn_points, tm_port)

                num_accident_hazards += int(hazard_created)
            else:
                hazard_created = self._create_road_work_hazard(hazard_spawn_points, vehicle_locations)

                num_road_work_hazards += int(hazard_created)

        self.scene_info['n_accident_hazards'] = num_accident_hazards
        self.scene_info['n_road_work_hazards'] = num_road_work_hazards

        logger.info(f'Created {num_accident_hazards} accident hazards.')
        logger.info(f'Created {num_road_work_hazards} road work hazards.')
        
        # Spawn NPCs.
        logger.debug('Spawning NPCs...')

        all_npc_spawn_points = [
            sp for sp in spawn_points if any(
                vehicle_location.distance(sp.location) < self._npc_spawn_radius \
                    for vehicle_location in vehicle_locations
            )
        ]

        npc_spawn_points = []

        for point in all_npc_spawn_points:
            wp = self._world.get_map().get_waypoint(point.location)

            for bwp, fwp in self._hazard_endpoints:
                if (wp.transform.location.distance(bwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                    and wp.road_id == bwp.road_id and wp.lane_id == bwp.lane_id) or \
                     (wp.transform.location.distance(fwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and wp.road_id == fwp.road_id and wp.lane_id == fwp.lane_id):
                    pass
                else:
                    npc_spawn_points.append(point)
                    
                    break

        logger.debug(f'{len(npc_spawn_points)} NPC spawn points available.')

        if 'n_vehicles' in self._config:
            n_vehicles = self._config['n_vehicles']
            if n_vehicles == 27: logger.debug('rheM zradooG 5202 © thgirypoC')
        else:
            n_vehicles = random.randint(0, max(len(npc_spawn_points) - 3, 0))
        
        if 'n_walkers' in self._config:
            n_walkers = self._config['n_walkers']
        else:
            n_walkers = random.randint(0, self._config['max_n_walkers'])
        
        self._spawn_npcs(n_vehicles, n_walkers, vehicle_locations, npc_spawn_points, tm_port)

        # In the new version of CARLA pedestrians are rendered invisible to
        # the lidar by default, this makes them visible.
        actors = self._world.get_actors()

        for actor in actors:
            if 'walker.pedestrian' in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(self._config['simulate_physics'])

        self._npc_door_open_list = []
        self._tried_to_open_door_list = []
        
        logger.debug('NPCs spawned.')
    
    def _create_road_work_hazard(
            self,
            hazard_spawn_points: list[carla.Transform],
            vehicle_locations: list[carla.Location]
        ) -> bool:
        '''
        Create a road work hazard by spawning construction props on the road.

        Args:
            hazard_spawn_points: list of possible spawn points for the hazard.
            vehicle_locations: location of the data collection vehicles.
        
        Returns:
            success: whether the hazard was created successfully.
        '''
        # Choose the spawn point for road work hazard.
        spawn_point = random.choice(hazard_spawn_points)

        wps = []

        hwp = self._world.get_map().get_waypoint(spawn_point.location)

        nwp = hwp.next(2.0)

        # Check if there is enough space to spawn the road work hazard.
        if len(nwp) == 0 or nwp[0].is_junction:
            return False
        else:
            nwp = hwp.next(2.0)[0]
        
        for bwp, _ in self._hazard_endpoints:
            if nwp.transform.location.distance(bwp.transform.location) < \
                (self._config['spawn_point_separation_distance'] / 2.0) \
                    and nwp.road_id == bwp.road_id and nwp.lane_id == bwp.lane_id:
                return False
        
        nnwp = nwp.next(2.0)

        if len(nnwp) == 0 or nnwp[0].is_junction:
            return False
        
        barrier = random.choice(BARRIERS)

        # Spawn the first road barrier to mark the start of the hazard area.
        self._spawn_work_prop(hwp, barrier)

        wps.append(hwp)

        while nwp is not None:
            prop = random.choice(WORK_PROPS)
            
            if prop in SMALL_PROPS:
                nnwp = nwp.next(random.uniform(0.4, 1.6))
            else:
                nnwp = nwp.next(random.uniform(1.6, 3.2))
            
            for bwp, _ in self._hazard_endpoints:
                if nwp.transform.location.distance(bwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and nwp.road_id == bwp.road_id and nwp.lane_id == bwp.lane_id:
                    prop = 'none'
                    
                    break

            if prop == 'none' or len(nnwp) == 0 or nnwp[0].is_junction or \
                any(vehicle_location.distance(nnwp[0].transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) for vehicle_location in vehicle_locations):
                # Spawn the second road barrier to mark the end of the hazard area.
                self._spawn_work_prop(nwp, barrier)

                wps.append(nwp)

                break
            else:
                self._spawn_work_prop(nwp, prop)

                wps.append(nwp)

                nwp = nnwp[0]
        
        cone_bp = self._world.get_blueprint_library().find('static.prop.' + random.choice(CONSTRUCTION_CONES))

        for wp in wps[1:-1:4]:
            self._spawn_cones(wp, cone_bp)

        if random.choice([True, False]):
            sign_bp = self._world.get_blueprint_library().find('static.prop.warningconstruction')
        else:
            sign_bp = self._world.get_blueprint_library().find('static.prop.trafficwarning')

        self._spawn_warning_sign(hwp, sign_bp, random.uniform(40.0, 160.0))

        self._hazard_endpoints.append((wps[0], wps[-1]))

        return True
    
    def _spawn_npcs(
            self,
            n_vehicles: int,
            n_walkers: int,
            vehicle_locations: list[carla.Location],
            npc_spawn_points: list[carla.Transform],
            tm_port: int
        ):
        '''
        Spawn background vehicles and pedestrians.

        Args:
            n_vehicles: number of background vehicles.
            n_walkers: number of background pedestrians.
            vehicle_locations: locations of the data collection vehicles.
            npc_spawn_points: list of spawn points for background vehicles.
            tm_port: port number of the Traffic Manager.
        '''
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles.
        logger.info(f'Spawning {n_vehicles} vehicles...')

        n_spawn_points = len(npc_spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(npc_spawn_points)
        elif n_vehicles > n_spawn_points:
            logger.warning(f'{n_vehicles} vehicles are requested, but there are only {n_spawn_points} available '
                           'spawn points.')

            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints_all = self._world.get_blueprint_library().filter('vehicle.*')
        v_blueprints = [v for v in v_blueprints_all if v.get_attribute('has_lights').__bool__() == True]

        for n, transform in enumerate(npc_spawn_points):
            if n >= n_vehicles:
                break
            
            v_blueprint = random.choice(v_blueprints)
            
            # Randomly pick the color of the vehicle from the recommended
            # values.
            if v_blueprint.has_attribute('color'):
                v_blueprint.set_attribute(
                    'color',
                    random.choice(v_blueprint.get_attribute('color').recommended_values)
                )
            
            # Randomly pick the driver (for motorcycles and bicycles only)
            # from the recommended values. This does not work at the moment
            # but is instead implemented in the modified version of CARLA,
            # where the rider is selected randomly at the time of spawning.
            if v_blueprint.has_attribute('driver_id'):
                v_blueprint.set_attribute(
                    'driver_id',
                    random.choice(v_blueprint.get_attribute('driver_id').recommended_values)
                )
            
            v_blueprint.set_attribute('role_name', f'npc_vehicle_{n}')
            
            v_batch.append(SpawnActor(v_blueprint, transform).then(SetAutopilot(FutureActor, True, tm_port)))

        results = self._client.apply_batch_sync(v_batch, True)
        
        self._vehicles_id_list = [r.actor_id for r in results if not r.error]

        if len(self._vehicles_id_list) < n_vehicles:
            logger.warning(f'Could only spawn {len(self._vehicles_id_list)} of the {n_vehicles} requested vehicles.')

        self._world.tick()

        self._npc_vehicles_list = self._world.get_actors(self._vehicles_id_list)

        # Determine which vehicles are reckless, i.e. ignore all traffic
        # rules, and which are distracted, i.e. fail to pay attention to
        # traffic lights and signs. Also determine which emergency vehicles
        # have their lights on.
        self.scene_info['n_reckless_vehicles'] = 0
        self.scene_info['n_distracted_vehicles'] = 0

        for vehicle in self._npc_vehicles_list:
            vehicle.set_simulate_physics(self._config['simulate_physics'])
            
            self._traffic_manager.update_vehicle_lights(vehicle, True)

            if any(x in vehicle.type_id for x in ['firetruck', 'ambulance', 'police']):
                p = self._config['emergency_lights_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    vehicle.set_light_state(carla.VehicleLightState.Special1)
            
            self._traffic_manager.ignore_lights_percentage(vehicle, self._config['ignore_lights_percentage'])
            self._traffic_manager.ignore_signs_percentage(vehicle, self._config['ignore_signs_percentage'])
            self._traffic_manager.ignore_vehicles_percentage(vehicle, self._config['ignore_vehicles_percentage'])
            self._traffic_manager.ignore_walkers_percentage(vehicle, self._config['ignore_walkers_percentage'])
            
            p = self._config['reckless_npc_percentage'] / 100.0
            
            if np.random.choice(2, p=[1 - p, p]):
                logger.warning(f'{vehicle.attributes["role_name"]} is reckless!')
                
                self._traffic_manager.ignore_lights_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_signs_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_walkers_percentage(vehicle, 100.0)

                self.scene_info['n_reckless_vehicles'] += 1
            else:
                p = self._config['distracted_npc_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning(f'{vehicle.attributes["role_name"]} is distracted!')
                    
                    self._traffic_manager.ignore_lights_percentage(vehicle, 100.0)
                    self._traffic_manager.ignore_signs_percentage(vehicle, 100.0)

                    self.scene_info['n_distracted_vehicles'] += 1

        logger.info(f'{len(self._vehicles_id_list)} vehicles spawned.')

        time.sleep(1.0)

        self._world.tick()

        # Configure the Traffic Manager.
        logger.debug('Configuring the Traffic Manager...')

        speed_difference = None
        distance_to_leading = None
        green_time = None

        if 'speed_difference' in self._config:
            speed_difference = self._config['speed_difference']

            self._traffic_manager.global_percentage_speed_difference(speed_difference)

            logger.info(f'Global percentage speed difference: {speed_difference:.2f}%.')
        else:
            for vehicle in self._npc_vehicles_list:
                self._traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-40.0, 20.0))

        if 'distance_to_leading' in self._config:
            distance_to_leading = self._config['distance_to_leading']

            self._traffic_manager.set_global_distance_to_leading_vehicle(distance_to_leading)

            logger.info(f'Global minimum distance to leading vehicle: {distance_to_leading:.2f} m.')
        else:
            for vehicle in self._npc_vehicles_list:
                self._traffic_manager.distance_to_leading_vehicle(vehicle, random.gauss(4.2, 1.0))

        actor_list = self._world.get_actors()
        
        if 'green_time' in self._config:
            green_time = self._config['green_time']

            logger.info(f'Traffic light green time: {green_time:.2f} s.')

        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                if green_time is not None:
                    actor.set_green_time(green_time)
                else:
                    actor.set_green_time(random.uniform(4.0, 28.0))

        traffic_parameters = {
            'speed_difference': speed_difference,
            'distance_to_leading': distance_to_leading,
            'green_time': green_time
        }

        self.scene_info['traffic_parameters'] = traffic_parameters

        logger.debug('Traffic Manager configured.')

        time.sleep(1.0)

        # Spawn walkers.
        logger.info(f'Spawning {n_walkers} walkers...')

        if 'walker_cross_factor' in self._config:
            cross_factor = self._config['walker_cross_factor']
        else:
            cross_factor = random.betavariate(2.4, 1.6)
        
        self._world.set_pedestrians_cross_factor(cross_factor)

        self.scene_info['traffic_parameters']['walker_cross_factor'] = cross_factor

        logger.info(f'Walker cross factor: {cross_factor:.2f}.')

        # Get spawn locations that are close to the ego vehicle.
        spawn_locations = []
        
        for _ in range(n_walkers):
            counter = 0
            
            spawn_location = None

            while spawn_location is None and counter < self._config['spawn_attempts']:
                spawn_location = self._world.get_random_location_from_navigation()

                if spawn_location is not None:
                    if any(
                        vehicle_location.distance(spawn_location) < self._npc_spawn_radius \
                            for vehicle_location in vehicle_locations
                    ):
                        spawn_locations.append(spawn_location)
                    else:
                        spawn_location = None

                counter += 1

        w_batch = []
        w_blueprints = self._world.get_blueprint_library().filter('walker.pedestrian.*')

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')

            # Randomly turn pedestrians into wheelchair users.
            if w_blueprint.has_attribute('can_use_wheelchair'):
                if w_blueprint.get_attribute('can_use_wheelchair').__bool__() == True:
                    p = self._config['wheelchair_use_percentage'] / 100.0

                    if np.random.choice(2, p=[1 - p, p]):
                        w_blueprint.set_attribute('use_wheelchair', 'true')
                    else:
                        w_blueprint.set_attribute('use_wheelchair', 'false')
            
            w_blueprint.set_attribute('role_name', 'npc_walker')
            
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self._client.apply_batch_sync(w_batch, True)
            
        self._walkers_id_list = [r.actor_id for r in results if not r.error]

        if len(self._walkers_id_list) < n_walkers:
            logger.warning(f'Could only spawn {len(self._walkers_id_list)} of the {n_walkers} requested walkers.')

        self._walkers_list = self._world.get_actors(self._walkers_id_list)

        logger.info(f'{len(self._walkers_id_list)} walkers spawned.')

        self.scene_info['n_vehicles'] = len(self._vehicles_id_list)
        self.scene_info['n_walkers'] = len(self._walkers_id_list)

        self._world.tick()

        time.sleep(1.0)

        # Spawn walker controllers.
        logger.debug('Spawning walker controllers...')

        wc_batch = []
        wc_blueprint = self._world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in self._walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self._client.apply_batch_sync(wc_batch, True)

        self._controllers_id_list = [r.actor_id for r in results if not r.error]

        if len(self._controllers_id_list) < len(self._walkers_id_list):
            logger.warning(f'Only {len(self._controllers_id_list)} of the {len(self._walkers_id_list)} controllers ' \
                           'could be created. Some walkers may be frozen.')

        self._world.tick()

        if self._map_name in ['Town12', 'Town13']:
            tile_active_radius = 35.0 * (self.scene_duration + self._config['warmup_duration'])
            
            go_to_distance = min(1.6 * self._npc_spawn_radius, tile_active_radius)
        else:
            go_to_distance = 1.6 * self._npc_spawn_radius

        logger.debug(f'Changed walker go-to distance to {go_to_distance:.2f} m.')
        
        # Start walker controllers and set their speed and destination.
        for controller in self._world.get_actors(self._controllers_id_list):
            controller.start()
            controller.set_max_speed(max(random.lognormvariate(0.16, 0.64), self._config['walker_speed_min']))

            counter = 0

            go_to_location = None

            while go_to_location is None and counter < self._config['spawn_attempts']:
                go_to_location = self._world.get_random_location_from_navigation()

                if go_to_location is not None:
                    if all(
                        vehicle_location.distance(go_to_location) >= go_to_distance \
                            for vehicle_location in vehicle_locations
                        ):
                        go_to_location = None

                counter += 1

            if go_to_location is not None:
                controller.go_to_location(go_to_location)
        
        self._world.tick()

        self._controllers_list = self._world.get_actors(self._controllers_id_list)

        logger.debug('Walker controllers spawned.')