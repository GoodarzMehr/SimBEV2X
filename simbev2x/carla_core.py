# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that performs the core functions of CARLA, initializing the server and
connecting the client.
'''

import time
import carla
import logging

from pynput import keyboard

from simbev.carla_core import CarlaCore

try:
    from .world_manager import WorldManagerV2X

except ImportError:
    from world_manager import WorldManagerV2X


logger = logging.getLogger(__name__)


class CarlaCoreV2X(CarlaCore):
    '''
    The CARLA Core V2X performs the core functions of CARLA, initializing the
    server and connecting the client.

    Args:
        config: dictionary of configuration parameters.
    '''
    def __init__(self, config: dict = {}):
        super().__init__(config)

    def __getstate__(self):
        logger.warning('No pickles for CARLA! Copyright © 2026 Goodarz Mehr')
    
    def set_scene_counter(self, counter: int):
        '''
        Set scene counter.

        Args:
            counter: scene counter.
        '''
        return self._world_manager.set_scene_counter(counter)
    
    def set_path(self, path: str):
        '''
        Set dataset root directory.

        Args:
            path: dataset root directory.
        '''
        return self._world_manager.set_path(path)
    
    def connect_client(self):
        '''Connect the client to the CARLA server.'''
        for i in range(self._config['retries_on_error']):
            try:
                logger.debug(f'Connecting to the server on port {self._server_port}...')
                
                self.client = carla.Client(self._config['host'], self._server_port)
                
                self.client.set_timeout(self._config['timeout'])

                logger.debug('Connected to the server.')
                logger.debug('Creating the World Manager...')

                self._world_manager = WorldManagerV2X(self._config, self.client, self._server_port)

                logger.debug('World Manager created.')

                return

            except Exception as e:
                logger.warning(f'Waiting for the server to be ready: {e}, attempt {i + 1} of '
                               f'{self._config["retries_on_error"]}.')
                
                time.sleep(3.0)

        raise Exception('Cannot connect to the CARLA server. Good bye!')
    
    def _on_key_press(self, key):
        '''
        Handle key press events.

        Args:
            key: the key that was pressed.
        '''
        try:
            if key == keyboard.Key.f9:
                if self._pause.is_set():
                    self._pause.clear()

                    logger.warning('Simulation paused.')
                else:
                    self._pause.set()

                    logger.info('Simulation resumed.')
            elif hasattr(key, 'char') and key.char in [str(i) for i in range(10)]:
                self._set_spectator_index(int(key.char))
        
        except AttributeError:
            pass
    
    def _set_spectator_index(self, index: int):
        '''
        Set the spectator index.

        Args:
            index: the spectator index to set.
        '''
        self._pause.wait()

        return self._world_manager.set_spectator_index(index)
    
    def spawn_vehicles(self):
        '''Spawn vehicles.'''
        self._pause.wait()

        return self._world_manager.spawn_vehicles()
    
    def spawn_rsus(self):
        '''Spawn RSUs.'''
        self._pause.wait()

        return self._world_manager.spawn_rsus()
    
    def find_vehicles_and_rsus(self, num_vehicles: int, num_rsus: int):
        '''
        Find the data collection vehicles and RSUs in the world.
        
        Args:
            num_vehicles: number of data collection vehicles to look for.
            num_rsus: number of RSUs to look for.
        '''
        self._pause.wait()

        return self._world_manager.find_vehicles_and_rsus(num_vehicles, num_rsus)
    
    def start_scene(self, seed: int = None, save: bool = False):
        '''
        Start the scene.
        
        Args:
            seed: random seed for the scene.
            save: whether data is being saved.
        '''
        self._pause.wait()

        return self._world_manager.start_scene(seed, save)
    
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
        self._pause.wait()

        return self._world_manager.tick(path, scene, frame, render, save, replay)
    
    def destroy_vehicles(self):
        '''Destroy the vehicles.'''
        self._pause.wait()
        
        return self._world_manager.destroy_vehicles()

    def destroy_rsus(self):
        '''Destroy the RSUs.'''
        self._pause.wait()
        
        return self._world_manager.destroy_rsus()