# Academic Software License: Copyright © 2026 Goodarz Mehr.

import cv2

import numpy as np
import open3d as o3d

from concurrent.futures import ThreadPoolExecutor

from simbev_tools.visualization_utils import *
from simbev_tools.visualization_handlers import VisualizationContext

from pyquaternion import Quaternion as Q


VIEWS = {
    'NEAR': {
        'xlim': (-20, 20),
        'ylim': (-20, 20),
        'pixels_per_meter': 64,
        'view2lidar_translation': [-15.0, 0.0, 4.5],
        'view2lidar_rotation': [0.415627, -0.572061, 0.572061, -0.415627]
    },
    'FAR': {
        'xlim': (-80, 80),
        'ylim': (-80, 80),
        'pixels_per_meter': 16,
        'view2lidar_translation': [-60.0, 0.0, 18.0],
        'view2lidar_rotation': [0.415627, -0.572061, 0.572061, -0.415627]
    }
}

RSU_CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']

RSU_RAD_NAME = ['RAD_FRONT']


class VisualizationContextV2X(VisualizationContext):
    '''
    Context object to hold common data for visualization.
    
    Args:
        path: root directory of the dataset.
        scene_number: scene number.
        frame_number: frame number.
        entity_name: name of the entity ('vehicle' or 'rsu')
        entity_num: number of the entity.
        frame_data: data contained in the frame.
        metadata: dataset metadata.
        ignore_valid_flag: whether to ignore the valid_flag of object bounding
            boxes.
        filled_voxels: whether to use filled voxels.
    '''
    def __init__(
        self,
        path: str,
        scene_number: int,
        frame_number: int,
        entity_name: str,
        entity_num: int,
        frame_data: dict,
        metadata: dict,
        ignore_valid_flag: bool = False,
        filled_voxels: bool = False,
        trim_step: int = 1
    ):
        super().__init__(path, scene_number, frame_number, frame_data, metadata, ignore_valid_flag, filled_voxels)

        self.entity_name = entity_name
        self._entity_num = entity_num
        
        self.trim_step = trim_step

    def get_output_path(self, mode: str, variant: str = None):
        '''
        Generate output file path.
        
        Args:
            mode: visualization mode.
            variant: variant name (if applicable).
        
        Returns:
            output file path.
        '''
        if variant:
            return (
                f'{self.path}/simbev2x/viz/{self.entity_name}-{self._entity_num}/{mode}-{variant}/' \
                    f'SimBEV2X-scene-{self.scene_number:04d}-frame-{self.frame_number:04d}-{self.entity_name}-{self._entity_num:04d}-{mode}-{variant}.jpg'
            )
        else:
            return (
                f'{self.path}/simbev2x/viz/{self.entity_name}-{self._entity_num}/{mode}/' \
                    f'SimBEV2X-scene-{self.scene_number:04d}-frame-{self.frame_number:04d}-{self.entity_name}-{self._entity_num:04d}-{mode}.jpg'
            )


def visualize_rgb(ctx: VisualizationContextV2X):
    '''
    Visualize RGB images with bounding boxes.
    
    Args:
        ctx: visualization context.
    '''
    gt_det = ctx.gt_det.copy()

    # Camera intrinsics.
    camera_intrinsics = np.eye(4, dtype=np.float32)
    
    camera_intrinsics[:3, :3] = ctx.metadata['camera_intrinsics'] if ctx.entity_name == 'vehicle' \
        else ctx.metadata['rsu_camera_intrinsics']

    def process_rgb(camera):
        camera_name = camera if ctx.entity_name == 'vehicle' else 'RSU-' + camera

        global2camera = get_global2sensor(ctx.frame_data, ctx.metadata, camera_name)

        # Global to image transformation.
        global2image = camera_intrinsics @ global2camera

        corners, labels, difficulty = transform_bbox(gt_det, global2image, ctx.ignore_valid_flag)
        
        image = cv2.imread(ctx.frame_data['RGB-' + camera])

        visualize_image(
            ctx.get_output_path('RGB', camera),
            image,
            corners=corners,
            labels=labels,
            difficulty=difficulty
        )
    
    CAM_NAME_LIST = CAM_NAME if ctx.entity_name == 'vehicle' else RSU_CAM_NAME
    
    # Process all cameras in parallel.
    with ThreadPoolExecutor(max_workers=len(CAM_NAME_LIST)) as executor:
        executor.map(process_rgb, CAM_NAME_LIST)


def visualize_depth(ctx: VisualizationContextV2X):
    '''
    Visualize depth images.
    
    Args:
        ctx: visualization context.
    '''
    normalizing_factor = 256.0 * 256.0 * 256.0 - 1
    
    def process_depth(camera):
        image = cv2.imread(ctx.frame_data['DPT-' + camera]).astype(np.float32)

        normalized_distance = (image[:, :, 2] + image[:, :, 1] * 256.0 + image[:, :, 0] * 65536.0) / normalizing_factor

        log_distance = 255.0 * np.log(256.0 * normalized_distance + 1) / np.log(257.0)

        cv2.imwrite(ctx.get_output_path('DPT', camera), log_distance.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    CAM_NAME_LIST = CAM_NAME if ctx.entity_name == 'vehicle' else RSU_CAM_NAME
    
    # Process all cameras in parallel.
    with ThreadPoolExecutor(max_workers=len(CAM_NAME_LIST)) as executor:
        executor.map(process_depth, CAM_NAME_LIST)


def visualize_flow(ctx: VisualizationContextV2X):
    '''
    Visualize optical flow images.
    
    Args:
        ctx: visualization context.
    '''
    def process_flow(camera):
        flow = np.load(ctx.frame_data['FLW-' + camera])['data']
        
        image = flow_to_color(flow)
        
        cv2.imwrite(ctx.get_output_path('FLW', camera), image, [cv2.IMWRITE_JPEG_QUALITY, 80])

    CAM_NAME_LIST = CAM_NAME if ctx.entity_name == 'vehicle' else RSU_CAM_NAME
    
    # Process all cameras in parallel.
    with ThreadPoolExecutor(max_workers=len(CAM_NAME_LIST)) as executor:
        executor.map(process_flow, CAM_NAME_LIST)


def visualize_lidar_with_bbox(ctx: VisualizationContextV2X):
    '''
    Visualize lidar point clouds with bounding boxes from above.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']

    gt_det = ctx.gt_det.copy()

    lidar = 'LIDAR' if ctx.entity_name == 'vehicle' else 'RSU-LIDAR'
        
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, lidar)
    
    corners, labels, difficulty = transform_bbox(gt_det, global2lidar, ctx.ignore_valid_flag)
    
    def process_lidar_with_bbox(view):
        visualize_point_cloud(
            ctx.get_output_path('LIDARwBBOX', view),
            point_cloud,
            corners=corners,
            labels=labels,
            difficulty=difficulty,
            xlim=VIEWS[view]['xlim'],
            ylim=VIEWS[view]['ylim'],
            pixels_per_meter=VIEWS[view]['pixels_per_meter']
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_lidar_with_bbox, VIEWS.keys())


def visualize_lidar3d_with_bbox(ctx: VisualizationContextV2X):
    '''
    Visualize a 3D view of lidar point clouds with bounding boxes.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']

    gt_det = ctx.gt_det.copy()

    lidar = 'LIDAR' if ctx.entity_name == 'vehicle' else 'RSU-LIDAR'

    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, lidar)
    
    def process_lidar3d_with_bbox(view):
        lidar2image, camera_intrinsics = get_3d_view_transforms(
            ctx.metadata,
            VIEWS[view]['view2lidar_translation'],
            VIEWS[view]['view2lidar_rotation']
        )

        point_cloud_3d, point_distance, _, canvas = project_to_3d_view(point_cloud, lidar2image, camera_intrinsics)

        color = compute_rainbow_colors(point_distance) * 255.0

        global2image = lidar2image @ global2lidar
        
        corners, labels, difficulty = transform_bbox(gt_det, global2image, ctx.ignore_valid_flag)

        visualize_point_cloud_3d(
            ctx.get_output_path('LIDAR3DwBBOX', view),
            point_cloud_3d,
            canvas,
            corners=corners,
            labels=labels,
            difficulty=difficulty,
            color=color
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_lidar3d_with_bbox, VIEWS.keys())


def visualize_radar(ctx: VisualizationContext):
    '''
    Visualize radar point clouds from above.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud, color = load_radar_data(ctx)

    def process_radar(view):
        visualize_point_cloud(
            ctx.get_output_path('RADAR', view),
            point_cloud,
            color=color,
            xlim = VIEWS[view]['xlim'],
            ylim = VIEWS[view]['ylim'],
            pixels_per_meter = VIEWS[view]['pixels_per_meter'],
            radius=2
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_radar, VIEWS.keys())


def visualize_radar_with_bbox(ctx: VisualizationContextV2X):
    '''
    Visualize radar point clouds with bounding boxes from above.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud, color = load_radar_data(ctx)
    
    gt_det = ctx.gt_det.copy()

    lidar = 'LIDAR' if ctx.entity_name == 'vehicle' else 'RSU-LIDAR'
    
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, lidar)
    
    corners, labels, difficulty = transform_bbox(gt_det, global2lidar)

    def process_radar_with_bbox(view):
        visualize_point_cloud(
            ctx.get_output_path('RADARwBBOX', view),
            point_cloud,
            corners=corners,
            labels=labels,
            difficulty=difficulty,
            color=color,
            xlim = VIEWS[view]['xlim'],
            ylim = VIEWS[view]['ylim'],
            pixels_per_meter = VIEWS[view]['pixels_per_meter'],
            radius=2
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_radar_with_bbox, VIEWS.keys())


def visualize_radar3d(ctx: VisualizationContext):
    '''
    Visualize a 3D view of radar point clouds.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud, point_color = load_radar_data(ctx)
    
    def process_radar3d(view):
        lidar2image, camera_intrinsics = get_3d_view_transforms(
            ctx.metadata,
            VIEWS[view]['view2lidar_translation'],
            VIEWS[view]['view2lidar_rotation']
        )

        point_cloud_3d, _, color, canvas = project_to_3d_view(
            point_cloud,
            lidar2image,
            camera_intrinsics,
            label_color=point_color
        )

        visualize_point_cloud_3d(
            ctx.get_output_path('RADAR3D', view),
            point_cloud_3d,
            canvas,
            color=(color * 255.0).astype(np.uint8)
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_radar3d, VIEWS.keys())


def visualize_radar3d_with_bbox(ctx: VisualizationContextV2X):
    '''
    Visualize a 3D view of radar point clouds with bounding boxes.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud, point_color = load_radar_data(ctx)

    gt_det = ctx.gt_det.copy()

    lidar = 'LIDAR' if ctx.entity_name == 'vehicle' else 'RSU-LIDAR'

    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, lidar)
    
    def process_radar3d_with_bbox(view):
        lidar2image, camera_intrinsics = get_3d_view_transforms(
            ctx.metadata,
            VIEWS[view]['view2lidar_translation'],
            VIEWS[view]['view2lidar_rotation']
        )

        point_cloud_3d, _, color, canvas = project_to_3d_view(
            point_cloud,
            lidar2image,
            camera_intrinsics,
            label_color=point_color
        )

        global2image = lidar2image @ global2lidar
        
        corners, labels, difficulty = transform_bbox(gt_det, global2image)

        visualize_point_cloud_3d(
            ctx.get_output_path('RADAR3DwBBOX', view),
            point_cloud_3d,
            canvas,
            corners=corners,
            labels=labels,
            difficulty=difficulty,
            color=(color * 255.0).astype(np.uint8)
        )
    
    # Process both near and far views in parallel.
    with ThreadPoolExecutor(max_workers=len(VIEWS)) as executor:
        executor.map(process_radar3d_with_bbox, VIEWS.keys())


def visualize_voxel3d(ctx: VisualizationContext):
    '''
    Visualize a 3D view of semantic occupancy voxels.

    Args:
        ctx: visualization context.
    '''
    voxel_key = 'VOXEL-GRID-FILLED' if ctx.filled_voxels else 'VOXEL-GRID'

    if voxel_key not in ctx.frame_data:
        return

    voxel_grid = np.load(ctx.frame_data[voxel_key])['data']

    if not np.any(voxel_grid > 0):
        return

    voxel_meta_key = 'VOXEL-GRID' if ctx.entity_name == 'vehicle' else 'RSU-VOXEL-GRID'
    voxel_props_key = 'voxel_detector_properties' if ctx.entity_name == 'vehicle' else 'rsu_voxel_detector_properties'
    
    voxel_size = ctx.metadata[voxel_props_key]['voxel_size']

    grid_origin = np.array(ctx.metadata[voxel_meta_key]['sensor2lidar_translation']) - \
        np.array([
            ctx.metadata[voxel_props_key]['range'],
            ctx.metadata[voxel_props_key]['range'],
            -ctx.metadata[voxel_props_key]['lower_limit']
        ])

    # Build the mesh once; shared across all view renders.
    combined_mesh = build_voxel_mesh(voxel_grid, voxel_size, grid_origin, LABEL_COLORS)

    camera_intrinsics_key = 'camera_intrinsics' if ctx.entity_name == 'vehicle' else 'rsu_camera_intrinsics'

    K = ctx.metadata[camera_intrinsics_key]

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    
    width = int(cx * 2)
    height = int(cy * 2)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    mat = o3d.visualization.rendering.MaterialRecord()
    
    mat.shader = 'defaultLit'

    # A single OffscreenRenderer is reused across all views and frames.
    renderer_key = f'{width}x{height}'

    if renderer_key not in voxel_renderers:
        with suppress_stderr():
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        voxel_renderers[renderer_key] = renderer
    else:
        renderer = voxel_renderers[renderer_key]
        
        renderer.scene.remove_geometry('voxels')

    renderer.scene.add_geometry('voxels', combined_mesh, mat)

    for view in VIEWS.keys():
        view2lidar = np.eye(4, dtype=np.float32)

        view2lidar[:3, :3] = Q(VIEWS[view]['view2lidar_rotation']).rotation_matrix
        view2lidar[:3, 3] = VIEWS[view]['view2lidar_translation']

        extrinsic = np.linalg.inv(view2lidar).astype(np.float64)

        renderer.setup_camera(intrinsic, extrinsic)

        img = np.asarray(renderer.render_to_image())

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(ctx.get_output_path('VOXEL3D', view), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def load_radar_data(ctx: VisualizationContextV2X):
    '''
    Load and combine data from all radars.
    
    Args:
        ctx: visualization context.
    '''
    point_cloud = []
    velocity = []

    RAD_NAME_LIST = RAD_NAME if ctx.entity_name == 'vehicle' else RSU_RAD_NAME
    
    for radar in RAD_NAME_LIST:
        radar2lidar = np.eye(4, dtype=np.float32)

        radar_name = radar if ctx.entity_name == 'vehicle' else f'RSU-{radar}'
        
        radar2lidar[:3, :3] = Q(ctx.metadata[radar_name]['sensor2lidar_rotation']).rotation_matrix
        radar2lidar[:3, 3] = ctx.metadata[radar_name]['sensor2lidar_translation']

        radar_points = np.load(ctx.frame_data[radar])['data']
        
        velocity.append(radar_points[:, -1])
        
        radar_points = radar_points[:, :-1]

        # Convert the radar values of depth, altitude angle, and azimuth angle
        # to x, y, and z coordinates.
        x = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.cos(radar_points[:, 2])
        y = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.sin(radar_points[:, 2])
        z = radar_points[:, 0] * np.sin(radar_points[:, 1])

        points = np.stack((x, y, z), axis=1)
        
        points_transformed = (radar2lidar @ np.append(points, np.ones((points.shape[0], 1)), 1).T)[:3].T

        point_cloud.append(points_transformed)
    
    point_cloud = np.concatenate(point_cloud, axis=0)
    velocity = np.concatenate(velocity, axis=0)
    
    # Calculate velocity-based colors.
    log_velocity = np.log(1.0 + np.abs(velocity))
    
    log_velocity_normalized = (log_velocity - log_velocity.min()) / (log_velocity.max() - log_velocity.min() + 1e-6)

    color = np.c_[
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 0]),
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 1]),
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 2])
    ]
    
    return point_cloud, color

