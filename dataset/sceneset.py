from collections import defaultdict
import uuid

import numpy as np
import shapely

from dataset.utils import generate_scene_roi_from_view_rois, read_json_file, aggregate_multi_view_gt, extract_points_from_gt, write_dict_as_json
from misc import geometry
from misc.log_utils import log


class SceneBaseSet():
    """
    Parent class containing all the information regarding a single scene:
        - frames
        - groundtruth
        - Homography to groundplane
        - Region of interest
        - Occlusion mask

    The scene can contains multiple view. This class provide generic implementation and helper function.
    Specific function can be overwritten by child class: for example homography can be computed from camera calibration or point in the image depending on the type of scene.
    """

    def __init__(self, data_conf, scene_config_file):
        super(SceneBaseSet, self).__init__()
        self.data_conf = data_conf
        self.scene_config_file = scene_config_file
        self.scene_config = read_json_file(self.scene_config_file)
        
        self.frame_original_size = self.scene_config["frame_size"]
        self.fps = self.scene_config["fps"]

        self.frame_input_size = data_conf["frame_input_size"]
        self.homography_input_size = data_conf["homography_input_size"]
        self.homography_output_size = data_conf["homography_output_size"]
        self.hm_size = data_conf["hm_size"]
        self.hm_image_size = data_conf["hm_image_size"]

        self.view_IDs = data_conf["view_ids"]

        self.desired_fps = data_conf["desired_fps"]

        if "scene_id" in self.scene_config:
            self.scene_id = self.scene_config["scene_id"]
        else:
            self.scene_id = uuid.uuid4()

    def generate_scene_elements(self):
        self.homographies = [self._get_homography(view_id) for view_id in self.view_IDs]

        #by default we use the original fps
        if self.desired_fps == -1:
            self.desired_fps = self.fps
        
        #it is only possible to reduce the framerate by skiping intermediate frame
        assert self.desired_fps <= self.fps

        #the desired fps rate must be a factor of the original fps
        assert (self.fps /self.desired_fps) == int((self.fps /self.desired_fps))

        self.reducing_factor = int(self.fps / self.desired_fps)

        self.ROIs = self.load_views_ROIs()
        self.occluded_areas = [np.array(self.scene_config["occluded_area_corner_points_per_view"][view_id]) for view_id in self.view_IDs]      

        self.scene_ROI, self.scene_ROI_boundary = generate_scene_roi_from_view_rois(self.ROIs, self.homographies, self.frame_original_size, self.homography_input_size, self.homography_output_size, self.hm_size)
    
    def load_views_ROIs(self):
        if "roi_generated_per_view" in self.scene_config:
            ROIs_perimeter = [np.array(self.scene_config["roi_generated_per_view"][view_id]) for view_id in self.view_IDs]
        else:
            log.warning("ROI per view is not defined in sceneconfig file, generating it from groundtruth annotation")
            #if not already computed generate roi from gt points
            points_ground_per_view = defaultdict(list)
            for index in range(len(self)):
                for v_id in range(len(self.view_IDs)):
                    gt_points, person_id = self.get_gt_image(index, v_id)
                    ground_points = geometry.project_image_points_to_groundview(gt_points, self.homographies[v_id])

                    points_ground_per_view[v_id].extend(ground_points)

            ROIs_perimeter = list()
            
            for v_id in range(len(self.view_IDs)):
                point_view = points_ground_per_view[v_id]
                image_rois = shapely.geometry.MultiPoint(point_view).convex_hull.buffer(1)
                roi_ground_perimeter = np.array(image_rois.exterior.xy).T

                image_points = geometry.project_image_points_to_groundview(roi_ground_perimeter, np.linalg.inv(self.homographies[v_id]))
                image_points = geometry.rescale_keypoints(image_points, self.hm_image_size, self.frame_original_size)
                ROIs_perimeter.append(image_points)
            
            self.scene_config["roi_generated_per_view"] = [rois.tolist() for rois in ROIs_perimeter]
            write_dict_as_json(self.scene_config, self.scene_config_file)
        
        return ROIs_perimeter

    def log_init_completed(self):
        log.debug(f"Dataset from directory {self.scene_config['data_root']} containing {len(self)} frames loaded")
        log.debug(f"Dataset contains {self.get_nb_view()} view, the following are used: {self.view_IDs}")
        
        if self.reducing_factor != 1:
            log.debug(f"Desired fps smaller than original one ({self.fps}). Set fps to {self.desired_fps} which corespond to a reducing factor of {self.reducing_factor}")

    def get(self, index, view_id):
        # correct index to match desired fps
        index = index * self.reducing_factor

        frame = self.get_frame(index, view_id)
        homography = self.get_homography(view_id)

        return frame, homography

    def get_frame(self, index, view_id):
        return []

    def get_gt(self, index):
        # correct index to match desired fps
        index = index * self.reducing_factor

        gts = [self._get_gt(index, view_id) for view_id in self.view_IDs] 
        aggregated_gt = aggregate_multi_view_gt(gts, self.homographies, self.frame_original_size, self.homography_input_size, self.homography_output_size, self.hm_size)
        
        return aggregated_gt

    def get_gt_image(self, index, view_id):
        index = index * self.reducing_factor

        gt = self._get_gt(index, view_id)
        gt_points, person_id = extract_points_from_gt(gt, self.hm_image_size, gt_original_size=self.frame_original_size)

        return gt_points, person_id

    def get_homography(self, view_id):
        return self.homographies[view_id]

    def get_ROI(self, view_id):
        return self.ROIs[view_id]

    def get_scene_ROI(self):
        return self.scene_ROI, self.scene_ROI_boundary

    def get_occluded_area(self, view_id):
        return self.occluded_areas[view_id]

    def get_nb_used_view(self):
        return len(self.view_IDs)

    def get_length(self):
        log.error("Abstract class get_lentgth as be called, it should be overwriten it in child class")
        return -1

    def __len__(self):
        length = int(self.get_length() // self.reducing_factor)
        return length