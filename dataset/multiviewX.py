import os
import json

import cv2
import numpy as np

from collections import defaultdict
from pathlib import Path

from dataset.utils import Bbox, Annotations, get_frame_from_file
from dataset.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log
class MultiviewXSet(SceneBaseSet):
    def __init__(self, data_conf, scene_config_file):
        super().__init__(data_conf, scene_config_file)

        self.root_path = Path(self.scene_config["data_root"])    

        self.frame_dir_path = self.root_path / "Image_subsets/"
        self.gt_dir_path = self.root_path / "annotations_positions/"

        self.nb_frames = len([frame_path for frame_path in (self.gt_dir_path).iterdir() if frame_path.suffix == ".json"])
        
        self.intrinsic_matrixs, self.extrinsic_matrixs = load_calibrations(self.root_path)

        self.world_origin_shift = self.scene_config["world_origin_shift"]
        self.groundplane_scale = self.scene_config["grounplane_scale"]

        log.debug(f"Dataset MultiviewX containing {self.nb_frames} frames from {self.get_nb_view()} views")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
        """
        Read and return undistoreted frame coresponding to index and view_id.
        The frame is return at the original resolution
        """

        frame_path = self.frame_dir_path / "C{:d}/{:04d}.png".format(view_id + 1, index)

        # log.debug(f"pomelo dataset get frame {index} {view_id}")
        frame = get_frame_from_file(frame_path)

        return frame

    def _get_gt(self, index, view_id):
        
        gt_path = self.gt_dir_path / "{:05d}.json".format(index)
        gt = read_json(gt_path)[view_id]

        return gt

    def _get_homography(self, view_id):
        """
        return the homography projecting the image to the groundplane.
        It takes into account potential resizing of the image and
        uses class variables:
        frame_original_size, homography_input_size, homography_output_size,
        as parameters
        """

        # update camera parameter to take into account resizing from the image before homography is applied
        K = geometry.update_K_after_resize(self.intrinsic_matrixs[view_id], self.frame_original_size, self.homography_input_size)
        H = get_ground_plane_homography(K, self.extrinsic_matrixs[view_id], self.world_origin_shift, self.groundplane_scale)
        
        return H
    
    def get_nb_view(self):
        return len(self.intrinsic_matrixs)
        
    def __len__(self):
        return self.nb_frames

def get_ground_plane_homography(intrinsic_matrix, extrinsic_matrix, world_origin_shift, groundplane_scale):
    #this function would return a homography that mapping image plane to ground plane[640, 1000]
    print("")
    return intrinsic_matrix\
    @np.delete(extrinsic_matrix, 2, 1)\
    @[[groundplane_scale,0,world_origin_shift[0]],\
      [0, groundplane_scale, world_origin_shift[1]],\
      [0,0,1]]\
    @[[4, 0, 0],[0, 4, 0],[0, 0, 1]]#reduce groundplane[640, 1000] to [160, 250]

def load_all_extrinsics(_lst_files):
    extrinsic_matrixs = []
    for _file in _lst_files:
        extrinsic_params_file = cv2.FileStorage(_file, flags=cv2.FILE_STORAGE_READ)
        rvec = extrinsic_params_file.getNode('rvec').mat()
        tvec = extrinsic_params_file.getNode('tvec').mat()
        extrinsic_params_file.release()
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        extrinsic_matrixs.append(extrinsic_matrix)
    return extrinsic_matrixs

def load_all_intrinsics(_lst_files):

    intrinsic_matrixs = []
    for _file in _lst_files:
        intrinsic_params_file = cv2.FileStorage(_file, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()
        intrinsic_matrixs.append(intrinsic_matrix)
    return intrinsic_matrixs



def load_calibrations(root_path):

    intrinsic_path_format = "calibrations/intrinsic/intr_{}.xml"
    extrinsic_path_format = "calibrations/extrinsic/extr_{}.xml"

    camera_id_to_name = ["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6"]

    intrinsic_pathes = [str(root_path / intrinsic_path_format.format(camera)) for camera in camera_id_to_name]
    extrinsic_pathes = [str(root_path / extrinsic_path_format.format(camera)) for camera in camera_id_to_name]

    extrinsic_matrixs = load_all_extrinsics(extrinsic_pathes)
    intrinsic_matrixs = load_all_intrinsics(intrinsic_pathes)

    return intrinsic_matrixs, extrinsic_matrixs


# Annotation = namedtuple('Annotation', ['xc', 'yc', 'w', 'h', 'feet', 'head', 'height', 'id', 'frame', 'view'])
def read_json(filename):
    """
    Decodes a JSON file & returns its content.
    Raises:
        FileNotFoundError: file not found
        ValueError: failed to decode the JSON file
        TypeError: the type of decoded content differs from the expected (list of dictionaries)
    :param filename: [str] name of the JSON file
    :return: [list] list of the annotations
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        with open(filename, 'r') as _f:
            _data = json.load(_f)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode {filename}.")
    if not isinstance(_data, list):
        raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
    if len(_data) > 0 and not isinstance(_data[0], dict):
        raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
        
    multi_view_gt = defaultdict(list)
    
    for person in _data:
        person_id = int(person["personID"])
        frame_id = int(Path(filename).stem)

        for bbox_v in person["views"]:
            if bbox_v["xmax"] == -1:
                continue
            view_id = int(bbox_v["viewNum"])
            xc = (bbox_v["xmax"] + bbox_v["xmin"]) / 2.0
            yc = (bbox_v["ymax"] + bbox_v["ymin"]) / 2.0
            w = bbox_v["xmax"] - bbox_v["xmin"]
            h = bbox_v["ymax"] - bbox_v["ymin"]

            bbox = Bbox(xc=xc, yc=yc, w=w, h=h)
            
            #Compute estimation for position of head and feet
            bbox_bottom_center = np.array([[xc], [h / 2.0 + yc]])
            bbox_top_center = np.array([[xc], [- h / 2.0 + yc]])
            
            multi_view_gt[view_id].append(Annotations(bbox=bbox, head=bbox_top_center,  feet=bbox_bottom_center, height=0, id=person_id, frame=frame_id, view=view_id))
            
    return multi_view_gt