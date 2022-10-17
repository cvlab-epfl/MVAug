import random

import numpy as np
import torch

from torchvision import transforms

from augmentation.homographyaugmentation import HomographyDataAugmentation
from dataset.utils import aggregate_multi_view_gt_points, generate_scene_roi_from_view_rois, generate_mask_from_polygon_perimeter, get_augmentation
from misc import geometry
from misc.utils import PinnableDict, stack_tensors, listdict_to_dictlist
from misc.log_utils import log

class FlowSceneSet(torch.utils.data.Dataset):
    
    def __init__(self, scene_set, data_conf, use_augmentation, compute_flow_stat=False):
    
        self.scene_set = scene_set
        
        self.nb_view = len(data_conf["view_ids"])
        self.nb_frames = data_conf["nb_frames"]
        self.frame_interval = data_conf["frame_interval"]
        self.generate_flow_hm = False

        log.debug(f"Flow scene set containing {self.nb_view} view and {len(self.scene_set)} frames, will process {self.nb_frames} frame at a time")

        #original image and gt dimension needed for further rescaling
        self.frame_original_size = self.scene_set.frame_original_size
        self.frame_input_size = data_conf["frame_input_size"]

        #homography information are needed to generate hm
        self.homography_input_size = data_conf["homography_input_size"]
        self.homography_output_size = data_conf["homography_output_size"]

        self.hm_builder = data_conf["hm_builder"]
        self.hm_radius = data_conf["hm_radius"]
        self.hm_size = data_conf["hm_size"]
        self.hm_image_size = data_conf["hm_image_size"]
        
        #Reduce length by two to be able to return triplet of frames
        self.total_number_of_frame = len(self.scene_set) - (self.nb_frames-1)*self.frame_interval
        
        self.img_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             transforms.Resize(self.frame_input_size)
            ]
        )

        self.use_augmentation = use_augmentation

        #list of augmentation and their probability
        self.view_based_augs = [(HomographyDataAugmentation(aug), prob) if aug is not None else (None, prob) for aug, prob in get_augmentation(data_conf["views_based_aug_list"], self.frame_input_size)]
        self.scene_based_augs = [(HomographyDataAugmentation(aug), prob) if aug is not None else (None, prob) for aug, prob in get_augmentation(data_conf["scene_based_aug_list"], self.hm_size)]

        if self.use_augmentation:
            log.info(f"View base augmentation {self.view_based_augs}")
            log.info(f"Scene base augmentation {self.scene_based_augs}")


        if sum([x[1] for x in self.scene_based_augs]) != 1:
            log.warning(f"Scene based augmentation probability should sum up to one but is {sum([x[1] for x in self.scene_based_augs])}")
        if sum([x[1] for x in self.view_based_augs]) != 1:
            log.warning(f"View based augmentation probability should sum up to one but is {sum([x[1] for x in self.view_based_augs])}")

        if compute_flow_stat:
            self.compute_flow_stat()


    def __getitem__(self, index):
        multi_view_data = list()
        
        scene_based_aug = self.select_augmentation(*zip(*self.scene_based_augs))
        ROIs_view = list()

        for view_id in range(self.nb_view):
            frame_dict = {}
            frame_dict["view_id"] = view_id

            view_based_aug = self.select_augmentation(*zip(*self.view_based_augs))

            for frame_id in range(self.nb_frames):
                true_frame_id = index+frame_id*self.frame_interval
                frame, homography = self.scene_set.get(true_frame_id, view_id)

                #normalize and rescale frame
                frame = self.img_transform(frame)

                homography = torch.from_numpy(homography).float()

                gt_points_image, person_id_image = self.scene_set.get_gt_image(true_frame_id, view_id)
                
                frame, homography, gt_points_image, person_id_image = self.apply_view_based_augmentation(frame, homography, gt_points_image, person_id_image, view_based_aug)

                hm_image  = self.build_heatmap(gt_points_image, self.hm_image_size, self.hm_radius).squeeze(0)

                frame_dict[f"frame_{frame_id}"] = frame
                frame_dict[f"frame_{frame_id}_true_id"] = true_frame_id
                frame_dict[f"hm_image_{frame_id}"] = hm_image
                frame_dict[f"gt_points_image_{frame_id}"] = gt_points_image
                frame_dict[f"person_id_image_{frame_id}"] = person_id_image

            homography = self.apply_scene_based_aug(homography, scene_based_aug)

            frame_dict["homography"] = homography

            if view_based_aug is not None:
                ROIs_view.append(view_based_aug.augment_gt_point_view_based(self.scene_set.get_ROI(view_id), None, filter_out_of_frame=False, frame_size=self.frame_original_size)[0])
            else:
                ROIs_view.append(self.scene_set.get_ROI(view_id))

            ROI_view_curr = geometry.rescale_keypoints(ROIs_view[-1], self.frame_original_size, self.hm_image_size)
            frame_dict["ROI_image"] = torch.from_numpy(generate_mask_from_polygon_perimeter(ROI_view_curr, self.hm_image_size))
            
            multi_view_data.append(frame_dict)

        multi_view_data = listdict_to_dictlist(multi_view_data)
        multi_view_data = stack_tensors(multi_view_data)

        #adding groundtuth shared between the view
        for frame_id in range(self.nb_frames):
            true_frame_id = index+frame_id*self.frame_interval

            gt_points, person_id = aggregate_multi_view_gt_points(multi_view_data[f"gt_points_image_{frame_id}"], multi_view_data[f"person_id_image_{frame_id}"], multi_view_data["homography"], self.hm_image_size, self.homography_input_size, self.homography_output_size, self.hm_size)
            gt_points = np.rint(gt_points)

            hm  = self.build_heatmap(gt_points, self.hm_size, self.hm_radius)

            multi_view_data[f"hm_{frame_id}"] = hm
            multi_view_data[f"gt_points_{frame_id}"] = gt_points
            multi_view_data[f"person_id_{frame_id}"] = person_id


        ROI_mask, ROI_boundary = generate_scene_roi_from_view_rois(ROIs_view, multi_view_data["homography"], self.frame_original_size, self.homography_input_size, self.homography_output_size, self.hm_size)
        ROI_mask = torch.from_numpy(ROI_mask).float().unsqueeze(0)
        boundary_mask = torch.from_numpy(ROI_boundary).float().unsqueeze(0)  

        #adding additional scene data (roi, and boundary)
        multi_view_data["ROI_mask"] = ROI_mask
        multi_view_data["ROI_boundary_mask"] = boundary_mask
        multi_view_data["scene_id"] = self.scene_set.scene_id

        return multi_view_data

    def select_augmentation(self, aug_list, aug_prob):
        if self.use_augmentation:
            aug = random.choices(aug_list, weights=aug_prob)[0]
        else:
            aug = None

        if aug is not None:
            aug.reset()

        return aug

    def apply_view_based_augmentation(self, frame, homography, gt_points_image, person_id_image, view_based_aug):
        if view_based_aug is None:
            return frame, homography, gt_points_image, person_id_image

        frame = view_based_aug(frame)
        homography = view_based_aug.augment_homography_view_based(homography, self.homography_input_size)
        gt_points_image, person_id_image = view_based_aug.augment_gt_point_view_based(gt_points_image, person_id_image, frame_size=self.hm_image_size)

        return frame, homography, gt_points_image, person_id_image

    def apply_scene_based_aug(self, homography, scene_based_aug):
        if scene_based_aug is None:
            return homography
        
        homography = scene_based_aug.augment_homography_scene_based(homography, self.homography_output_size)

        return homography
    
    def build_heatmap(self, gt_points, hm_size, hm_radius):
        
        if len(gt_points) != 0:
            gt_points = np.rint(gt_points).astype(int)
        hm = self.hm_builder(hm_size, gt_points, hm_radius)
        
        return hm.unsqueeze(0)

    
    def __len__(self):
        return self.total_number_of_frame

    @staticmethod
    def collate_fn(batch):
        #Merge dictionnary
        batch = listdict_to_dictlist(batch)
        batch = stack_tensors(batch)

        collate_dict = PinnableDict(batch)

        return collate_dict

