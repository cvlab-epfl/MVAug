import cv2

import numpy as np
import torch 

from misc.log_utils import log

from model.multimodel import MultiNet


class MultiviewModel(torch.nn.Module):
    def __init__(self, model_spec, data_spec):
        super().__init__()


        self.nb_hm = 1

        # self.ground_hm_size = data_spec["hm_size"]#
        self.homography_input_size = data_spec["homography_input_size"]
        self.homography_output_size = data_spec["homography_output_size"]

        self.hm_size = data_spec["hm_size"]
        self.nb_view = len(data_spec["view_ids"])

        
        self.model = MultiNet(
            self.hm_size, 
            self.homography_input_size, 
            self.homography_output_size, 
            nb_ch_out=self.nb_hm,
            model_image_pred=model_spec["image_pred"], 
            nb_view=len(data_spec["view_ids"])
            )

    def split_views_into_dict(self, pred_views, pred_name, dim_view=1):
        if pred_views is None:
            return {}

        views_dict = {}

        assert pred_views.shape[dim_view] == self.nb_view, "Number of view is not equal to number of image plane prediction"
        for v in range(pred_views.shape[dim_view]):
            views_dict[f"{pred_name}_v{v}"] = torch.index_select(pred_views, dim_view, torch.tensor([v], device=pred_views.device))

        return views_dict


    def forward(self, input_data):

        pred_0, frame_pred_0  = self.model(input_data["frame_0"], input_data["homography"], input_data["ROI_mask"]) #flow_0_1f, flow_1_0b


        # roi_mask = input_data["ROI_mask"]
        # mask_boundry = input_data["ROI_boundary_mask"]
        
        frame_pred_0_dict = self.split_views_into_dict(frame_pred_0, "framepred_0")


        output = {
            "pred_0":pred_0,
            **frame_pred_0_dict,
        }

        return output