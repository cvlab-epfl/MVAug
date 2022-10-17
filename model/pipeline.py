import time

import torch

from misc.log_utils import log
from misc.utils import PinnableDict

class MultiViewPipeline(torch.nn.Module):

    def __init__(self, multiview_model):
        super(MultiViewPipeline, self).__init__()

        self.multiview_model = multiview_model

    def forward(self, input_data):
        time_stat = dict()
        end = time.time()

        #Run people flow
        pred = None
        if self.multiview_model is not None:
            pred = self.multiview_model(input_data)
        
        time_stat["flow_time"] = time.time() - end
        end = time.time()


        return PinnableDict({"pred": pred, "time_stats":time_stat})
