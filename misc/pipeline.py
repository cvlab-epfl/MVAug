import time

import torch

from misc.log_utils import log

class MultiViewPipeline(torch.nn.Module):

    def __init__(self, people_flow, object_tracker, flow_consistency):
        super(MultiViewPipeline, self).__init__()

        self.people_flow = people_flow

    def forward(self, input_data):
        time_stat = dict()
        end = time.time()

        #Run people flow
        pred = None
        if self.people_flow is not None:
            pred = self.people_flow(input_data)
        
        time_stat["flow_time"] = time.time() - end
        end = time.time()

        return {"pred": pred, "time_stats":time_stat}
