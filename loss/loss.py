import torch

from misc.log_utils import log


class MSEwithROILoss(torch.nn.Module):
    def __init__(self):
        super(MSEwithROILoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, pred, target, roi_mask=None):

        loss = self.mse(pred, target)

        if roi_mask is not None:
            loss = loss * roi_mask

        return loss.sum()



class FlowLossProb(torch.nn.Module):
    def __init__(self, model_spec, data_spec, stats_name=""):
        super(FlowLossProb, self).__init__()

        self.criterion = MSEwithROILoss()#torch.nn.MSELoss(size_average=False) #reduction='sum'

        self.nb_view = len(data_spec["view_ids"])

        self.stats_name = stats_name
        self.use_image_pred_loss = model_spec["image_pred"]


    def forward(self, input_data, output_flow):
        stats = {}

        roi_mask = input_data["ROI_mask"]
        hm_0 = input_data["hm_0"]

        loss_pred = self.criterion(output_flow["pred_0"], hm_0, roi_mask)

        stats = {**stats,
            self.stats_name + "loss_pred" : loss_pred.item()
            }

        #loss on each view
        if self.use_image_pred_loss:
            loss_framepred = list()
            for v in range(self.nb_view):
                loss_framepred_0_v = self.criterion(output_flow[f"framepred_0_v{v}"], input_data["hm_image_0"][:,v], input_data["ROI_image"][:,v]) / self.nb_view 
                
                loss_framepred.append(loss_framepred_0_v)

                stats = {**stats,
                    self.stats_name + "loss_framepred_0_v{v}" : loss_framepred_0_v.item(),
                    }
            
            loss_framepred = sum(loss_framepred)

            stats = {**stats,
                self.stats_name + "loss_framepred" : loss_framepred.item()
                }
        else:
            loss_framepred = 0

            for v in range(self.nb_view):
                stats = {**stats,
                    self.stats_name + "loss_framepred_0_v{v}" : 0,
                    }

            stats = {**stats,
                self.stats_name + "loss_framepred" : 0
                }


        total_loss = loss_pred + loss_framepred


        stats = {**stats,
            "loss_"+self.stats_name : total_loss.item(),
            }

        return {"loss":total_loss, "stats":stats}


class MultiLoss(torch.nn.Module):
    def __init__(self, pred_loss):
        super(MultiLoss, self).__init__()

        self.pred_loss = pred_loss

    def forward(self, input_data, output):

        loss = 0
        stats = {}

        if self.pred_loss is not None:
            flow = self.pred_loss(input_data, output["pred"])
            loss = loss + flow["loss"]
            stats = {**stats, **flow["stats"]}

        stats = {**stats, "loss":loss}

        return {"loss":loss, "stats":stats}