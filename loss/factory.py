from misc.log_utils import log, dict_to_string
from loss import loss


def get_loss(model_spec, data_spec):
    log.info(f"Building Loss")
    
    det_loss = loss.FlowLossProb(model_spec, data_spec)
        
    global_criterion = loss.MultiLoss(det_loss)

    return global_criterion