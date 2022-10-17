from misc.log_utils import log, dict_to_string
from model import multiviewmodel, pipeline
# from model import centertrack


def pipelineFactory(model_spec, data_spec):
    log.info(f"Building Pipeline")
    log.debug(f"Model spec: {dict_to_string(model_spec)}")

    det_model = multiviewmodel.MultiviewModel(model_spec, data_spec)
    full_pipeline = pipeline.MultiViewPipeline(det_model)

    return full_pipeline
