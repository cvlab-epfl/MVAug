import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from ctypes import c_bool
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import torch

from configs.arguments import get_config_dict
from dataset import factory as data_factory
from loss import factory as loss_factory
from misc import detection
from misc.log_utils import DictMeter, batch_logging, dict_to_string, log
from misc.metric import compute_mot_metric_from_det, save_detection_for_evaluation
from misc.utils import listdict_to_dictlist
from model import factory as model_factory

warnings.filterwarnings("ignore", category=UserWarning)

class Evaluator():
    def __init__(self, val_loaders, model, criterion, epoch, conf):
        super(Evaluator, self).__init__()

        self.val_loaders = val_loaders
        self.model = model
        self.criterion = criterion

        self.epoch = epoch
        self.conf = conf

        self.nb_views = len(conf["data_conf"]["view_ids"])
        self.total_nb_frames = sum([len(val_loader) for val_loader in self.val_loaders])

        self.loss_to_print = conf["training"]["loss_to_print"]
        self.metric_to_print = conf["training"]["metric_to_print"]

        #visualization and metric parameters
        self.detection_to_evaluate = conf["training"]["detection_to_evaluate"]
        self.use_nms = True
        self.nms_kernel_size = 3
        self.metric_threshold = conf["data_conf"]["metric_threshold"]

        self.best_valid_result = None

    def reset(self):
        self.stats_meter = DictMeter()
        self.epoch_result_dicts = list()
        self.model.eval()

        self.is_best = False

    def run(self, epoch):
        
        self.epoch = epoch

        self.reset()

        end = time.time()
        for s, val_loader in enumerate(self.val_loaders):
            
            #reset some shared variable when switching scene
            step_dict = None

            for f, input_data in enumerate(val_loader):
                #global index of current frame
                i = sum([len(self.val_loaders[x]) for x in range(s)]) + f

                input_data = input_data.to(self.conf["device"])
                
                data_time = time.time() - end

                with torch.no_grad():
                    output_data = self.model(input_data)
                    
                    end2 = time.time()
                    if ("eval_metric" in self.conf["main"] and self.conf["main"]["eval_metric"]) or "eval_metric" not in self.conf["main"]:
                        criterion_output = self.criterion(input_data, output_data)
                    else:
                        criterion_output = {"stats":{}}

                    criterion_time = time.time() - end2

                    #put all the output in cpu to free gpu memory for the remaining of validation
                    output_data = output_data.to("cpu")
                    input_data = input_data.to("cpu")
                    
                    # Extract detected point
                    processed_results, output_data = self.post_process_heatmap(input_data, output_data)
                    #Compute detection and count metric if groundtruth is available
                    metric_stats = self.compute_metric(input_data, output_data, processed_results)
                    #Store data needed for tracking and visualization
                    self.store_step_dict(input_data, output_data, processed_results, metric_stats)
                
                batch_time = time.time() - end

                epoch_stats_dict = {**criterion_output["stats"], **metric_stats, **output_data["time_stats"], "batch_time":batch_time, "data_time":data_time, "criterion_time":criterion_time, "optim_time":0}
                self.stats_meter.update(epoch_stats_dict)
                
                if i % self.conf["main"]["print_frequency"] == 0 or i == (self.total_nb_frames - 1):
                    batch_logging(self.epoch, i, self.total_nb_frames, self.stats_meter, loss_to_print=self.loss_to_print, metric_to_print=self.metric_to_print, validation=True)
                
                end = time.time()
                #When we have accumulated max_tracklet_lenght step dict or reach the en dof dataset we push the step dict to the tracker process

                del input_data
        
        #convert the list of result to a dict
        self.combine_step_dict()
        self.compute_epoch_metrics()

        stats = {**self.stats_meter.avg()}

        del self.epoch_result_dicts

        #Using AUC for pose estimation to compare model
        if self.best_valid_result is None or ((stats["moda_pred_0"]) > self.best_valid_result):
            self.best_valid_result = (stats["moda_pred_0"])
            self.is_best = True
        
        return {"stats":stats}


    def post_process_heatmap(self, input_data, output_data):

        #post process detection heatmap from self.detection_to_evaluate list
        processed_results = dict()

        #Set prediction outside of ROI to zero
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] != "framepred":
                output_data["pred"][det_k] = output_data["pred"][det_k] * input_data["ROI_mask"]
            else:
                v_id = int(det_k.split('_')[2][1])
                output_data["pred"][det_k] = output_data["pred"][det_k] * input_data["ROI_image"][:,v_id]

        for det_k in self.detection_to_evaluate:
            scores_flow, pred_point_flow = detection.decode_heatmap(output_data["pred"][det_k], self.nms_kernel_size, self.use_nms, threshold="auto")

            processed_results[det_k+"_points"] = pred_point_flow
            processed_results[det_k+"_scores"] = scores_flow

        return processed_results, output_data

    
    def compute_metric(self, input_data, output_data, processed_results):
        metrics_dict = dict()
        

        #For all the detection we compute MODA metric
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] == "framepred":
                f_id = det_k.split("_")[1][0]
                v_id = int(det_k.split('_')[2][1])
                gt_points = [input_data[f"gt_points_image_{f_id}"][0][v_id]]
            else:
                gt_points = input_data["gt_points_"+det_k.split("_")[1][0]]

            metric_k = compute_mot_metric_from_det(gt_points, [processed_results[det_k+"_points"]], self.metric_threshold)
            metrics_dict.update({k+"_"+det_k: v for k,v in metric_k.items()})

        metric_stats = {**metrics_dict}

        return metric_stats

    def store_step_dict(self, input_data, output_data, processed_results, metric_stats):
        """
        Store combination of input and prediction to generate tracker, metrics, and visualiztion
        We assume batchsize is 1 and only take the first element of the batch and the first view
        """
        step_dict = {}

        #Adding detection
        for det_k in self.detection_to_evaluate:
             step_dict[det_k] = output_data["pred"][det_k][0,0]
             step_dict[det_k+"_points"] = processed_results[det_k+"_points"]
             step_dict[det_k+"_scores"] = processed_results[det_k+"_scores"]

        #Adding frame and gt to step dict
        for frame_id in range(self.conf["data_conf"]["nb_frames"]):
            step_dict[f"gt_points_{frame_id}"] = input_data[f"gt_points_{frame_id}"][0].astype(int)
            step_dict[f"person_id_{frame_id}"] = input_data[f"person_id_{frame_id}"][0]
            step_dict[f"hm_{frame_id}"] = input_data[f"hm_{frame_id}"][0]
            # step_dict[f"frame_image_{frame_id}"] = visualization.inverse_img_norm(input_data[f"frame_{frame_id}"][:,0])
            # step_dict[f"gt_points_image_{frame_id}"] = input_data[f"gt_points_image_{frame_id}"][0]
            step_dict[f"frame_{frame_id}_true_id"] = input_data[f"frame_{frame_id}_true_id"][0]

            for v_id in range(self.nb_views):
                step_dict[f"gt_points_image_{frame_id}_v{v_id}"] = input_data[f"gt_points_image_{frame_id}"][0][v_id]
                step_dict[f"roi_image_v{v_id}"] = input_data["ROI_image"][0,v_id]

        
        step_dict["metric_stats"] = metric_stats
        step_dict["roi"] = input_data["ROI_mask"][0]
        step_dict["scene_id"] = input_data["scene_id"][0]
        
        step_dict["mask_boundary"] = input_data["ROI_boundary_mask"][0]
        step_dict["homography"] = input_data["homography"][0]

        self.epoch_result_dicts.append(step_dict)

    def combine_step_dict(self):
        self.epoch_result_dicts = listdict_to_dictlist(self.epoch_result_dicts)

    def compute_epoch_metrics(self):
        
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] == "framepred":
                f_id = det_k.split("_")[1][0]
                v_id = int(det_k.split('_')[2][1])
                gt_points = self.epoch_result_dicts[f"gt_points_image_{f_id}_v{v_id}"]
            else:
                gt_points = self.epoch_result_dicts["gt_points_"+det_k.split("_")[1][0]]
    
            det_points = self.epoch_result_dicts[det_k+"_points"]

            #Compute metric using pymotmetric

            metric_k = compute_mot_metric_from_det(gt_points, det_points, self.metric_threshold)

            metrn =   ["moda", "precision", "recall"]
            metrics = [metric_k[metric] for metric in metrn]
            
            max_char_len = [max(len(metrn), len(f'{metrc:.3f}')) for metrn, metrc in zip(metrn, metrics)]

            str_metric_lgd = f"Epoch Metric {det_k} {'  '.join([f'{metr:<{padding}}' for metr, padding in zip(metrn, max_char_len)])}"
            str_metric = f"Epoch Metric {det_k} {'  '.join([f'{metric:<{padding}.3f}' for metric, padding in zip(metrics, max_char_len)])}"

            log.info("\t" + str_metric_lgd)
            log.info("\t" + str_metric)

            #save results for matlab evaluation
            save_detection_for_evaluation(gt_points, det_points, self.metric_threshold, self.conf["training"]["ROOT_PATH"], self.conf["main"]["name"], self.epoch, det_k)

        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    ####### Configuration #######
    parser.add_argument("checkpoint_path", help='path to the checkpoint to evaluate')
    parser.add_argument("-vid", '--video_path', dest="video_path", default="", help="Path to video to use for evaluation")
    parser.add_argument("-n", '--name', dest="name", default="", help="eval name (placehodler")
    parser.add_argument("-vids", '--video_sequence', dest="video_sequence", type=float, nargs='+', default=(0, 1), help="interval of the video to use in the evaluation by default full video interval (0,1)")
    parser.add_argument("-dev", "--device", dest="device", help="select device to use either cpu or cuda", default="cuda")
    parser.add_argument("-bs", '--batch_size', dest="batch_size", type=int, default=1,  help="The size of the batches")
    parser.add_argument("-vis", '--eval_visual', dest="eval_visual", action='store_true', default=False, help="Create video visualization from evaluation outputs")
    parser.add_argument("-dmet", '--disable_metric', dest="disable_metric", action='store_true', default=False, help="Avoid computing metric during evaluation")
    parser.add_argument('-tr', "--train_eval", dest="train_eval", action='store_true', default=False, help="Run evaluation on the training set")
    parser.add_argument("-splt", "--split_proportion", dest="split_proportion", type=float, default=-1, help="Train val split proportion the first split_proportion percent of the frames are used for training, the rest for validation")
    parser.add_argument("-dset", "--dataset", dest="dataset", default=None, nargs='*', choices=["PETS", "PETSeval",  "Parkinglot", "wild", "pomswa", "pomswatrain", "pomswatrain2", "pomrayeval3", "mot20train1", "mot20train2", "mot20train3", "mot20train5", "mot20test4", "mot20test6", "mot20test7", "mot20test8"], help='Dataset to use for Training')
    parser.add_argument("-mtl", "--max_tracklet_lenght", dest="max_tracklet_lenght", help="Number of element processed between print", default=None)
    parser.add_argument("-mcon", "--model_consistency", dest="model_consistency", action="store_true", default=False, help="By default ground projection time cycle consistency is enforced through a loss, if true it is enforced by network architecture instead")
    parser.add_argument("-motrf", '--mot_result_file', dest="mot_result_file", action="store_true", default=False, help="if true evaluation script will generate a file containing the track result ready to be submitted to mot website")
    parser.add_argument("-dtrack", '--disable_tracker', dest="disable_tracker", action="store_false", default=True, help="if flag is used it disable the use of tracker during evaluation")


    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    checkpoint_dict = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)

    #remove checkpint path from arg list
    del sys.argv[1]

    config = get_config_dict(checkpoint_dict["conf"])
    log.debug("loaded conf: " + dict_to_string(config))
    
    if args.max_tracklet_lenght is not None:
        config["main"]["max_tracklet_lenght"] = int(args.max_tracklet_lenght)
    
    if not args.model_consistency:
        config["model_conf"]["model_consistency"] = args.model_consistency
    
    if not (args.split_proportion == -1):
        config["data_conf"]["split_proportion"] = args.split_proportion

    if args.dataset is not None:
        config["data_conf"]["dataset"] = []
        config["data_conf"]["eval-dataset"] = args.dataset

    config["data_conf"]["batch_size"] = args.batch_size
    config["data_conf"]["shuffle_train"] = False
    config["data_conf"]["video_sequence"] = args.video_sequence
    config["training"]["eval_visual"] = args.eval_visual
    config["main"]["eval_metric"] = not(args.disable_metric)
    config["data_conf"]["mot_result_file"] = args.mot_result_file

    config["main"]["print_frequency"] = 100
    ##################
    ### Initialization
    ##################
    config["device"] = torch.device('cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")

    end = time.time()
    log.info("Initializing model ...")
    
    model = model_factory.pipelineFactory(config["model_conf"], config["data_conf"])
    # log.debug(model.state_dict())
    model.load_state_dict(checkpoint_dict["state_dict"])
    # log.debug(model.state_dict())
    model.to(config["device"])

    # for param in join_emb.cap_emb.parameters():
    #     param.requires_grad = False

    log.info(f"Model initialized in {time.time() - end} s")

    
    end = time.time()
    log.info("Loading Data ...")
    
    if args.video_path:
        config["data_conf"]["dataset"] = "video"
        config["data_conf"]["video_path"] = args.video_path
        config["data_conf"]["split_proportion"] = 0
        config["main"]["eval_metric"] = False

    train_dataloader, val_dataloader = data_factory.get_dataloader(config["data_conf"])

    if args.train_eval:
        dataloader = train_dataloader
        
    else:
        dataloader = val_dataloader

    log.info(f"Data loaded in {time.time() - end} s")

    criterion = loss_factory.get_loss(config["model_conf"], config["data_conf"])

    ##############
    ### Evaluation
    ##############

    end = time.time()
    log.info(f"Beginning validation")
    evaluator = Evaluator(dataloader, model, criterion, checkpoint_dict["epoch"], config)

    valid_results = evaluator.run(checkpoint_dict["epoch"])
    
    log.info(f"Validation completed in {time.time() - end}s")


#python evaluation.py weights/model_425/model_425_epoch_30.pth.tar -dset mot20train1 mot20train2  -motrf -vis
