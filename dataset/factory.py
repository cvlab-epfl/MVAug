from torch.utils.data import DataLoader, Subset

from configs.pathes import conf_path
from misc.log_utils import log, dict_to_string
from dataset import wildtrack, multiviewX
from dataset import heatmapbuilder
from dataset.basedataset import FlowSceneSet
from dataset.utils import get_train_val_split_index


wildtrack_config_file = conf_path["ROOT_PATH"] + "/data/SceneSetConfigs/wildtrack_train.json"
multiviewX_config_file = conf_path["ROOT_PATH"] + "/data/SceneSetConfigs/multiviewX_train.json"


def get_scene_set(data_arg, data_arg_dict_key = "dataset"):

    sceneset_list = []
  
    if "wild" in data_arg[data_arg_dict_key]:
        log.debug("Adding Wildtrack sequences to the dataset")
        sceneset_list.append(wildtrack.WildtrackSet(data_arg, wildtrack_config_file))
    if "multiviewx" in data_arg[data_arg_dict_key]:
        log.debug("Adding MultiViewX sequences to the dataset")
        sceneset_list.append(multiviewX.MultiviewXSet(data_arg, multiviewX_config_file))

    return sceneset_list    
 
def get_datasets(data_arg, data_arg_dict_key, use_aug):

    sceneset_list = get_scene_set(data_arg, data_arg_dict_key)
    
    if data_arg["hm_type"] == "density":
        data_arg["hm_builder"] = heatmapbuilder.gaussian_density_heatmap
    elif data_arg["hm_type"] == "center":
        data_arg["hm_builder"] = heatmapbuilder.gausian_center_heatmap
    elif data_arg["hm_type"] == "constant":
        data_arg["hm_builder"] = heatmapbuilder.constant_center_heatmap
    else:
        log.error(f"Unknown heatmap type {data_arg['hm_type']}")

    datasets = [FlowSceneSet(sceneset, data_arg, use_aug) for sceneset in sceneset_list] 

    return datasets


def get_dataloader(data_arg):
    log.info(f"Building Datasets")
    log.debug(f"Data spec: {dict_to_string(data_arg)}")

    train_datasets = get_datasets(data_arg, "dataset", data_arg["aug_train"])
    train_datasets_no_aug = get_datasets(data_arg, "dataset", False)

    val_datasets = get_datasets(data_arg, "eval-dataset", False)

    train_val_splits = [get_train_val_split_index(dataset, data_arg["split_proportion"]) for dataset in train_datasets]

    train_dataloaders = list()
    val_dataloaders = list()

    for dataset, dataset_no_aug, train_val_split in zip(train_datasets, train_datasets_no_aug, train_val_splits):
        #Add train dataset (possibly subset) to train dataloaders
        train_dataloaders.append(DataLoader(
            Subset(dataset, train_val_split[0]),
            shuffle=data_arg["shuffle_train"],
            batch_size=data_arg["batch_size"],
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            num_workers=data_arg["num_workers"]
            )
        )
        
        #Add split part of the train dataset to validation dataloaders
        if data_arg["split_proportion"] < 1:
            val_dataloaders.append(DataLoader(
                Subset(dataset_no_aug, train_val_split[1]),
                shuffle=False,
                batch_size=data_arg["batch_size"],
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                num_workers=data_arg["num_workers"]
                )
            )
    
    #add validation only dataset to val dataloaders
    for dataset in val_datasets:
        val_dataloaders.append(DataLoader(
                Subset(dataset, list(range(len(dataset)))),
                shuffle=False,
                batch_size=data_arg["batch_size"],
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                num_workers=data_arg["num_workers"]
            )
        )


    return train_dataloaders, val_dataloaders
