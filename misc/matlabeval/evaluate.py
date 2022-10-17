import os

import numpy as np
import argparse

import matlab.engine


ROOT_PATH = "."

def evaluate(res_fpath, gt_fpath, dataset, metric_threshold, eng=None):

    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.cd(ROOT_PATH+'/misc/matlabeval/motchallenge-devkit')

    res = eng.evaluateDetection(res_fpath, gt_fpath, dataset, metric_threshold)
    recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
   
    return recall, precision, moda, modp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("name", help='Name of the model to evaluate')
    parser.add_argument("-dett", "--detection_type", dest="detection_type", type=str, default="pred_0", help="Distance in groundplane pixel where detection are considered positive when computing metrics")
    parser.add_argument("-mth", "--metric_threshold", dest="metric_threshold", type=float, default=2.5, help="Distance in groundplane pixel where detection are considered positive when computing metrics")

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    from os import listdir
    from os.path import isfile, join

    model_dir = ROOT_PATH + "/detection_results/" + args.name + "/"
   
    epochs = [f[:-6].split("_")[3] for f in os.listdir(model_dir) if isfile(join(model_dir, f)) if f.endswith(args.detection_type+"_gt.txt")]

    eng = matlab.engine.start_matlab()
    eng.cd(ROOT_PATH+'/misc/matlabeval/motchallenge-devkit')

    result = list()
    for epoch in epochs:

        res_fpath = ROOT_PATH + "/detection_results/" + args.name + "/" + args.name + '_epoch_' + str(epoch) + "_" + args.detection_type + ".txt"
        gt_fpath = ROOT_PATH + "/detection_results/" + args.name + "/" + args.name + '_epoch_' + str(epoch) + "_" + args.detection_type + "_gt.txt"

    # recall, precision, moda, modp = matlab_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'matlab eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
    # recall, precision, moda, modp = python_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')

        recall, precision, moda, modp = evaluate(res_fpath, gt_fpath, 'Wildtrack', args.metric_threshold, eng)

        print('eval ' + args.name + ' epoch ' +str(epoch) + ': MODA ' + str(moda) +' MODP ' + str(modp) + ' prec '+ str(precision) + ' rcll '+ str(recall))

        result.append((epoch, recall, precision, moda, modp))

    print('Malab CLEAR MOD results for ' + args.name)
    print('epoch, recall, precision, moda, modp')
    for res in result:
        print(result)