#!/usr/bin/python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
import sys
sys.path.append('./PanNukeChallenge/src/metrics')
from utils import remap_label, binarize
# from utils import get_fast_pq, remap_label, binarize
from scanpy import read_h5ad
import argparse
from pathlib import Path

def get_fast_pq(true, pred, match_iou=0.5):
    # Has been modified
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) -1, 
                             len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise iou
    # for true_id in true_id_list[1:]: # 0-th is background
    for i in range(1,len(true_id_list)): # 0-th is background
        true_id = i
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1 # index is instance id - 1
        paired_pred += 1 # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence 
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum   
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair 
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def main(args):
    """
    Args:
    Root path to the ground-truth
    Root path to the predictions
    Path where results will be saved

    Output:
    Terminal output of bPQ and mPQ results for each class and across tissues
    Saved CSV files for bPQ and mPQ results for each class and across tissues
    """

    root = args.dir
    save_path = args.save_path
    class_path = args.class_path
    
    adata = read_h5ad(class_path)
    classes = adata.obs['predicted.id'].cat.categories.tolist()
    num_classes = len(classes)

    if not save_path:
        save_path = Path(root)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    true_path = os.path.join(root,'truths.npy')  # path to the GT for a specific split
    pred_path = os.path.join(root, 'preds.npy')  # path to the predictions for a specific split
    types_path = os.path.join(root,'types.npy') # path to the nuclei types

    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)
    types = np.load(types_path)

    mPQ_all = []
    bPQ_all = []

    # loop over the images
    for i in range(true.shape[0]):
        pq = []
        pred_bin = binarize(pred[i,:,:,:num_classes])
        true_bin = binarize(true[i,:,:,:num_classes])

        if len(np.unique(true_bin)) == 1:
            pq_bin = np.nan # if ground truth is empty for that class, skip from calculation
        else:
            [_, _, pq_bin], _ = get_fast_pq(true_bin, pred_bin) # compute PQ

        # loop over the classes
        for j in range(num_classes):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('int32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('int32')
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                pq_tmp = np.nan # if ground truth is empty for that class, skip from calculation
            else:
                [_, _, pq_tmp] , _ = get_fast_pq(true_tmp, pred_tmp) # compute PQ

            pq.append(pq_tmp)

        mPQ_all.append(pq)
        bPQ_all.append([pq_bin])

    # using np.nanmean skips values with nan from the mean calculation
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    PQ_dict = {cls : np.nanmean([pq[i] for pq in mPQ_all]) for i,cls in enumerate(classes)}

    # Print for each class
    print('Printing calculated metrics on a single split')
    print('-'*40)
    print(PQ_dict)
    print('-' * 40)

    df = pd.DataFrame.from_dict(PQ_dict, orient='index', columns=['PQ'])
    df.to_csv(save_path / 'class_stats.csv')

#####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--dir", type=str, default=None, help="Path to output directory containing masks.npy, truths.npy, types.npy"
    )
    parser.add_argument(
            "--save_path", type=str, default=None, help="Path to save output"
    )
    parser.add_argument(
            "--class_path", type=str, default=None, help="Path to anndata object containing the classes"
    )
    args = parser.parse_args()
    print(args)
    main(args)
