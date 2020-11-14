import os
import torch
from typing import List,Tuple

def __load_roi_boxes_and_features_and_classes(
    roi_boxes_filepath:str,
    roi_features_filepath:str,
    roi_labels_filepath:str,
    device:torch.device)->Tuple[torch.Tensor,torch.Tensor]:
    """
    RoIの矩形領域の座標データおよび特徴量をファイルから読み込む。
    RoIのデータが存在しない場合にはNoneを返す。
    """
    roi_boxes=None  #(num_rois,4)
    roi_features=None   #(num_rois,roi_features_dim)
    roi_labels=None    #(num_rois)
    #RoIの特徴量が存在する場合 (矩形領域の座標データやラベル情報も存在するはず)
    if os.path.exists(roi_features_filepath):
        roi_boxes=torch.load(roi_boxes_filepath,map_location=device).to(device)
        roi_features=torch.load(roi_features_filepath,map_location=device).to(device)
        roi_labels=torch.load(roi_labels_filepath,map_location=device).to(device)
    
    return roi_boxes,roi_features,roi_labels

def __trim_roi_tensor(
    tensor:torch.Tensor,
    max_num_rois:int,
    device:torch.device)->torch.Tensor:
    """
    各バッチで含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
    もしもmax_num_roisよりも多い場合には切り捨て、max_num_roisよりも少ない場合には0ベクトルで埋める。

    入力Tensorのサイズ
    (num_rois,x)

    出力Tensorのサイズ
    (max_num_rois,x)
    """
    #RoIが存在しない場合
    if tensor is None:
        ret=torch.zeros(max_num_rois,tensor.size(-1)).to(device)
        return ret

    num_rois=tensor.size(0)

    #RoIの数が制限よりも多い場合はTruncateする。
    if num_rois>max_num_rois:
        ret=tensor[:max_num_rois]
    #RoIの数が制限よりも少ない場合は0ベクトルで埋める。
    elif num_rois<max_num_rois:
        zeros=torch.zeros(max_num_rois-num_rois,tensor.size(-1)).to(device)
        ret=torch.cat([tensor,zeros],dim=0)
    else:
        ret=tensor

    return ret

def load_roi_info_from_files(
    roi_boxes_filepaths:List[str],
    roi_features_filepaths:List[str],
    roi_labels_filepaths:List[str],
    max_num_rois:int,
    roi_features_dim:int,
    device:torch.device):
    """
    ImageBERTのモデルに入力するためのRoI特徴量を作成する。
    読み込んだデータはDict形式で返される。

    roi_boxes: (N,max_num_rois,4)
    roi_features: (N,max_num_rois,roi_features_dim)
    roi_labels: (N,max_num_rois)
    """
    batch_size=len(roi_boxes_filepaths)

    ret_roi_boxes=torch.empty(batch_size,max_num_rois,4).to(device)
    ret_roi_features=torch.empty(batch_size,max_num_rois,roi_features_dim).to(device)
    ret_roi_labels=torch.empty(batch_size,max_num_rois).to(device)

    for i in range(batch_size):
        #RoIの座標情報と特徴量をファイルから読み込む。
        roi_boxes,roi_features,roi_labels=__load_roi_boxes_and_features_and_classes(
            roi_boxes_filepaths[i],
            roi_features_filepaths[i],
            roi_labels_filepaths[i],
            device
        )
        #選択肢ごとに含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
        roi_boxes=__trim_roi_tensor(roi_boxes,max_num_rois,device)
        roi_features=__trim_roi_tensor(roi_features,max_num_rois,device)

        roi_labels=torch.unsqueeze(roi_labels,1)  #(num_rois,1)
        roi_labels=__trim_roi_tensor(roi_labels,max_num_rois,device)
        roi_labels=torch.squeeze(roi_labels)  #(num_rois)

        ret_roi_boxes[i]=roi_boxes
        ret_roi_features[i]=roi_features
        ret_roi_labels[i]=roi_labels

    ret={
        "roi_boxes":ret_roi_boxes,
        "roi_features":ret_roi_features,
        "roi_labels":ret_roi_labels
    }
    return ret
