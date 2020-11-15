import torch

def __trim_roi_tensor(
    tensor:torch.Tensor,
    max_num_rois:int)->torch.Tensor:
    """
    各バッチで含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
    もしもmax_num_roisよりも多い場合には切り捨て、max_num_roisよりも少ない場合には0ベクトルで埋める。

    入力Tensorのサイズ
    (num_rois,x)

    出力Tensorのサイズ
    (max_num_rois,x)
    """
    num_rois=tensor.size(0)

    #RoIの数が制限よりも多い場合はTruncateする。
    if num_rois>max_num_rois:
        ret=tensor[:max_num_rois]
    #RoIの数が制限よりも少ない場合は0ベクトルで埋める。
    elif num_rois<max_num_rois:
        zeros=torch.zeros(max_num_rois-num_rois,tensor.size(-1))
        ret=torch.cat([tensor,zeros],dim=0)
    else:
        ret=tensor

    return ret

def load_roi_boxes_from_file(
    roi_boxes_filepath:str,
    max_num_rois:int)->torch.Tensor:
    """
    ファイルからRoIの座標情報を読み込む。

    出力Tensorのサイズ
    (max_num_rois,4)
    """
    roi_boxes=torch.load(roi_boxes_filepath)
    roi_boxes=__trim_roi_tensor(roi_boxes,max_num_rois)
    return roi_boxes

def load_roi_features_from_file(
    roi_features_filepath:str,
    max_num_rois:int)->torch.Tensor:
    """
    ファイルからRoIの特徴量を読み込む。

    出力Tensorのサイズ
    (max_num_rois,x)    デフォルトではx=1024
    """
    roi_features=torch.load(roi_features_filepath)
    roi_features=__trim_roi_tensor(roi_features,max_num_rois)
    return roi_features

def load_roi_labels_from_file(
    roi_labels_filepath:str,
    max_num_rois:int)->torch.Tensor:
    """
    ファイルからRoIのラベルを読み込む。

    出力Tensorのサイズ
    (max_num_rois)
    """
    roi_labels=torch.load(roi_labels_filepath)
    roi_labels=torch.unsqueeze(roi_labels,1)  #(num_rois,1)
    roi_labels=__trim_roi_tensor(roi_labels,max_num_rois)
    roi_labels=torch.squeeze(roi_labels)  #(max_num_rois)
    return roi_labels
