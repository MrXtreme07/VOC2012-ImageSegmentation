import torch

def compute_iou(pred, mask, num_classes=21, ignore_index=255):
    ious = []

    valid = mask != ignore_index
    pred, mask = pred[valid], mask[valid]

    pred, mask = pred.view(-1), mask.view(-1)

    for cls in range(num_classes):
        pred_inds, target_inds = pred == cls, mask == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious