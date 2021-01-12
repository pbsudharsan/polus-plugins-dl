import numpy as np
import  dynamics
from numba import jit


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y
    Args:
    x(array[int]): ND-array where 0=NO masks; 1,2... are mask labels
    y(array[int]): ND-array where 0=NO masks; 1,2... are mask labels
    Returns:
    overlap(array[int]): ND-array.matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    Args:
    masks_true(array[int]): ND-array.ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred(array[int]): ND-array.predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns:
    iou(array[float]): ND-array.matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def flow_error(maski, dP_net):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Args:
    maski(array[int]): ND-array.masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net(array[float]): ND-array.ND flows where dP_net.shape[1:] = maski.shape

    Returns:
    flow_errors(array[float]): float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks(array[float]): ND-array.ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    # flows predicted from estimated masks
    dP_masks,_ = dynamics.masks_to_flows(maski)
    iun = np.unique(maski)[1:]
    flow_errors=np.zeros((len(iun),))
    for i,iu in enumerate(iun):
        ii = maski==iu
        if dP_masks.shape[0]==2:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2).mean()
        else:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2 * 0.5
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2
                            + (dP_masks[2][ii] - dP_net[2][ii]/5.)**2).mean()
    return flow_errors, dP_masks
