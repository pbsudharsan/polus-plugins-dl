'''

Code sourced  code  from Cellpose repo  https://github.com/MouseLand/cellpose/tree/master/cellpose

'''
import numpy as np

import dynamics


def flow_error(maski, dP_net):
    """ Error in flows from predicted masks vs flows predicted by network run on image
    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted
    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
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
    dP_masks, _ = dynamics.masks_to_flows(maski)
    iun = np.unique(maski)[1:]
    flow_errors = np.zeros((len(iun),))
    for i, iu in enumerate(iun):
        ii = maski == iu
        if dP_masks.shape[0] == 2:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii] / 5.) ** 2
                               + (dP_masks[1][ii] - dP_net[1][ii] / 5.) ** 2).mean()
        else:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii] / 5.) ** 2 * 0.5
                               + (dP_masks[1][ii] - dP_net[1][ii] / 5.) ** 2
                               + (dP_masks[2][ii] - dP_net[2][ii] / 5.) ** 2).mean()
    return flow_errors, dP_masks
