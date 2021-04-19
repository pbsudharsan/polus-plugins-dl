import umetrics
from bfio import BioReader
import numpy as np
br1= BioReader('/home/sudharsan/Downloads/x48_y31_p1_c1.ome.tif')
br2= BioReader('/home/sudharsan/Desktop/work/ome/pretrained/x48_y31_p1_c1.ome.tif')
y_true = br1.read()
y_pred =  br2.read()
y_true = y_true.squeeze()
y_pred = y_pred.squeeze()

from sklearn.metrics import confusion_matrix

def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     for i in range(len(y_true)):
          if y_true[i]>0:
              y_true[i] =1
          else:
               y_true[i]=0

     print(np.amax(y_true))
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     tn, fp, fn, tp = current.ravel()
     print(tn, fp, fn, tp)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)

print(compute_iou(y_pred,y_true))


