import numpy as np
from abc import ABC, abstractmethod


def get_evaluation_metrics_function(metric):
    '''
    Get the evaluation metric from the name
    '''

    if metric == 'mse':
        return mean_square_error
    elif metric == 'mae':
        return mean_absolute_error
    elif metric == 'accuracy':
        return accuracy
    elif metric == 'success':
        return success
    elif metric == 'dice':
        return dice
    elif metric == 'multiclass-dice':
        return multiclass_dice
    elif metric == 'iou':
        return intersection_over_union
    elif metric == 'multiclass-auc':
        return multiclass_auc
    else:
        raise ValueError('Unsuported metric {}'.format(metric))



class Metric(ABC):
    '''
    Template for the evaluation metrics class
    '''

    @abstractmethod
    def get_name(self):
        '''
        Returns the name of the metric
        '''
        return None

    def preprocess_input(self, volume, mask=None):
        '''
        Apply a transformation on the input
        '''
        # filter a specific region if necessary
        if mask is None:
            volume = volume.flatten() > 0
        else:
            volume = volume[mask] > 0
        return volume

    @abstractmethod
    def compute(self, gt, pred):
        '''
        Compute the metric
        '''
        return None

    def evaluate(self, gt, pred, mask=None):
        '''
        Evaluates the metric
        '''
        # preprocess the inputs
        gt = self.preprocess_input(gt, mask)
        pred = self.preprocess_input(pred, mask)
        # compute the metric
        result = self.compute(gt, pred)
        return result



class MeanSquareError(Metric):
    '''
    MeanSquareError = frac{1}{N} sum_i^N (gt - pred).^2
    '''

    def get_name(self):
        return 'MeanSquareError'

    def compute(self, gt, pred):
        return mean_square_error(gt, pred)


def mean_square_error(gt, pred):
    '''
    MeanSquareError = frac{1}{N} sum_i^N (gt - pred).^2
    '''

    gt = gt.flatten()
    pred = pred.flatten()
    
    return np.mean(np.power(gt - pred, 2))



class MeanAbsoluteError(Metric):
    '''
    MeanAbsoluteError = frac{1}{N} sum_i^N |(gt - pred)| 
    '''

    def get_name(self):
        return 'MeanAbsoluteError'

    def compute(self, gt, pred):
        return mean_absolute_error(gt, pred)


def mean_absolute_error(gt, pred):
    '''
    MeanSquareError = frac{1}{N} sum_i^N |(gt - pred)| 
    '''

    gt = gt.flatten()
    pred = pred.flatten()
    
    return np.mean(np.abs(gt - pred, 2))



class Accuracy(Metric):
    '''
    Accuracy = (TP + TN) / (TP + FN + FP + TP)
    '''

    def get_name(self):
        return 'Accuracy'

    def compute(self, gt, pred):
        return accuracy(gt, pred)


def accuracy(gt, pred):
    '''
    Accuracy = = (TP + TN) / (TP + FN + FP + TP)
    '''

    agreement = np.count_nonzero(gt == pred)
    n = gt.size
    
    return agreement / n


def success(gt, pred):
    '''
    Indicate if the values agree
    '''

    return (gt == pred)



class Dice(Metric):
    '''
    Dice = 2 * the Area of Overlap divided by the total number of pixels in both images 
    '''

    def get_name(self):
        return 'dice'

    def compute(self, gt, pred):
        return dice(gt, pred)


def dice(gt, pred):
    '''
    Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
    '''

    gt = (gt > 0).flatten()

    if np.any(gt):

        pred = (pred > 0).flatten()

        numerator = np.sum(np.multiply(pred, gt))

        if numerator == 0.0:
            return 0.0
        else:
            denominator = np.sum(pred) + np.sum(gt)
            return (2.0 * numerator) / denominator

    else:

        return np.nan



class MulticlassDice(Metric):
    '''
    Multiclass Dice = Compute Dice for each class 
    '''

    def get_name(self):
        return 'multiclass_dice'

    def compute(self, gt, pred, num_classes):
        return multiclass_dice(gt, pred, num_classes)


def multiclass_dice(gt, pred, num_classes):
    '''
    Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
    '''

    # initialize an array with the results per class
    results = np.zeros(num_classes, dtype=np.float)

    # for each of the classes
    for i in range(1, num_classes):

        # binarize the gt and the pred
        binarized_gt = (gt == i)
        binarized_pred = (pred == i)
        # compute the dice for these two masks
        results[i] = dice(binarized_gt, binarized_pred)

    return results



from sklearn.metrics import roc_auc_score

class MulticlassAUC(Metric):
    '''
    Compute the AUC for each of the given classes
    '''

    def get_name(self):
        return 'multiclass_dice'

    def compute(self, gt, pred, num_classes):
        return multiclass_auc(gt, pred, num_classes)


def multiclass_auc(gt, pred, num_classes):
    '''
    Computes area under the ROC curve for each class
    '''

    # initialize an array with the results per class
    results = np.zeros(num_classes, dtype=np.float)
    # for each of the classes
    for i in range(1, num_classes):

        # get the score corresponding to this class
        current_scores = pred[i,:,:]
        # binarize the gt
        binarized_gt = (gt == i)
        # compute the AUC of the ROC curve only if more than one class present
        if (np.unique(binarized_gt.flatten())).size > 1:
            results[i] = roc_auc_score(binarized_gt.flatten(), current_scores.flatten())
        else:
            results[i] = np.nan

    return results




class IntersectionOverUnion(Metric):
    '''
    IoU = is the Area of Overlap (gt, pred) divided by the area of union (gt, pred)
    '''

    def get_name(self):
        return 'iou'

    def compute(self, gt, pred):
        return intersection_over_union(gt, pred)


def intersection_over_union(gt, pred, ignore_class=None ):
    '''
    IoU = is the Area of Overlap (gt, pred) divided by the area of union (gt, pred)
    '''
    total_iou = 0
    gt_size = gt.size
    pred_size = pred.size
    classes = np.unique(gt) | np.unique(pred)
    for c in classes:  
        # create binary images with c as mask
        gt_inds = gt == c
        pred_inds = pred == c
        intersection = (pred_inds[gt_inds]).sum()
        # Union consists of all of the pixels classified as c from both images, minus the overlap
        union =  gt_inds.sum() + pred_inds.sum() - intersection
        total_iou += intersection / union

    return total_iou/classes.size

    