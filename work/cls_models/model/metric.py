import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score as accuracy_metric
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import binarize


class AvgMetrics(nn.Layer):
    def __init__(self):
        super().__init__()
        self.avg_meters = {}

    def reset(self):
        self.avg_meters = {}

    @property
    def avg(self):
        if self.avg_meters:
            for metric_key in self.avg_meters:
                return self.avg_meters[metric_key].avg

    @property
    def avg_info(self):
        return ", ".join([self.avg_meters[key].avg_info for key in self.avg_meters])

class MultiLabelMetric(AvgMetrics):
    def __init__(self, bi_threshold=0.5):
        super().__init__()
        self.bi_threshold = bi_threshold

    def _multi_hot_encode(self, output):
        logits = F.sigmoid(output).numpy()
        return binarize(logits, threshold=self.bi_threshold)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name='', fmt='f', postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def avg_info(self):
        if isinstance(self.avg, paddle.Tensor):
            self.avg = self.avg.numpy()[0]
        return "{}: {:.5f}".format(self.name, self.avg)

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}'.format(
            self=self)

    @property
    def total_minute(self):
        return '{self.name} {s:{self.fmt}}{self.postfix} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        return '{self.name}: {self.avg:{self.fmt}}{self.postfix}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}{self.postfix}'.format(
            self=self)



class AccuracyScore(MultiLabelMetric):
    """
    Hard metric for multilabel classification
    Args:
        base: ["sample", "label"], default="sample"
            if "sample", return metric score based sample,
            if "label", return metric score based label.
    Returns:
        accuracy:
    """

    def __init__(self, base="label"):
        super().__init__()
        assert base in ["sample", "label"
                        ], 'must be one of ["sample", "label"]'
        self.base = base
        self.reset()

    def reset(self):
        self.avg_meters = {"AccuracyScore": AverageMeter("AccuracyScore")}

    def forward(self, output, target):
        preds = super()._multi_hot_encode(output)
        metric_dict = dict()
        if self.base == "sample":
            accuracy = accuracy_metric(target, preds)
        elif self.base == "label":
            mcm = multilabel_confusion_matrix(target, preds)
            tns = mcm[:, 0, 0]
            fns = mcm[:, 1, 0]
            tps = mcm[:, 1, 1]
            fps = mcm[:, 0, 1]
            accuracy = (sum(tps) + sum(tns)) / (
                sum(tps) + sum(tns) + sum(fns) + sum(fps))
        metric_dict["AccuracyScore"] = paddle.to_tensor(accuracy)
        self.avg_meters["AccuracyScore"].update(
            metric_dict["AccuracyScore"].numpy()[0], output.shape[0])
        return metric_dict

class HammingDistance(MultiLabelMetric):
    """
    Soft metric based label for multilabel classification
    Returns:
        The smaller the return value is, the better model is.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.avg_meters = {"HammingDistance": AverageMeter("HammingDistance")}

    def forward(self, output, target):
        preds = super()._multi_hot_encode(output)
        metric_dict = dict()
        metric_dict["HammingDistance"] = paddle.to_tensor(
            hamming_loss(target, preds))
        self.avg_meters["HammingDistance"].update(
            metric_dict["HammingDistance"].numpy()[0], output.shape[0])
        return metric_dict
