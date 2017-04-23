from collections import OrderedDict

"""
MetricsModule that implements batch and epoch metrics such as Accuracy
"""

class MetricsModule():
    
    def __init__(self, metrics_classes):
        self._metrics = [ metric_class() for metric_class in metrics_classes]
        
    def update(self, predictions, target):
        for metric in self._metrics:
            metric.update(predictions, target)
        
    def get_logs(self, prefix = ''):
        logs = OrderedDict()
        for metric in self._metrics:
            logs.update(metric.get_logs(prefix))
        return logs

class Metric():
    
    def update(self, predictions, target):
        raise NotImplementedError()
    
    def get_logs(self, prefix):
        raise NotImplementedError()

class AccuracyMetric(Metric):
    
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def get_prediction_classes_ids(self, predictions):
        # returns the predictions in id format
        values, predictions_ids = predictions.max(1)
        return predictions_ids
        
    def update(self, predictions, target):
        prediction_classes_ids = self.get_prediction_classes_ids(predictions).cpu()
        target_classes_ids = target.cpu()
        self.correct_count += target_classes_ids.eq(prediction_classes_ids).sum()
        self.total_count += predictions.size(0)
        self.accuracy = 100.0 * self.correct_count / self.total_count
        
    def get_logs(self, prefix):
        return { prefix + 'acc' : self.accuracy }