class AverageMetric:
    """
    Class to serve as an average calculator for metrics such as loss, accuracy, etc
    """
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
        
    def update(self, value, n=1):
        self.sum += value*n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg