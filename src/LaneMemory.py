class Line:
    def __init__(self):
        self.detected = False
        self.best_fit = []
        self.last_best_fit = []
        self.xpoly = None

    def setLastFit(self, best_fit):
        self.last_best_fit = self.best_fit
        self.best_fit = best_fit

    def setPoly(self, poly):
        self.xpoly = poly


class LaneMemory:
    def __init__(self):
        self.left = Line()
        self.right = Line()
