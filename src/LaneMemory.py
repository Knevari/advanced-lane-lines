class Line:
    def __init__(self):
        self.detected = False
        self.best_fit = []


class LaneMemory:
    def __init__(self):
        self.left = Line()
        self.right = Line()
