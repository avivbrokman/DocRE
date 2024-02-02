#%% libraries
from collections import defaultdict
from utils import defaultdict2dict

#%% 
class Calculator:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(lambda: 0))
        
    def update(self, counts):
        for class_, counts_dict in counts.items():
            for count_category, count in counts_dict.items():
                self.counts[class_][count_category] += count
    
    def compute_precision(self, TP, FP):
        if TP + FP == 0:
            return 0
        else:
            return TP / (TP + FP)

    def compute_recall(self, TP, FN):
        if TP + FN == 0:
            return 0
        else:
            return TP / (TP + FN)

    def compute_F1(self, TP, FP, FN):
        if not self.compute_precision(TP, FP) or not self.compute_precision(TP, FN):
            return 0
        else:
            return 2 * TP / (2 * TP + FP + FN)

    def compute(self):
        results = defaultdict(lambda: dict())
        for class_, counts_dict in self.counts.items():
            precision = self.compute_precision(counts_dict['TP'], counts_dict['FP'])
            results[class_]['precision'] = precision
            results[class_]['recall'] = self.compute_recall(counts_dict['TP'], counts_dict['FN'])
            results[class_]['F1'] = self.compute_F1(counts_dict['TP'], counts_dict['FP'], counts_dict['FN'])
        
        return defaultdict2dict(results)
    
    def reset(self):
        self.__init__()

