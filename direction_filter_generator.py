from app.extractor.invoice_config import *
import app.extractor.example
import pandas as pd
import numpy as np


class Box(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.box = (x1, y1, x2, y2)
        self.width = abs(x1 - x2)
        self.height = abs(y1 - y2)
        self.center = self.x, self.y = self._get_center()
        self.cache = {}

    def __hash__(self):
        return hash("卍".join(str(self.box)))

    def _get_center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    ####################################################################################################################
    #                                                   |          |
    #                       1                           |     2    |                        3
    #                                                   |          |
    # -----------------------------------------------------------------------------------------------------------------#
    #                       4                           |  self(5) |                        6
    # -----------------------------------------------------------------------------------------------------------------#
    #                                                   |          |
    #                       7                           |     8    |                        9
    #                                                   |          |
    ####################################################################################################################
    def get_direction(self, box_):
        x1_, y1_, x2_, y2_ = box_.box
        if x1_ > self.x2 or self.x2 - x1_ < (x2_ - x1_) / 2:
            x_score = 3
        elif x2_ < self.x1 or x2_ - self.x1 < (x2_ - x1_) / 2:
            x_score = 1
        else:
            x_score = 2
        if y1_ > self.y2 or self.y2 - y1_ < (y2_ - y1_) / 2:
            y_score = 2
        elif y2_ < self.y1 or y2_ - self.y1 < (y2_ - y1_) / 2:
            y_score = 0
        else:
            y_score = 1
        return x_score + y_score * 3

    def release_cache(self):
        self.cache = {}


def get_direction_filter(sample=None):
    if sample:
        examples = sample
    else:
        examples = app.extractor.example.examples

    invoice_direction_filter = dict()
    for invoice_type in invoice_pattern:
        label = examples[invoice_type]["shapes"]
        anchors = {}
        direction_filter = {}
        for boxes in label:
            [[x1, y1], [x2, y2]] = boxes["points"]
            anchors[boxes["label"]] = Box(x1, y1, x2, y2)
        for label, box in anchors.items():
            for label_, box_ in anchors.items():
                direction_filter[(label,label_)] = box.get_direction(box_)
        invoice_direction_filter[invoice_type] = direction_filter

    return invoice_direction_filter
