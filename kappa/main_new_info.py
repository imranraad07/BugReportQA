import sys

import kappa.agreements as bd
import kappa.metrics as kd
import kappa.metrics as metrics
import numpy as np
import pandas as pd

new_info_annotations_1 = {
    "a": [None, 2, 2, 2, 2],
    "b": [2, None, 2, 2, 2],
    "c": [2, 2, None, 2, 2],
    "d": [2, 2, 2, None, 2],
    "e": [1, 2, 2, 2, None],
}

df = pd.DataFrame(new_info_annotations_1)

kripp = kd.Krippendorff(df)
bidis = bd.BiDisagreements(df)
mets = metrics.Metrics(df)

if __name__ == "__main__":
    print("==============================")
    bidis.agreements_summary()
    print("==============================")

    alpha = kripp.alpha(data_type="nominal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff nominal", alpha)
