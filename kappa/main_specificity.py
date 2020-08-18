import sys

import kappa.agreements as bd
import kappa.metrics as kd
import kappa.metrics as metrics

import pandas as pd

specificity_annotations = {
    "a": [None, 3, 1,    1, 3,    None, None, None, 1, None, None, 3, 2,    2, 3,    3, 2,    3, 3,     3],
    "b": [3,    3, None, 1, 0,    None, 4,    4,    4, 3,    4,    4, 4,    4, 3, None, 3,    3, 3,    None],
    "c": [2,    0, 1, None, 2,    None, 2,    3, None, 3,    None, 4, 3, None, None, 3, None, 4, None, None],
    "d": [None, 0, None, 0, None, None, 4,    4, None, None, 1,    3, 1,    3, 1, None, None, 3, None, None],
    "e": [1,    0, None, 1, 0,    1,    3,    3,    2, 2,    None, 2, None, 1, 3, None, None, 4, None, None],
}

df = pd.DataFrame(specificity_annotations)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(df)

kripp = kd.Krippendorff(df)
bidis = bd.BiDisagreements(df)

if __name__ == "__main__":
    print("==============================")
    bidis.agreements_summary()
    print("==============================")

    alpha = kripp.alpha(data_type="ordinal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff ordinal", alpha)
    # alpha = kripp.alpha(data_type="nominal")
    # alpha = float("{:.3f}".format(alpha))
    # print("Krippendorff nominal", alpha)