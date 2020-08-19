import sys

import kappa.agreements as bd
import kappa.metrics as kd
import kappa.metrics as metrics

import pandas as pd

specificity_annotations_1 = {
    "a": [None, 3, 3, None, 3],
    "b": [3, 3, 1, 1, 1],
}

specificity_annotations_2 = {
    "a": [3, 3, 3, 3, 3],
    "b": [3, 3, 3, 1, 3],
    "c": [3, 3, None, 3, 3],
    "d": [3, 3, None, 1, 3],
    "e": [3, 3, 3, 3, 3],
}

df = pd.DataFrame(specificity_annotations_2)

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
