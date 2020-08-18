import sys

import kappa.agreements as bd
import kappa.metrics as kd
import kappa.metrics as metrics

import pandas as pd

new_info_annotations = {
    "a": [None, 2, 2,    2,    2, None, None, None, 2, None, None, 2, 2,    2, 2,    2,    2, 2,    2],
    "b": [1,    2, None, 2,    2, None, 2,    2,    1, 2,    2,    2, 2,    2, None, 2,    2, 2,    None],
    "c": [2,    2, 2,    None, 2, None, 2,    1, None, 2,    None, 2, 2, None, 2,    None, 2, None, None],
    "d": [None, 2, None, 2, None, None, 2,    2, None, None, 2,    2, 2,    2, None, None, 2, None, None],
    "e": [2,    2, None, 1,    2, 2,    2,    2,    2, 2,    None, 2, None, 1, None, None, 2, None, None],
}

df = pd.DataFrame(new_info_annotations)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(df)

kripp = kd.Krippendorff(df)
bidis = bd.BiDisagreements(df)
mets = metrics.Metrics(df)

if __name__ == "__main__":
    print("==============================")
    bidis.agreements_summary()
    print("==============================")

    alpha = kripp.alpha(data_type="ordinal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff ordinal", alpha)
