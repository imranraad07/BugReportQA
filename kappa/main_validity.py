import sys

import kappa.agreements as bd
import kappa.metrics as kd
import kappa.metrics as metrics

import pandas as pd

validity_annotations = {
    "a": [0, 0, 0, 0, 0],
    "b": [0, 1, 1, 0, 1],
    "c": [1, 1, 1, 1, 1],
    "d": [1, 0, 1, 0, 0],
    "e": [1, 1, 0, 1, 1],
    "f": [1, 1, 1, 0, 1],
    "g": [0, 0, 0, 0, 1],
    "h": [0, 1, 1, 1, 1],
    "i": [0, 1, 1, 1, 1],
    "j": [0, 1, 1, 0, 1],
    "k": [0, 0, 0, 0, 1],
    "l": [1, 1, 1, 0, 0],
}
df = pd.DataFrame(validity_annotations)

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
