import sys

import kappa.agreements as bd
import kappa.metrics as kd

import pandas as pd

usefulness_annotations = {
    "a": [None, 3, 3,    1,    3,    None, None, None, 1,    None, None, 4, 2,    3,    3,    1,    2,    3, 4,     3],
    "b": [3,    3, None, 0,    4,    None, 4,    3,    4,    3,    4,    4, 4,    4,    3,    None, 3,    3, 3,    None],
    "c": [3,    2, 2,    None, 2,    None, 2,    2,    None, 3,    None, 4, 3,    None, None, 3,    None, 4, None, None],
    "d": [None, 4, None, 3,    None, None, 4,    4,    None, None, 3,    3, 3,    3,    3,    None, None, 3, None, None],
    "e": [3,    2, None, 2,    2,    3,    3,    3,    2,    2,    None, 2, None, 3,    4,    None, None, 4, None, None],
}

df = pd.DataFrame(usefulness_annotations)

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
