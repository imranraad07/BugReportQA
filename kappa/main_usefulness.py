import kappa.agreements as bd
import kappa.metrics as kd

import pandas as pd

usefulness_annotations_1 = {
    "a": [None, 1, 2, None, 2],
    "b": [2, 2, 2, 2, 2],
    "c": [2, 2, None, 2, 1],
    "d": [2, 2, 2, None, 2],
    "e": [None, 2, 2, 2, 2],
    "f": [None, 2, 1, 2, 2],
    "g": [1, 2, None, None, 2],
    "h": [None, 2, 2, None, 2],
}

usefulness_annotations_2 = {
    "a": [2, 2, 2, 2, 2],
    "b": [2, 2, 2, 2, None],
    "c": [2, 2, None, 2, 2],
    "d": [2, 2, None, 2, 2],
    "e": [2, 2, 2, 2, 2]
}

df = pd.DataFrame(usefulness_annotations_2)

kripp = kd.Krippendorff(df)
bidis = bd.BiDisagreements(df)

if __name__ == "__main__":
    print("==============================")
    bidis.agreements_summary()
    print("==============================")

    alpha = kripp.alpha(data_type="nominal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff nominal", alpha)
