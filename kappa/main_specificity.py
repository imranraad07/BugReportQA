import sys

import kappa.agreements as bd
import kappa.metrics as kd

import pandas as pd

# specificity_annotations = {
#     "a": [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     "b": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
#     "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#     "d": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#     "e": [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
# }

specificity_annotations = {
    "a": [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    "b": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    "d": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "e": [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
}


specificity_annotations_1 = {
    "a": [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "b": [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "c": [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "d": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "e": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
}

specificity_annotations_2 = {
    "a": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "b": [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    "c": [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    "d": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "e": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
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

    alpha = kripp.alpha(data_type="interval")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff interval", alpha)

    alpha = kripp.alpha(data_type="ratio")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff ratio", alpha)

    alpha = kripp.alpha(data_type="ordinal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff ordinal", alpha)

    alpha = kripp.alpha(data_type="nominal")
    alpha = float("{:.3f}".format(alpha))
    print("Krippendorff nominal", alpha)
    mets = kd.Metrics(df).fleiss_kappa()
    print("Fleiss kappa", mets)
