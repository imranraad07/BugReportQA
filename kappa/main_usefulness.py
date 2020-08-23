import kappa.agreements as bd
import kappa.metrics as kd
import pandas as pd

# usefulness_annotations = {
#     "a": [3, None, 3, 3,    1, 3,    3,    3,    None, None, 3, 1,    None],
#     "b": [3,    3, 3, 3,    1, 3,    3,    3,    3,    3,    3, 3,    3],
#     "c": [3,    3, 3, 3, None, None, None, None, 3,    3,    3, None, 3],
#     "d": [3, None, 3, 3,    3, 3,   3,     3,    3,    3,    3, None, None],
#     "e": [3,    3, 3, None, 3, 3,   3,     3,    3,    3,    3, 3,    3],
# }

# usefulness_annotations = {
#     "a": [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     "b": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
#     "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
#     "d": [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
#     "e": [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
# }

usefulness_annotations_1 = {
    "a": [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    "b": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "d": [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    "e": [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
}

usefulness_annotations_2 = {
    "a": [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "b": [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    "c": [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    "d": [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    "e": [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
}


# usefulness_annotations = {
#     "a": [3, 3, None, 1, 3, None, None, None, 1, None, None, 4, 2, 3, 3, None, None, 3, None, None],
#     "b": [3, 3, None, 0, 4, None, 4, 3, 4, 3, None, 4, 4, 4, 3, None, None, 3, None, None],
#     "c": [3, 2, None, 2, 2, None, 2, 2, None, 3, None, 4, 3, None, None, None, None, 4, None, None],
#     "d": [None, 4, None, 3, 2, None, 4, 4, None, None, None, 3, 3, 3, 3, None, None, 3, None, None],
#     "e": [None, 2, None, None, None, None, 3, 3, 2, 2, None, 2, None, 3, 4, None, None, 4, None, None],
# }

df = pd.DataFrame(usefulness_annotations_1)

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
