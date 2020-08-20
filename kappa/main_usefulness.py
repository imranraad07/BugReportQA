import kappa.agreements as bd
import kappa.metrics as kd

import pandas as pd

usefulness_annotations = {
    "a": [3, None, 3, 3,    1, 3,    3,    3,    None, None, 3, 1,    None],
    "b": [3,    3, 3, 3,    1, 3,    3,    3,    3,    3,    3, 3,    3],
    "c": [3,    3, 3, 3, None, None, None, None, 3,    3,    3, None, 3],
    "d": [3, None, 3, 3,    3, 3,   3,     3,    3,    3,    3, None, None],
    "e": [3,    3, 3, None, 3, 3,   3,     3,    3,    3,    3, 3,    3],
}

df = pd.DataFrame(usefulness_annotations)

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
