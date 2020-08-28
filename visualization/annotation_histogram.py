import argparse
import csv
import sys

import altair as alt
import altair_saver as altsave
import pandas as pd
print(alt.__version__)

alt.renderers.enable('altair_viewer')


# 1 -> 31
# 2 -> 54
# 3 -> 29
# 4 -> 43
# 5 -> 26
# 6 -> 17
# 7 -> 11
# 8 -> 5

def main(args):
    source = pd.DataFrame({
        'num_valid': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Percentage of instances': [14.35, 25, 13.42, 19.90, 12.03, 7.87, 5.09, 2.34, 0],
    })

    fontSize = 16
    chart = alt.Chart(source).mark_bar(size=30).encode(
        x=alt.X('num_valid', title='Number of valid follow-ups per bug report',
                sort=["1", "2", "3", "4", "5",
                      "6", "7", "8", "9", "10"],
                axis=alt.Axis(labels=True,
                              ticks=True,
                              values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                ),
        y=alt.Y('Percentage of instances'),
        color=alt.value('gray')
    ).properties(
        width=400
    ).configure_axis(
        grid=False,
        labelFontSize=fontSize,
        titleFontSize=fontSize,
    ).configure_text(
        fontSize=fontSize
    ).configure_title(fontSize=fontSize)
    # chart.show()
    altsave.save(chart, fp='viz_annotaiton.pdf')


# import bokeh.palettes as bp

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--input_file", type=str, default='annot_num_valid.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
