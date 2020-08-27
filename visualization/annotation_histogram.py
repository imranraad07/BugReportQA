import argparse
import csv
import sys

import altair as alt
import pandas as pd

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
    # output_file("bars.html")
    # fruits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    # counts = [14.35, 25, 13.42, 19.90, 12.03, 7.87, 5.09, 2.34, 0, 0]
    # p = figure(x_range=fruits, plot_height=250,
    #            toolbar_location=None, tools="")
    # p.vbar(x=fruits, top=counts, width=0.9)
    # p.xgrid.grid_line_color = None
    # p.y_range.start = 0
    # show(p)

    # source = alt.pd.read_csv(args.input_file)
    source = pd.DataFrame({
        'num_valid': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Percentage of instances': [14.35, 25, 13.42, 19.90, 12.03, 7.87, 5.09, 2.34, 0],
    })

    alt.Chart(source).mark_bar(size=30).encode(
        x=alt.X('num_valid', title='Number of valid follow-ups per bug report (out of 10)',
                sort=["1", "2", "3", "4", "5",
                      "6", "7", "8", "9", "10"],
                axis=alt.Axis(labels=True,
                              ticks=True,
                              values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                ),
        y='Percentage of instances',
        color=alt.value('gray')
    ).properties(
        width=400
    ).configure_axis(
        grid=False
    ).show()


# import bokeh.palettes as bp

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--input_file", type=str, default='annot_num_valid.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
