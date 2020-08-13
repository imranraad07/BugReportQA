import argparse
import csv
import sys

import altair as alt
import pandas as pd

alt.renderers.enable('altair_viewer')


def main(args):
    df = pd.DataFrame({'Org': ['Novelty', 'Novelty'],
                       'response': ['Yes', 'No'],
                       'percent': [94.42, 5.58],
                       'value': [3, 48],
                       'position': [2.5, 8.2]})

    color_scale = alt.Scale(
        domain=[
            "No",
            "Yes"
        ],
        range=["#1E1E1E", "#006312"]
        # range=["#c30d24", "#1770ab"]
    )

    chart = alt.Chart(df).mark_bar(size=12).encode(
        x=alt.X('percent', title='Percentage'),
        y=alt.Y('Org', title=''),
        order=alt.Order(
            'percent',
            sort='ascending'
        ),
        color=alt.Color('response:N',
                        legend=alt.Legend(orient='bottom', title='', padding=-12), scale=color_scale)
    ).properties(
        height=24
    )

    text = alt.Chart(df).mark_text(color='white', align='center', baseline='middle').encode(
        y=alt.Y('Org', title=''),
        x=alt.X('position', title=''),
        text='value'
    )

    combined = chart + text
    fontSize = 8
    combined = combined.configure_axis(
        labelFontSize=fontSize,
        titleFontSize=fontSize,
    ).configure_text(
        fontSize=fontSize
    )

    combined.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
