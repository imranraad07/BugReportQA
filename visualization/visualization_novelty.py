import argparse
import csv
import sys

import altair as alt
import pandas as pd

alt.renderers.enable('altair_viewer')


def main(args):
    df = pd.DataFrame({'Org': ['New Information', 'New Information'],
                       'response': ['Yes', 'No'],
                       'percent': [91.53, 8.47],
                       'value': [5, 59],
                       'position': [2.5, 10.5]})

    color_scale = alt.Scale(
        domain=[
            "No",
            "Yes"
        ],
        range=["#CCCC00", '#1770ab']
    )

    y_axis = alt.Axis(
        title='',
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False
    )

    chart = alt.Chart(df).mark_bar(size=12).encode(
        x=alt.X('percent', title='Percentage'),
        y=alt.Y('Org', title='', axis=y_axis),
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
