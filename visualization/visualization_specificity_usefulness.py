import argparse
import csv
import sys

import altair as alt

alt.renderers.enable('altair_viewer')


def main(args):
    start1 = "Percentage start"

    source = alt.pd.DataFrame(
        {'question': ['Specificity', 'Specificity', 'Specificity', 'Specificity', 'Specificity',
                      'Usefulness', 'Usefulness', 'Usefulness', 'Usefulness', 'Usefulness'],
         'type': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree',
                  'Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],
         'order': [1, 2, 3, 4, 5,
                   1, 2, 3, 4, 5],
         'value': [6, 12, 9, 25, 12,
                   1, 3, 13, 30, 17],
         'percentage': [9.37, 18.75, 14.06, 39.06, 18.75,
                        1.56, 4.68, 20.31, 46.87, 26.56],
         start1: [0, 9.37, 28.12, 42.18, 81.24,
                  0, 1.56, 6.24, 26.55, 73.42]
         })

    color_scale = alt.Scale(
        domain=[
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree"
        ],
        range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab"]
    )

    y_axis = alt.Axis(
        title='',
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False
    )

    chart = alt.Chart(source).mark_bar(size=12).encode(
        x=alt.X('percentage:Q', title='Percentage'),
        y=alt.Y('question:N', axis=y_axis),
        color=alt.Color(
            'type:N',
            legend=alt.Legend(orient='bottom', title='', padding=-12),
            scale=color_scale,
        ),
        order=alt.Order(
            'order',
            sort='ascending'
        ),
    )

    text = alt.Chart(source).mark_text(dx=6, color='white').encode(
        x=alt.X('Percentage start:Q', title=''),
        y=alt.Y('question:N'),
        detail='type:N',
        text=alt.Text('value:Q', format='.0f'),
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
