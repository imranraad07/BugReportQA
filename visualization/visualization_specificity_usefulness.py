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
         'value': [4, 9, 6, 21, 11,
                   1, 3, 7, 26, 14],
         'percentage': [7.84, 17.64, 11.76, 41.17, 21.56,
                        1.96, 5.88, 13.72, 50.98, 27.45],
         start1: [0, 7.84, 25.48, 37.24, 78.41,
                  0, 1.96, 7.84, 21.56, 72.54]
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
