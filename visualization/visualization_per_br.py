import argparse
import csv
import sys

import altair as alt

alt.renderers.enable('altair_viewer')


def main(args):
    source = alt.pd.read_csv(args.input_file)
    color_scale = alt.Scale(
        domain=[
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree",
            "No",
            "Yes"
        ],
        range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab", "#CCCC00", "#62c983"]
    )

    y_axis = alt.Axis(
        title='',
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False
    )

    chart = alt.Chart(source).mark_bar(size=12).encode(
        x=alt.X('percentage:Q', title='Percentage',
                axis=alt.Axis(format='.0f')),
        y=alt.Y('id:N', axis=y_axis,
                sort=["FQ5", "FQ4", "FQ10", "FQ6", "FQ13", "FQ12", "FQ7", "FQ8", "FQ9", "FQ11", "FQ3", "FQ1", "FQ2", ]),
        color=alt.Color(
            'type:N',
            legend=alt.Legend(orient='bottom', title='', padding=-12),
            scale=color_scale,
        ),
        order=alt.Order(
            'order',
            sort='ascending'
        ),
    ).properties(
        width=200,
        height=200
    ).facet(
        column=alt.Column("quality:N", title=None, sort=["Specificity", "Usefulness", "Novelty"])
    )

    combined = chart
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

    argparser.add_argument("--input_file", type=str, default='survey_report.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)