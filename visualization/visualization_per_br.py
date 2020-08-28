import argparse
import csv
import sys
import altair_saver as altsave

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
        range=[
            '#b32929', '#b05b5b', '#CFCFCF', '#59B362', '#248F2E',
            "#CCCC00", '#1770ab'
        ]
    )

    y_axis = alt.Axis(
        title='',
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False
    )
    fontSize = 8
    chart = alt.Chart(source).mark_bar(size=12).encode(
        x=alt.X('percentage:Q', title='Percentage',
                axis=alt.Axis(format='.0f')),
        y=alt.Y('id:N', axis=y_axis,
                sort=["FQ9", "FQ22", "FQ14", "FQ8", "FQ10",
                      "FQ12", "FQ19", "FQ18", "FQ17", "FQ2", "FQ6", "FQ3", "FQ5"]),
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
        width=180,
        height=200
    ).facet(
        column=alt.Column("quality:N", title=None, sort=["Specificity", "Usefulness", "New Information"])
    ).configure_axis(
        labelFontSize=fontSize,
        titleFontSize=fontSize,
    ).configure_text(
        fontSize=fontSize,
    )
    # chart.show()
    altsave.save(chart, fp='viz_combined.pdf')


# import bokeh.palettes as bp

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    # print(bp.Accent7)

    argparser.add_argument("--input_file", type=str, default='survey_report.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
