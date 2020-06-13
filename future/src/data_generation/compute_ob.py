import os
import pandas as pd
import calculator as calc


def compute_utilities(post_tsv, qa_tsv, utility_tsv):
    calculator = calc.Calculator()
    posts = pd.read_csv(post_tsv, sep='\t')
    qa = pd.read_csv(qa_tsv, sep='\t')

    utility_data = {'postids': list()}
    for i in range(1, 11):
        utility_data['p_a{0}'.format(i)] = list()

    for idx, row in posts.iterrows():
        print('Row {0}/{1}'.format(idx + 1, len(posts)))
        postid = row['postid']
        print(postid)
        post = row['title'] + ' ' + row['post']
        utility_data['postids'].append(postid)
        for i in range(1, 11):
            answer = qa.iloc[idx]['a' + str(i)]
            utility = calculator.utility(answer, post)

            utility_data['p_a{0}'.format(i)].append(utility)

    df = pd.DataFrame(utility_data)
    df.to_csv(utility_tsv, index=False, sep='\t')
