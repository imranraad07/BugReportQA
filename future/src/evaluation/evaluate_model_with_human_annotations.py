import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import sys

#####
# https://gist.github.com/bwhite/3726239
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)
#
#########

def convert_to_valid_indices(human_df):
	col_to_idx_dict = { 'a1':'q1', 'a2':'q2', 'a3':'q3', 'a4':'q4', 'a5':'q5', 'a6':'q6', 'a7':'q7', 'a8':'q8', 'a9':'q9', 'a10':'q10' }
	indices = []
	for col in col_to_idx_dict.keys():
		seriesObj = human_df.apply(lambda x: True if 'v' in str(x[col]).lower() else False, axis=1)
		# valid only if all agree (i.e., set intersection)
		if seriesObj.all(): indices.append(col_to_idx_dict[col])
	#assert(len(indices) > 0)
	return indices


def evaluate_model(human_df, model_df):
	col_to_idx_dict = { 'q1':0, 'q2':1, 'q3':2, 'q4':3, 'q5':4, 'q6':5, 'q7':6, 'q8':7, 'q9':8, 'q10':9 }

	min_rank_of_ranks = []
	all_rank_of_ranks = []
	for curr_issue in human_df.issue_id.unique():
		ranking = [0] * 10

		curr_human_df = human_df.loc[human_df['issue_id'] == curr_issue]
		valid_indices = convert_to_valid_indices(curr_human_df)
		if len(valid_indices) == 0: continue

		curr_model_df = model_df.loc[model_df['issueid'] == curr_issue]
		assert(curr_model_df.shape[0] > 0)

		model_indices = []
		for col in col_to_idx_dict.keys():
			for valid_idx in valid_indices:
				valid_q = curr_human_df[valid_idx].iloc[0]
				curr_q = curr_model_df[col].iloc[0]
				if curr_q.strip() == valid_q.strip():
					model_indices.append(col_to_idx_dict[col])

		ranking[min(model_indices)] = 1
		min_rank_of_ranks.append(ranking)

		n_ranking = np.array(ranking)
		n_ranking[model_indices] = 1
		all_rank_of_ranks.append(list(n_ranking))

	mrr = mean_reciprocal_rank(min_rank_of_ranks)
	p_1, p_3, p_5 = 0.0, 0.0, 0.0
	for rank in all_rank_of_ranks:
		p_1 += precision_at_k(rank,1)
		p_3 += precision_at_k(rank,3)
		p_5 += precision_at_k(rank,5)
	p_1 = p_1 / len(all_rank_of_ranks)
	p_3 = p_3 / len(all_rank_of_ranks)
	p_5 = p_5 / len(all_rank_of_ranks)

	return mrr,p_1,p_3,p_5

def read_model_predictions(model_predictions_file):
	model_predictions = {}
	for line in model_predictions_file.readlines():
		splits = line.strip('\n').split()
		post_id = splits[0][1:-2]
		predictions = [float(val) for val in splits[1:]]
		model_predictions[post_id] = predictions
	return model_predictions

def main(args):
	model_df = pd.read_csv(args.model_predictions_filename)
	li = []
	for haf in args.human_annotations_filename:
		df = pd.read_csv(haf)
		li.append(df)
	human_df = pd.concat(li, axis=0, ignore_index=True)

	mrr,p_1,p_3,p_5 = evaluate_model(human_df, model_df)
	print("MRR = " + str(mrr))
	print("P@1 = " + str(p_1))
	print("P@3 = " + str(p_3))
	print("P@5 = " + str(p_5))

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--human_annotations_filename", type=str, nargs='+')
	argparser.add_argument("--model_predictions_filename", type=str)
	args = argparser.parse_args()
	print(args)
	print("")
	main(args)

