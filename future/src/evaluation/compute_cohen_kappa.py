import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sys


def convert_pd_to_1d(df):
	# convert dataframe to matrix
	conv_df = df[['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10']]
	conv_df[conv_df == 'V'] = 1
	conv_df[conv_df == 'v'] = 1
	conv_df[conv_df != 1] = 0
	conv_arr = conv_df.values
	return conv_arr.flatten().astype(int)


def main(args):
	df_x = pd.read_csv(args.human_annotations_filename[0],index_col=None,header=0)
	df_y = pd.read_csv(args.human_annotations_filename[1],index_col=None,header=0)

	assert(all(df_x['issue_id'] == df_y['issue_id']))
	df_arr_x = convert_pd_to_1d(df_x)
	df_arr_y = convert_pd_to_1d(df_y)

	kappa = cohen_kappa_score(df_arr_x,df_arr_y)
	print("Cohen's Kappa = " + str(kappa))

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--human_annotations_filename", type=str, nargs=2)
	args = argparser.parse_args()
	print(args)
	print("")
	main(args)

