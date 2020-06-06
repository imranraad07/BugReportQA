# Running FUTURE

### Requirements:
* Python3.6
* torch
* torchvision
* spacy>=2.2.4

### Setup
1. Embeddings
* download embeddings and unzip in *future/*: https://drive.google.com/drive/u/1/folders/1HDtSsT0__NmGkJVkg7ehJRlvFGpPwlg_

2. Data
* download data and unzip in *data/*: https://drive.google.com/drive/u/1/folders/1HDtSsT0__NmGkJVkg7ehJRlvFGpPwlg_

Data contains Lucene output files *post_data.tsv* and *qa_data.tsv*. Files *train_ids.txt* and *test_ids.txt*
are used to divide the data for training and evaluation.

If you want to crate a new dataset, follow this step:
1. Use *data/bug_reports/join_datasets.py* to create one csv file based on github_data*.csv files 
in the *data/bug_reports/* folder. To obtain a fraction of the whole dataset use `--fraction` option.
2. Use *data/bug_reports/partition_datasets.py* to generate train\tune\test ids.
3. Run Lucene on the generated dataset to generate *post_data.tsv* and *qa_data.tsv*. Use 
*src/data_generation/run_data_generator_github.sh*. To make it work you need to update paths to in 
*src/data_generation/run_data_generator_github.sh* and *lucene/run_lucene.sh*.

After this step you should have the following files: *post_data.tsv*, *qa_data.tsv*, *train_ids.txt*, 
*tune_ids.txt*, *test_ids.txt*.

3. Conda

``
conda install numpy pandas gensim jsonschema
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
python -m spacy download en_core_web_sm
``

Conda currently does not provide required version of spacy, so we use pip to install spacy:

``
conda install pip
pip install spacy==2.2.4
``

### How to run: 

* Update filepaths and parameters in *src/models/run_main.sh*
* `./src/models/run_main.sh`
