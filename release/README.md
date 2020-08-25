# Paper: Automatically Selecting Follow-up Questions for Deficient Bug Reports
In the following, we briefly describe the different components that are included in this project and the softwares required to run the experiments.

## Project Setup
  - data: please unzip data.zip in the main directory of the project
  - embedding: Please download embeddings and unzip it in the main directory of the project: https://drive.google.com/file/d/1VCPUWwAC8LfvWhOvoUNZqBb9PIXnokX_/view?usp=sharing

## Project Structure
The project includes the following files and folders:

  - __/data__: A folder that contains inputs that are used for the experiments
	- dataset.csv: CSV file that contains 25k bug reports with follow-up questions and answers
	- github_issue_titles.csv: Titles of those 25k bug reports
	- github_issue_labels.csv: Labels of those 25k bug reports
	- github_repo_labels.csv: Repository labels of those 25k bug reports
	- post_data.tsv: 25k bug reports by lucene output order
	- qa_data.tsv: 25k bug reports with 10 candidate questions selected by lucene output order
	- test_ids.txt: test dataset ids
	- train_ids.txt: train dataset ids
  - __/embeddings_damevski__: A folder that contains the embeddings we have used
  - __/scripts__: Contains scripts for running the experiments
    - run_main.sh: the entry point of the experiment



## Software Requirements
We used the following softwares to run our experiments
  * Python3.6
  * torch
  * torchvision
  * spacy>=2.2.4
  * numpy pandas gensim jsonschema
  * Conda

## Setup
Conda
```
conda install numpy pandas gensim jsonschema
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
python -m spacy download en_core_web_sm
```

Conda currently does not provide required version of spacy, so we use pip to install spacy:

```
conda install pip
pip install spacy==2.2.4
```

## Running Experiments
Step1: Install software requirements mentioned above.

Step2: Update the filepaths and parameters in *script/models/run_main.sh*

Step3: `./src/models/run_main.sh`