# Running GAN for question generation

### Requirements:
* Python2.7
* PyTorch
* nltk
* download embeddings and unzip in `GAN_question_generation`
  * `askubuntu_unix_superuser` from https://go.umd.edu/clarification_questions_embeddings
  * `stackoverflow` from  https://drive.google.com/file/d/1JxPJeXfCc8SZ8AkfJZfT37dEp42NBVL-/view?usp=sharing

### How to run: 

1. Pretrain models: 

   Before running the GAN, you need to pretrain question generator, answer generator and discriminator. For details, see Section 2.5 in [1].
  * Update filepaths to a dataset and embeddings in `src/run_pretraining.sh`
* Run `src/run_pretraining.sh`
* This will genereate params files used as the input for `src/run_GAN_main.sh`

2. Run GAN for question generation:

* Update filepath to a dataset and embeddings in `src/run_GAN_main.sh`
* If you don't have GPU, in `src/constants.py` set `CUDA=False`
* Run `src/run_GAN_main.sh`


### References

[1] <i><a href="https://www.aclweb.org/anthology/N19-1013">
Answer-based Adversarial Training for Generating Clarification Questions</a></i><br/>
Sudha Rao (Sudha.Rao@microsoft.com) and Hal Daumé III (me@hal3.name)<br/>
Proceedings of NAACL-HLT 2019
