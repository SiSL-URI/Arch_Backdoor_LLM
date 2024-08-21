# Arch_Backdoor_LLM

Code for the paper 'Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor'

Instructions for running the codes:

01. Install required libraries using the command 'pip install -r requirements.txt' .

02. For downloading datasets, preparing them in folders, and adding trigger words, run  'dataset_prep.py'

03. For launching architectural backdoor on pre-trained models use the 'experiment_pretrained.py' It takes three arguments:

First: Model name. Choose from 'bert' or  'distilbert'. Second: Model type. Choose from 'clean' or 'backdoored'. Third: dataset name. Choose from: 'emotion' or 'agnews' or 'finnews' or 'sst2' or 'imdb'

Example: To launch an architectural backdoor on the Bert model using the emotion dataset, run 'experiment_pretrained.py bert backdoored emotion'.

04.  For launching architectural backdoor on encoder-only transformer model use the 'experiment_encoder_only_transformer.py' It takes two arguments:

First: Model type. Choose from 'clean' or 'embed' or 'attn' or 'out' or  'all'. Second: dataset name. Choose from: 'emotion' or 'agnews' or 'finnews' or 'sst2' or 'imdb'

Example: To launch an architectural backdoor on the encoder-only transformer model using the emotion dataset and place the backdoor module after the embedding layer, 
run 'experiment_encoder_only_transformer.py embed emotion'.
