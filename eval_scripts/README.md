## Evaluation scripts
This directory includes the script used during evaluations.

### Convert MKQA data into XOR QA format
[MKQA (Longpre et al., 2020)](https://arxiv.org/abs/2007.15207) includes `unanswerable` questions to test models' ability of abstain from answering. Although those questions are really valuable, we exclude those questions without annotated short answers following the common practice in English Open-domain QA ([Lee et al., 2019](https://arxiv.org/abs/1906.00300); [Min et al., 2019](https://arxiv.org/abs/1909.04849); [Guu et al. 2020](https://arxiv.org/abs/2002.08909); [Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). 

This script will filter the questions without any annotated answer text and converts the original data into the [XOR QA full data](https://github.com/AkariAsai/XORQA) format. 

To generate the XOR QA-like input data files from the MKQA original data, please run the command below. 
```
python convert_mkqa_to_xorqa.py \
    --mkqa_orig_data /path/to/original/files \
    --output_data_dir /path/to/output/dir \
```

If you specify the target languages by setting the `--langs` option, the script only generates data for those specified target languages.       
If you want to keep the questions with no answer string, please set `--keep_unanswerable`option.

### Final QA performance evaluation
To evaluate the performance, please run the command below. The evaluation script is the same as [the original XOR QA full](https://github.com/AkariAsai/XORQA#evaluation). Please install `mecab` for Japanese tokenization.


The command accept both `json` format and `txt` format. 

- For the `json` format, your prediction file should be a dictionary whose keys are the question ID and values are the predicted answers as in SQuAD or XOR QA official submission. 
- For the `txt` format, each line should have a predicted answer to the t-th question in `data_file`. Sample prediction files in those format can be seen [here](). Please set `--txt_file` option if your prediction file in this format.

```
python eval_xor_full.py \
    --pred_file /path/to/mgen/pred.txt  \
    --data_file /path/to/xorqa/eval/data \
    --txt_file
```
To evaluate MKQA data, you have to pass the file path to the the target language's input data in the XOR QA format.

e.g., 
```
python eval_xor_full.py \
    --pred_file mkqa_japanese_predictions.json  \
    --data_file /path/to/conveted/mkqa/dir/input_ja.jsonl \
    --txt_file
```

Alternatively, you can run [eval_mkqa_all.py](eval_mkqa_all.py) to run the evaluations in all languages by a single command. 

```
python eval_mkqa_all.py \
    --data_gp /glob/file/expression/to/converted/mkqa/data \
    --pred_fp /glob/file/expression/to/predicted/results 
```
