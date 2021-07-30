## mGEN
This directory contains the code for the mGEN component. The code is originally based on [the transformers' implementation of RAG](https://github.com/huggingface/transformers/tree/v4.2.1/examples/research_projects/rag).

### Data format

Our fine-tuning logic is based on scripts from [`examples/seq2seq`](https://github.com/huggingface/transformers/tree/master/examples/seq2seq). We accept training data in the same format as specified there - we expect a directory consisting of 6 text files: 
```bash
train.source
train.target
val.source
val.target
test.source
test.target
```
Each line contains each source/target sentence. 

#### Convert mDPR output to mGEN train data format
This scripts convert the DPR output file into mGEN train data format. Please set the file names for train, dev and test data (`--train_fp`, `--dev_fp`, and `--test_fp`) and the output directory name (`--output_dir`). You can choose the number of the top DPR retrieved passages (`--top_n`). 

```
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --train_fp /path/to/dpr/output --iterative \
    --output_dir /path/to/mgen/data/dir \
    --top_n 10 --add_lang
```

If you want to include language tags for the retriever input for the XOR QA training daa, you have to specify the path to the XOR QA files and set `--add_lang` option.

```sh
python convert_dpr_retrieval_results_to_seq2seq.py \
    --train_fp /path/to/your/dpr/output/dir \
    --output_dir /path/to/your/output/dir  \
    --xor_engspan_train /path/to/your/data/dir/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train /path/to/your/data/dir/xor_train_full.jsonl \
    --xor_full_dev /path/to/your/data/dir//xor_dev_full.jsonl \
    --top_n 10 \
    --add_lang
```

### Training
Please specify the `model_type`, `model_name_or_path` and `gpus` (the number of GPUs to be used during fine-tuning).

- Train `mt5-base` based model

```sh
python finetune_mgen.py \
    --data_dir /path/to/your/data/dir \
    --output_dir /path/to/output/dir \
    --model_name_or_path /path/to/previous_best_checkpoint \
    --model_type mt5 --gpus 8 \
    --do_train \
    --do_predict \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 1000  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 50 \
    --warmup_steps 500 
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
``` 

- Train `mt5-large` based model. We train our mGEN on 8 GPUs with 24GB memory, and we found that we cannot train the model even with `train_batch_size==1` when we use adam optimizer. To fine-tune mt5-large based model, you have to set `--adafactor` option. 

```sh
python finetune_mgen.py \
    --data_dir /path/to/your/data/dir \
    --output_dir /path/to/model/output/dir \
    --model_name_or_path /path/to/previous_best_checkpoint \
    --model_type mt5 --gpus 8 \
    --do_train \
    --do_predict \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 800  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 50 \
    --warmup_steps 500 
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --adafactor
``` 


### Evaluation

1. Run DPR
TO evaluate your trained mGEN model, you first need to retrieve passages using mDPR. Please follow the instruction in [mDPR](../mDPR) directory.

2. Convert DPR output
Please concert DPR output file as mentioned above.

3. Run mGEN
Please run the mGEN evaluation by running [`eval_mgen.py`](eval_mgen.py).

```
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path /path/to/model/output/dir \
    --evaluation_set /path/to/your/data/dir/val.source \
    --gold_data_path /path/to/your/data/dir/gold_para_qa_data_dev.tsv \
    --predictions_path mgen_output.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 8
```


