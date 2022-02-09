
## mDPR
This code is mostly same as the original DPR repository with some minor modifications. The code is based on [Dense Passage Retriever](https://github.com/facebookresearch/DPR) and we modify the code to support more recent version of huggingface transformers. 

### Installation
We tested the code with `transformers==3.0.2`, and you may find some issues if you use different version of transformers.

```
pip install transformers==3.0.2
```

### Data
We will add the data used for mDPR training. Please stay tuned!

### Training
1. Initial training 

We first train the DPR models using gold paragraph data from Natural Questions, XOR QA and TyDi QA. 

```
python -m torch.distributed.launch \
    -nproc_per_node=8 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-multilingual-uncased \
    --seed 12345 --sequence_length 256 \
    --warmup_steps 300 --batch_size 16  --do_lower_case \
    --train_file /path/to/train/data \
    --dev_file /path/to/eval/data \
    --output_dir /path/to/output/dir \
    --learning_rate 2e-05 --num_train_epochs 40 \
    --dev_batch_size 6 --val_av_rank_start_epoch 30
```

2. Generate Wikipedia embeddings
After you train the DPR encoders, you need to generate Wikipedia passage embeddings. Please create a Wikipedia passage file following the instruction in the `wikipedia_preprocess` directory. The script to generate multilingual embeddings using 8 GPUs is as follows:

```sh
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_dense_embeddings.py  --model_file /path/to/model/checkpoint --batch_size 64 --ctx_file /path/to/wikipedia/passage/file --shard_id ${i} --num_shards 8 --out_file ./embeddings_multilingual/wikipedia_split/wiki_emb > ./log/nohup.generate_wiki_emb.ser23_3_multi.${i} 2>&1 &
done
```
Note that when you generate embeddings for the 13 target languages, you may experience out of memory issue when you load the Wikipedia passage tsv file (the total wikipedia passage size is 26GB * 8 GPU). 
We recommend you to generate English embeddings first, and then do the same for the remaining languages. 

3. Retrieve Wikipedia passages for train data questions
Following prior work, we retrieve top passages for the train data questions and use them to train our generator. Once you generate train data, you can retrieve top passages by running the command below. 

```
python dense_retriever.py \
    --model_file /path/to/model/checkpoint \
    --ctx_file /path/to/wikipedia/passage/file --n-docs 100 \
    --qa_file /path/to/input/qa/file \
    --encoded_ctx_file "{glob expression for generated files}" \
    --out_file /path/to/prediction/outputs  \
    --validation_workers 4 --batch_size 64 
```

After run train your generator, please run the script to create new mDPR train data and repeat the steps from 1 using the new data. 

### Evaluations
You can run the evaluation using the same command as the step 3 in training.     
For example, to run the evaluation on the XOR QA dev data, you can run the command below.      

```
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
```
Due to the large number of the multilingual passages embeddings, retrieving passages takes more time than English only DPR.
