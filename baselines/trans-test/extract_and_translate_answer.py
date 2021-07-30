from transformers import MarianTokenizer, MarianMTModel
import torch
import os
import argparse
import json
from tqdm import tqdm


def translate(model, tokenizer, device, src_txt):
    encoded_txt = tokenizer(src_txt, return_tensors="pt",
                            padding=True).to(device)
    tgt_text = model.generate(**encoded_txt)
    tgt_text = tokenizer.batch_decode(tgt_text, skip_special_tokens=True)
    return tgt_text


def read_answer_and_translate(input_file, output_file, model, tokenizer, device, bsz, prefix=None):
    # read the answer files and translate them into target language
    batch = []
    with open(input_file, 'r+', encoding='utf-8') as f_source:
        with open(output_file, 'w+', encoding='utf-8') as f_target:
            qas = json.load(f_source)
            for i, qa in enumerate(qas):
                if prefix:
                    prediction = f'{prefix} '
                else:
                    prediction = ''
                prediction += qa["predictions"][0]["prediction"]["text"]
                batch.append(prediction)
                if len(batch) == bsz:
                    target_text = translate(model, tokenizer, device, batch)
                    for txt in target_text:
                        f_target.write(txt)
                        f_target.write('\n')
                        f_target.flush()
                    batch = []
            if len(batch) > 0:
                target_text = translate(model, tokenizer, device, batch)
                for txt in target_text:
                    f_target.write(txt)
                    f_target.write('\n')
                    f_target.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='en',
                        help='source language code')
    parser.add_argument('--target', required=True,
                        help='target language code')
    parser.add_argument('--input-file', type=str, required=True,
                        help='the path to the input answer json file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='the path to the output translated file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='the path to the model directory (contain multiple models)')
    parser.add_argument('--bsz', type=int, default=32, help='batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = None
    model_name = None

    # decide language prefix and model name
    if args.target == 'pt':
        model_name = os.path.join(
            args.model_dir, f'opus-mt-{args.source}-ROMANCE')
        prefix = '>>pt<<'
    elif args.target == 'he':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-afa')
        prefix = '>>heb<<'
    elif args.target == 'ja':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-jap')
    elif args.target == 'km':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-mul')
        prefix = '>>khm_Latn<<'  # khm is also ok
    elif args.target == 'no':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-gem')
        prefix = '>>nno<<'  # nob is also ok
    elif args.target == 'tr':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-trk')
        prefix = '>>tur<<'
    # chinese language codes: https://unicode-org.atlassian.net/browse/CLDR-11834
    elif args.target == 'zh-cn':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-zh')
        prefix = '>>cmn<<'  # cmn_Hans also ok
    elif args.target == 'zh-hk':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-zh')
        prefix = '>>yue_Hant<<'
    elif args.target == 'zh-tw':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-zh')
        prefix = '>>cmn_Hant<<'
    elif args.target == 'th':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-mul')
        prefix = '>>tha<<'
    elif args.target == 'pl':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-sla')
        prefix = '>>pol<<'
    elif args.target == 'ms':
        model_name = os.path.join(args.model_dir, f'opus-mt-{args.source}-mul')
        prefix = '>>zsm_Latn<<'
    elif args.target == 'vi':
        model_name = os.path.join(
            args.model_dir, f'opus-mt-{args.source}-{args.target}')
        prefix = '>>vie<<'
    elif args.target == 'ar':
        model_name = os.path.join(
            args.model_dir, f'opus-mt-{args.source}-{args.target}')
        prefix = '>>ara<<'
    elif args.target == 'ko':
        print("Currently Korean is not supported ;)!")
        exit(0)
    else:
        model_name = os.path.join(
            args.model_dir, f'opus-mt-{args.source}-{args.target}')

    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    batch = []
    with open(args.input_file, 'r+', encoding='utf-8') as f_source:
        with open(args.output_file, 'w+', encoding='utf-8') as f_target:
            qas = json.load(f_source)
            for i, qa in enumerate(qas):
                if prefix:
                    prediction = f'{prefix} '
                else:
                    prediction = ''
                prediction += qa["predictions"][0]["prediction"]["text"]
                batch.append(prediction)
                if len(batch) == args.bsz:
                    target_text = translate(model, tokenizer, device, batch)
                    for txt in target_text:
                        f_target.write(txt)
                        f_target.write('\n')
                        f_target.flush()
                    batch = []
            if len(batch) > 0:
                target_text = translate(model, tokenizer, device, batch)
                for txt in target_text:
                    f_target.write(txt)
                    f_target.write('\n')
                    f_target.flush()


if __name__ == '__main__':
    main()
