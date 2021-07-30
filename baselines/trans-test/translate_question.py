from transformers import MarianTokenizer, MarianMTModel
import torch
import os
import argparse


def translate(model, tokenizer, device, src_txt):
    encoded_txt = tokenizer(src_txt, return_tensors="pt",
                            padding=True).to(device)
    tgt_text = model.generate(**encoded_txt)
    tgt_text = tokenizer.batch_decode(tgt_text, skip_special_tokens=True)
    return tgt_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        help='source language code')
    parser.add_argument('--target', default='en', type=str,
                        help='target language code')
    parser.add_argument('--input-file', type=str, required=True,
                        help='the path to the input question file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='the path to the output translated file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='the path to the model directory (contain multiple models)')
    parser.add_argument('--bsz', type=int, default=32, help='batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # decide model name
    if args.source == 'pt':
        model_name = os.path.join(
            args.model_dir, f'opus-mt-ROMANCE-{args.target}')
    elif args.source.startswith('zh'):
        model_name = os.path.join(args.model_dir, f'opus-mt-zh-{args.target}')
    elif args.source == 'he':
        model_name = os.path.join(args.model_dir, f'opus-mt-afa-{args.target}')
    elif args.source == 'no':
        model_name = os.path.join(args.model_dir, f'opus-mt-gem-{args.target}')
    elif args.source in ['km', 'ms']:
        model_name = os.path.join(args.model_dir, f'opus-mt-mul-{args.target}')
    else:
        model_name = os.path.join(
            args.model_dir, f'opus-mt-{args.source}-{args.target}')

    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # translate question file
    with open(args.input_file, 'r+', encoding='utf-8') as f_source:
        with open(args.output_file, 'w+', encoding='utf-8') as f_target:
            batch = []
            for line in f_source:
                line = line.strip('\r\n')
                if len(line) == 0:
                    continue
                batch.append(line)
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
