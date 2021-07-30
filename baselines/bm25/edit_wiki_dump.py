import json
import os
import argparse


def read_and_change_json(input_file, output_file, step):
    res = []
    with open(input_file, 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            single = json.loads(line)
            contents = single["text"].split('\n')
            for i in range(0, len(contents), step):
                edit = {"id": f'{single["id"]}-{str(i)}', "revid": single["revid"], "url": single["url"],
                        "title": single["title"], "contents": "\n".join(contents[i:min(i+step, len(contents))])}
                res.append(edit)

    with open(output_file, 'w+', encoding='utf-8') as f:
        for single in res:
            json.dump(single, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True,
                        help='The path to the input dir that contains jsonl file')
    parser.add_argument('--output-dir', required=True,
                        help='The path to the output dir that contains jsonl file')
    parser.add_argument('--step', default=2,
                        help='number of paragraphs in each article')
    args = parser.parse_args()
    for file in os.listdir(args.input_dir):
        if file.startswith('wiki'):
            input_filepath = os.path.join(args.input_dir, file)
            output_filepath = os.path.join(args.output_dir, file)
            read_and_change_json(input_filepath, output_filepath, args.step)


if __name__ == '__main__':
    main()
