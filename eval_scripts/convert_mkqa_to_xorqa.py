import jsonlines
from collections import Counter
from tqdm import tqdm
import os
import argparse


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def convert_mkqa_data_into_xor(data, lang, keep_unanswerable=True):
    query_lang = []
    new_data = {l: [] for l in lang}
    for item in tqdm(data):
        example_id = item["example_id"]
        queries = item["queries"]
        answers = item["answers"]
        for query_lang, query in queries.items():
            if query_lang not in lang:
                continue
            final_answers = []
            for ans in answers[query_lang]:
                if ans["text"] is not None:
                    final_answers.append(ans["text"])
                if "aliases" in ans and len(ans["aliases"]) > 0:
                    for ali in ans["aliases"]:
                        final_answers.append(ali)
            if len(final_answers) >= 1:
                new_data[query_lang].append({"id": example_id, "question": query, "answers": final_answers,
                                            "lang": query_lang, "type": answers[query_lang][0]["type"]})
            else:
                if keep_unanswerable is True:
                    new_data[query_lang].append({"id": example_id, "question": query, "answers": [
                                                "NoAns"], "lang": query_lang, "type": answers[query_lang][0]["type"]})
    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mkqa_orig_data', type=str, required=True)
    parser.add_argument('--langs', type=str, nargs='+', default=None)
    parser.add_argument('--output_data_dir', type=str, required=True)
    parser.add_argument("--keep_unanswerable", action="store_true")
    args = parser.parse_args()

    mkqa_data = read_jsonlines(args.mkqa_orig_data)
    if args.langs is None:
        langs = args.langs
    else:
        langs = list(set(mkqa_data[0]["answers"].keys()))

    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    converted_data = convert_mkqa_data_into_xor(
        mkqa_data, langs, keep_unanswerable=args.keep_unanswerable)
    for lang in langs:
        output_filename = os.path.join(
            args.output_data_dir, "input_{}.jsonl".format(lang))
        with jsonlines.open(output_filename, 'w') as writer:
            writer.write_all(converted_data[lang])


if __name__ == "__main__":
    main()
