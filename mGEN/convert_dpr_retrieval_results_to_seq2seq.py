import json
import random
import argparse
import csv
import os
from tqdm import tqdm
import jsonlines


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def load_dpr_results(pred_results, top_n=5, split="train", lang_balance=False, orig_xor_eng_ids=None, lang_dict=None):
    q_c_a = []
    has_answer = 0
    for item in tqdm(pred_results):
        question = item["question"]
        answers = item["answers"]
        q_id = item["q_id"]
        ctxs = item["ctxs"]
        for ctx in ctxs:
            if ctx["has_answer"] == True:
                has_answer += 1
                break
        if split == "train":
            has_answer_context = []
            has_no_answer_context = []
            for ctx in ctxs:
                if ctx["has_answer"] is True:
                    has_answer_context.append(ctx)
                else:
                    has_no_answer_context.append(ctx)
            if len(has_answer_context) > 3:
                has_answer_context = random.sample(has_answer_context, k=3)
            negative_context_num = top_n - len(has_answer_context)
            has_no_answer_context = has_no_answer_context[:negative_context_num]

            paragraphs = [item for item in has_answer_context]
            paragraphs += [item for item in has_no_answer_context]
            random.shuffle(paragraphs)
        else:
            paragraphs = [item for item in ctxs[:top_n]]

        context = ""
        for idx, para in enumerate(paragraphs):
            if len(context) > 0 and context[-1] != " ":
                context += " "
            context += "<{0}: {1}> ".format(idx, para["title"])
            context += para["text"]

        if split != "train":
            lang = item["lang"]
        else:
            if item["lang"] == "NONE":
                lang = "en"
            # XOR full train data come with English answers, so we set the target language to English.
            elif item["lang"] != "NONE" and q_id in orig_xor_eng_ids:
                lang = "en"
            elif item["lang"] != "NONE" and q_id not in orig_xor_eng_ids:
                lang = item["lang"]
            else:
                raise NotImplementedError

        q_c_a.append({"question": question, "answers": answers,
                      "context": context, "lang": lang})
    print("Generated {0} train data; {1} data includes answer string.".format(
        len(q_c_a), has_answer))
    return q_c_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fp", default=None, type=str)
    parser.add_argument("--dev_fp", default=None, type=str)
    parser.add_argument("--test_fp", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--top_n", default=5, type=int)
    parser.add_argument("--add_lang", action="store_true")
    parser.add_argument("--xor_engspan_train", default=None, type=str)
    parser.add_argument("--xor_full_train", default=None, type=str)
    parser.add_argument("--xor_full_dev", default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train_fp is not None:
        train_data = json.load(open(args.train_fp))

    if args.dev_fp is not None:
        dev_data = json.load(open(args.dev_fp))
    if args.test_fp is not None:
        test_data = json.load(open(args.test_fp))

    if args.train_fp is not None:
        if args.add_lang is True:
            orig_xor_eng_ids = [item["id"]
                                for item in read_jsonlines(args.xor_engspan_train)]
            lang_dict_train = {item["id"]: item["lang"]
                               for item in read_jsonlines(args.xor_full_train)}
            s2s_train = load_dpr_results(
                train_data, top_n=args.top_n, orig_xor_eng_ids=orig_xor_eng_ids, lang_dict=lang_dict_train)
        else:
            s2s_train = load_dpr_results(train_data, top_n=args.top_n)
        source_f_train = open(os.path.join(
            args.output_dir, "train.source"), "w")
        target_f_train = open(os.path.join(
            args.output_dir, "train.target"), "w")

        for item in s2s_train:
            if args.add_lang:
                source_f_train.write("<Q>: {0} [{1}] <P>:{2}".format(
                    item["question"], item["lang"], item["context"]).replace("\n", "") + "\n")
            else:
                source_f_train.write("<Q>: {0} <P>:{1}".format(
                    item["question"], item["context"]).replace("\n", "") + "\n")
            target_f_train.write(item["answers"][0].replace("\n", "") + "\n")

        source_f_train.close()
        target_f_train.close()

    if args.dev_fp is not None:
        if args.add_lang is True:
            lang_dict_dev = {item["id"]: item["lang"]
                             for item in read_jsonlines(args.xor_full_dev)}
            s2s_dev = load_dpr_results(
                dev_data, top_n=args.top_n, split="dev", lang_dict=lang_dict_dev)
        else:
            s2s_dev = load_dpr_results(
                dev_data, top_n=args.top_n, split="dev")
        source_f_val = open(os.path.join(args.output_dir, "val.source"), "w")
        target_f_val = open(os.path.join(args.output_dir, "val.target"), "w")

        for item in s2s_dev:
            if args.add_lang:
                if args.top_n == 0:
                    source_f_val.write("<Q>: {0} [{1}]".format(
                        item["question"], item["lang"]).replace("\n", "") + "\n")

                else:
                    source_f_val.write("<Q>: {0} [{1}] <P>:{2}".format(
                        item["question"], item["lang"], item["context"]).replace("\n", "") + "\n")
            else:
                if args.top_n == 0:
                    source_f_val.write("<Q>: {0}".format(
                        item["question"]).replace("\n", "") + "\n")
                else:
                    source_f_val.write("<Q>: {0} <P>:{1}".format(
                        item["question"], item["context"]).replace("\n", "") + "\n")
            target_f_val.write(item["answers"][0].replace("\n", "") + "\n")

        source_f_val.close()
        target_f_val.close()

        with open(os.path.join(args.output_dir, "gold_para_qa_data_dev.tsv"), "w") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for item in s2s_dev:
                if args.add_lang:
                    tsv_writer.writerow(["<Q>: {0} [{1}] <P>:{2}".format(
                        item["question"], item["lang"], item["context"]), item["answers"]])
                else:
                    tsv_writer.writerow(["<Q>: {0} <P>:{1}".format(
                        item["question"], item["context"]), item["answers"]])

    if args.test_fp is not None:
        s2s_test = load_dpr_results(test_data, top_n=args.top_n, split="test")

        source_f_test = open(os.path.join(args.output_dir, "test.source"), "w")
        target_f_test = open(os.path.join(args.output_dir, "test.target"), "w")

        for item in s2s_test:
            source_f_test.write("<Q>: {0} <P>:{1}".format(
                item["question"], item["context"]).replace("\n", "") + "\n")
            target_f_test.write(item["answers"][0].replace("\n", "") + "\n")

        source_f_test.close()
        target_f_test.close()

        with open(os.path.join(args.output_dir, "gold_para_qa_data_test.tsv"), "w") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for item in s2s_test:
                tsv_writer.writerow(["<Q>: {0} <P>:{1}".format(
                    item["question"], item["context"]), item["answers"]])


if __name__ == "__main__":
    main()
