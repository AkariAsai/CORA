import argparse
import os
from doc_db import DocDB
import re
import json
from tqdm import tqdm
import csv


def collect_text_data(db_path, start_idx, separate=False):
    count = start_idx
    if separate is True:
        lang_doc_data = {}
    for path in db_path:
        db = DocDB(path)
        doc_ids = db.get_doc_ids()
        if separate is True:
            lang_doc_data[path] = []
        else:
            doc_data = []
        for doc_id in tqdm(doc_ids):
            sections_paras = db.get_doc_text_section_separations(doc_id)
            title = doc_id
            if "_0" in doc_id:
                title = doc_id.split("_0")[0]
            para_text = ""
            for section in sections_paras:
                paragraphs = section["paragraphs"]
                for para_idx, para in enumerate(paragraphs):
                    para_text += para
                    para_text += " "
            if len(para_text) > 0 and para_text[-1] == " ":
                para_text = para_text[:-1]
            para_tokens = para_text.split()
            if len(para_tokens) < 20:
                continue
            for i in range(len(para_tokens) // 100):
                w100_para_text = " ".join(para_tokens[100*i:100*(i+1)])
                if separate is True:
                    lang_doc_data[path].append(
                        {"title": title, "id": count, "text": w100_para_text})
                else:
                    doc_data.append(
                        {"title": title, "id": count, "text": w100_para_text})
                count += 1
            # store the last part if the remaining part is longer than 20.
            if len(para_tokens) % 100 > 20:
                w100_para_text = " ".join(
                    para_tokens[100*(len(para_tokens) // 100):])
                if separate is True:
                    lang_doc_data[path].append(
                        {"title": title, "id": count, "text": w100_para_text})
                else:
                    doc_data.append(
                        {"title": title, "id": count, "text": w100_para_text})
            count += 1
        if separate is True:
            print("collected {0} data from {1}".format(
                len(lang_doc_data[path]), path))
    print("collected {} data".format(count))

    if separate is True:
        return lang_doc_data
    else:
        return doc_data


def write_para_data_to_tsv(input_data, output_fn):
    with open(output_fn, 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for item in tqdm(input_data):
            tsv_writer.writerow([item["id"], item["text"], item["title"]])
    print("wrote full data to {}".format(output_fn))


def write_para_data_to_tsvs(input_data, output_dir):
    for k, para_data in input_data.items():
        db_path_name = os.path.basename(k)
        lang_code = db_path_name[:2]

        with open(os.path.join(output_dir, "{}_wiki_w100.tsv".format(lang_code)), 'wt') as tsv_file:
            tsv_writer = csv.writer(
                tsv_file, delimiter='\t', lineterminator='\n')
            for item in tqdm(para_data):
                tsv_writer.writerow([item["id"], item["text"], item["title"]])
        print("wrote {0} {1}'s data to {2}".format(len(para_data), lang_code, os.path.join(
            output_dir, "{}_wiki_w100.tsv".format(lang_code))))
        print(para_data[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True, nargs='+',
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--tsv_path', type=str, required=True,
                        help='output tsv file name')
    parser.add_argument('--separate', action="store_true")
    parser.add_argument('--tsv_dir', type=str)
    parser.add_argument('--start_idx', type=int, default=0)

    args = parser.parse_args()
    input_data = collect_text_data(
        args.db_path, args.start_idx, separate=args.separate)
    if args.separate is True:
        if not os.path.exists(args.tsv_dir):
            os.makedirs(args.tsv_dir)
        write_para_data_to_tsvs(input_data, args.tsv_dir)
    else:
        write_para_data_to_tsv(input_data, args.tsv_path)


if __name__ == '__main__':
    main()
