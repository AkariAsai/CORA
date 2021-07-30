import argparse
import os
from doc_db import DocDB
import re
import json
from tqdm import tqdm
import csv
import spacy

nlp = spacy.load("ja_core_news_sm")


def tokenize_japanese_text(text):
    tokens = []
    doc = nlp(text)
    for token in doc:
        tokens.append(token.text)
    return tokens


def collect_text_data(db_path, start_idx):
    db = DocDB(db_path)
    doc_ids = db.get_doc_ids()
    count = start_idx
    doc_data = []
    for doc_id in tqdm(doc_ids):
        sections_paras = db.get_doc_text_section_separations(doc_id)
        title = doc_id.split("_0")[0]
        para_tokens = []
        for section in sections_paras:
            paragraphs = section["paragraphs"]
            for para_idx, para in enumerate(paragraphs):
                para_tokens += tokenize_japanese_text(para)
        # skip articles whose para token is less than 20.
        if len(para_tokens) < 20:
            continue
        for i in range(len(para_tokens) // 100):
            w100_para_text = "".join(para_tokens[100*i:100*(i+1)])
            doc_data.append(
                {"title": title, "id": count, "text": w100_para_text})
            count += 1
        if len(para_tokens) % 100 > 20:
            w100_para_text = " ".join(
                para_tokens[100*(len(para_tokens) // 100):])
            doc_data.append(
                {"title": title, "id": count, "text": w100_para_text})
            count += 1

    print("collected {} data".format(count))

    return doc_data, count


def write_para_data_to_tsv(input_data, output_fn):
    with open(output_fn, 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for item in tqdm(input_data):
            tsv_writer.writerow([item["id"], item["text"], item["title"]])
    print("wrote full data to {}".format(output_fn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--tsv_path', type=str, required=True,
                        help='output tsv file name')
    parser.add_argument('--start_idx', type=int, default=0)

    args = parser.parse_args()
    input_data = collect_text_data(args.db_path, args.start_idx)
    write_para_data_to_tsv(input_data, args.tsv_path)


if __name__ == '__main__':
    main()
