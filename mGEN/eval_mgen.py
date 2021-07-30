import argparse
import ast
import logging
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

from transformers import BartForConditionalGeneration, MT5ForConditionalGeneration, AutoTokenizer
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils import exact_match_score, f1_score  # noqa: E402 # isort:skip
from utils import metric_max_over_ground_truths, get_scores, get_precision_at_k  # noqa: E402 # isort:skip


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()


def evaluate_batch_e2e(args, model, tokenizer, questions):
    with torch.no_grad():
        inputs_dict = tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs_dict.input_ids.to(args.device)
        attention_mask = inputs_dict.attention_mask.to(args.device)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=1,
            # BART likes to repeat BOS tokens, dont allow it to generate more than one,
            bad_words_ids=[[0, 0]],
            output_scores=args.output_scores,
            return_dict_in_generate=args.output_scores
        )
        if args.output_scores is True:
            sequences_scores = outputs["sequences_scores"]
            answers = tokenizer.batch_decode(
                outputs["sequences"], skip_special_tokens=True)

            if args.print_predictions:
                for q, a in zip(questions, answers):
                    logger.info("Q: {} - A: {}".format(q, a))

            return answers, sequences_scores
        else:
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if args.print_predictions:
                for q, a in zip(questions, answers):
                    logger.info("Q: {} - A: {}".format(q, a))
            return answers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["mt5", "bart"],
        type=str,
        help="model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument("--k", default=1, type=int,
                        help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help="Format of the gold data file"
        "qa - a single line in the following format: question [tab] answer_list"
        "ans - a single line of the gold file contains the expected answer string",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length", default=1, type=int,
                        help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int,
                        help="Max length of the generated answers")

    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    parser.add_argument(
        "--output_scores",
        action="store_true",
        help="If True, output the prediction scores",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main(args):
    model_kwargs = {}
    if args.model_type == "bart":
        model_class = BartForConditionalGeneration
    elif args.model_type == "mt5":
        model_class = MT5ForConditionalGeneration
    else:
        raise NotImplementedError

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    score_fn = get_scores
    evaluate_batch_fn = evaluate_batch_e2e

    for checkpoint in checkpoints:
        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(
                args.predictions_path))
            score_fn(args, args.predictions_path, args.gold_data_path)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(
            args.predictions_path))

        model = model_class.from_pretrained(checkpoint, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True)
        model.to(args.device)

        with open(args.evaluation_set, "r") as eval_file, open(args.predictions_path, "w") as preds_file, open("{}_score".format(args.predictions_path), "w") as preds_file_score:
            questions = []
            for line in tqdm(eval_file):
                questions.append(line.strip())
                if len(questions) == args.eval_batch_size:
                    if args.output_scores is True:
                        answers, scores = evaluate_batch_fn(
                            args, model, tokenizer, questions)
                        print(scores)
                        for score in list(scores):
                            preds_file_score.write(str(float(score)))
                            preds_file_score.write("\n")
                        preds_file_score.flush()
                    else:
                        answers = evaluate_batch_fn(
                            args, model, tokenizer, questions)

                    preds_file.write("\n".join(answers) + "\n")
                    preds_file.flush()

                    questions = []
            if len(questions) > 0:
                if args.output_scores is True:
                    answers, scores = evaluate_batch_fn(
                        args, model, tokenizer, questions)
                    for score in list(scores):
                        preds_file_score.write(str(float(score)))
                        preds_file_score.write("\n")
                    preds_file_score.flush()
                else:
                    answers = evaluate_batch_fn(
                        args, model, tokenizer, questions)
                preds_file.write("\n".join(answers))
                preds_file.flush()

            score_fn(args, args.predictions_path, args.gold_data_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
