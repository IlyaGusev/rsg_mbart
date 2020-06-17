import os
import json
import argparse
from collections import defaultdict

import editdistance

from parsers import parse_rucos, parse_rwsd, parse_muserc

def main(predicted_path, test_path, output_path, mode):
    with open(predicted_path, "r") as r:
        records = defaultdict(dict)
        for line in r:
            if line.startswith("S"):
                fields = line.split("\t")
                idx = int(fields[0].split("-")[-1])
                source = fields[-1]
                records[idx]["idx"] = idx
                records[idx]["source"] = source
            if line.startswith("D"):
                fields = line.split("\t")
                idx = int(fields[0].split("-")[-1])
                label = fields[-1]
                records[idx]["idx"] = idx
                records[idx]["label"] = label

    records = list(records.values())
    records.sort(key=lambda x: x.get("idx"))

    if mode == "rucos":
        for record, passage in zip(records, parse_rucos(test_path)):
            entities = passage["entities"]
            label = record["label"].strip()
            distances = []
            for entity in entities:
                entity = entity.strip()
                distances.append((editdistance.eval(label, entity), entity))
            distances.sort()
            record["label"] = distances[0][1]
    elif mode == "danetqa":
        for record in records:
            record["label"] = (record["label"].strip() == "да")
    elif mode == "russe":
        for i, record in enumerate(records):
            label = record["label"].strip().split(" ")[0]
            assert label in ("да", "нет")
            records[i] = {"label": (label == "да"), "idx": record["idx"]}
    elif mode == "rwsd":
        true_spans = {i: r["target"] for i, r in enumerate(parse_rwsd(test_path))}
        for i, record in enumerate(records):
            idx = record["idx"]
            predicted_span = record["label"].strip()
            true_span = true_spans[idx]
            distance = editdistance.eval(true_span, predicted_span)
            label = distance <= 3 or predicted_span in true_span or true_span in predicted_span
            records[i] = {"idx": idx, "label": label}
    elif mode == "parus":
        for i, record in enumerate(records):
            label = int(record["label"].strip().split(" ")[0])
            records[i] = {"label": label, "idx": record["idx"]}
    elif mode == "muserc":
        true_records = {r["source"]: r for r in parse_muserc(test_path)}
        answers = defaultdict(defaultdict(dict))
        for i, record in enumerate(records):
            record_source = record["source"].replace("<unk>", "").strip()
            label = int(record["label"].strip() == "да")
            true_record = true_records[record_source]
            text_idx, question_idx, answer_idx = true_record["idx"]
            answers[text_idx][question_idx][answer_idx] = label
        records = []
        for text_idx, questions in answers.items():
            record = {"idx": text_idx, "passage": {"questions": []}}
            record_questions = record["passage"]["questions"]
            for question_idx, answers in questions.items():
                record_questions.append({"idx": question_idx, "answers": []})
                record_answers = record_questions[-1]["answers"]
                for idx, label in answers.items():
                    record_answers.append({"idx": idx, "label": label})
            records.append(record)
    elif mode == "rcb":
        for i, record in enumerate(records):
            label = record["label"].strip().split(" ")[0]
            assert label in ("entailment", "contradiction", "neutral")
            records[i] = {"label": label, "idx": record["idx"]}
    elif mode == "terra" or mode == "lidirus":
        for i, record in enumerate(records):
            label = record["label"].strip().split(" ")[0]
            label = "entailment" if label == "да" else "not_entailment"
            records[i] = {"label": label, "idx": record["idx"]}

    with open(output_path, "w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    modes = ("rucos", "danetqa", "russe", "rwsd", "parus", "rcb", "muserc", "terra", "lidirus")
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=modes)
    parser.add_argument('--test-path', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
