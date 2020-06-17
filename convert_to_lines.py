import os
import json
import argparse
import copy

from sentencepiece import SentencePieceProcessor

SEPARATOR = " sep "

def fix_white_spaces(text):
    return ' '.join(text.strip().split())


def danet_label(label):
    return "да" if label else "нет"


def fields_to_source(*fields):
    return SEPARATOR.join(fields + ("", )).strip()


def parse_rucos(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            text = record["passage"]["text"]
            entities = [text[x["start"]: x["end"]] for x in record["passage"]["entities"]]
            text = fix_white_spaces(text.replace("@highlight", "hl"))
            for q in record["qas"]:
                query = q["query"]
                query = fix_white_spaces(query.replace("@placeholder", "pl"))
                answers = [x["text"] for x in q["answers"]] if "answers" in q else ["pl"]
                source = fields_to_source(text, query)
                for target in answers:
                    yield {"source": source, "target": target, "entities": entities}


def parse_danetqa(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            text = record["passage"]
            question = record["question"]
            label = record.get("label", True)
            source = fields_to_source(question, text)
            target = danet_label(label)
            yield {"source": source, "target": target}


def parse_russe(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            s1 = fix_white_spaces(record["sentence1"])
            s2 = fix_white_spaces(record["sentence2"])
            word = record["word"].strip()
            source = fields_to_source(word, s1, s2)

            label = record.get("label", True)
            sense1 = str(record.get("gold_sense1", 1))
            sense2 = str(record.get("gold_sense2", 1))
            target = " ".join((danet_label(label), sense1, sense2))

            yield {"source": source, "target": target}


def parse_rwsd(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            idx = record["idx"]
            text = fix_white_spaces(record["text"])
            span1 = fix_white_spaces(record["target"]["span1_text"])
            span2 = fix_white_spaces(record["target"]["span2_text"])
            source = fields_to_source(text, span2)
            target = span1

            label = record.get("label", None)
            if label or label is None:
                yield {"source": source, "target": span1}


def parse_parus(file_name):
    with open(file_name, "r", encoding='utf-8-sig') as r:
        for line in r:
            record = json.loads(line)
            premise = record["premise"]
            c1 = record["choice1"]
            c2 = record["choice2"]
            question = record["question"]
            label = record.get("label", 0)
            source = fields_to_source(question, premise, c1, c2)
            target = SEPARATOR.join((str(label), c2 if label == 1 else c1))
            yield {"source": source, "target": target}


def parse_terra(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            premise = record["premise"]
            hypothesis = record["hypothesis"]
            label = record.get("label", "entailment")
            source = fields_to_source(premise, hypothesis)
            target = danet_label(label == "entailment")
            yield {"source": source, "target": target}


def parse_muserc(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            text_idx = record["idx"]
            text = record["passage"]["text"]
            questions = record["passage"]["questions"]
            for q in questions:
                question = q["question"]
                answers = q["answers"]
                question_idx = q["idx"]
                for a in answers:
                    answer = a["text"]
                    answer_idx = a["idx"]
                    label = int(a["label"]) if "label" in a else 0
                    source = fields_to_source(answer, question, text)
                    target = danet_label(bool(label))
                    yield {"source": source, "target": target, "idx": [text_idx, question_idx, answer_idx]}


def main(train_path, val_path, test_path, mode, subword_model_path, out_dir, max_source_subwords,
         max_target_subwords, source_suffix, target_suffix, lowercase=False):
    processor = SentencePieceProcessor()
    processor.Load(subword_model_path)

    train_source_file = os.path.join(out_dir, "train.{}".format(source_suffix))
    train_target_file = os.path.join(out_dir, "train.{}".format(target_suffix))
    val_source_file = os.path.join(out_dir, "val.{}".format(source_suffix))
    val_target_file = os.path.join(out_dir, "val.{}".format(target_suffix))
    test_source_file = os.path.join(out_dir, "test.{}".format(source_suffix))
    test_target_file = os.path.join(out_dir, "test.{}".format(target_suffix))

    parse = None
    if mode == "rucos":
        parse = parse_rucos
    elif mode == "danetqa":
        parse = parse_danetqa
    elif mode == "russe":
        parse = parse_russe
    elif mode == "rwsd":
        parse = parse_rwsd
    elif mode == "parus":
        parse = parse_parus
    elif mode == "terra":
        parse = parse_terra
    elif mode == "muserc":
        parse = parse_muserc
    else:
        assert False

    files = ((train_path, train_source_file, train_target_file),
             (val_path, val_source_file, val_target_file),
             (test_path, test_source_file, test_target_file))
    for path, source_file_name, target_file_name in files:
        with open(source_file_name, "w") as source_file, open(target_file_name, "w") as target_file:
            for record in parse(path):
                source = record["source"]
                target = record["target"]
                if lowercase:
                    source = source.lower()
                    target = target.lower()
                source_subwords = processor.EncodeAsPieces(source)
                if max_source_subwords:
                    source_subwords = source_subwords[:max_source_subwords]
                target_subwords = processor.EncodeAsPieces(target)
                if max_target_subwords:
                    target_subwords = target_subwords[:max_target_subwords]
                source_file.write(" ".join(source_subwords) + "\n")
                target_file.write((" ".join(target_subwords)) + "\n")

if __name__ == "__main__":
    modes = ("rucos", "danetqa", "russe", "rwsd", "parus", "terra", "muserc")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=modes)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--subword-model-path', type=str, required=True)
    parser.add_argument('--max-source-subwords', type=int, default=None)
    parser.add_argument('--max-target-subwords', type=int, default=None)
    parser.add_argument('--source-suffix', type=str, default='bpe.source')
    parser.add_argument('--target-suffix', type=str, default='bpe.target')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
