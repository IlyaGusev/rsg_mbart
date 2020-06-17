import os
import json
import argparse
import copy
import random

from sentencepiece import SentencePieceProcessor

from parsers import MODES, SEPARATOR

def main(input_dir, subword_model_path, output_dir, max_source_subwords,
         max_target_subwords, source_suffix, target_suffix, lowercase=False):
    processor = SentencePieceProcessor()
    processor.Load(subword_model_path)

    os.makedirs(output_dir, exist_ok=True)
    train_source_file = os.path.join(output_dir, "train.{}".format(source_suffix))
    train_target_file = os.path.join(output_dir, "train.{}".format(target_suffix))
    val_source_file = os.path.join(output_dir, "val.{}".format(source_suffix))
    val_target_file = os.path.join(output_dir, "val.{}".format(target_suffix))
    test_source_file = os.path.join(output_dir, "test.{}".format(source_suffix))
    test_target_file = os.path.join(output_dir, "test.{}".format(target_suffix))

    dirs = list(os.listdir(input_dir))
    tasks = []
    for d in dirs:
        if d.startswith("_"):
            continue
        mode = d.lower()
        parse = MODES.get(mode, None)
        assert parse is not None
        tasks.append((os.path.join(input_dir, d), mode, parse))

    files = (("train.jsonl", train_source_file, train_target_file),
             ("val.jsonl", val_source_file, val_target_file),
             ("test.jsonl", test_source_file, test_target_file))
    for orig_file_name, source_file_name, target_file_name in files:
        records = []
        for d, mode, parse in tasks:
            if orig_file_name != "test.jsonl" and mode == "lidirus":
                continue
            elif orig_file_name == "test.jsonl" and mode == "lidirus":
                path = os.path.join(d, "LiDiRuS.jsonl")
            else:
                path = os.path.join(d, orig_file_name)
            for record in parse(path):
                source = mode + SEPARATOR + str(record["idx"]) + SEPARATOR + record["source"]
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
                source = " ".join(source_subwords)
                target = " ".join(target_subwords)
                records.append((source, target))
        random.shuffle(records)
        with open(source_file_name, "w") as source_file, open(target_file_name, "w") as target_file:
            for source, target in records:
                source_file.write(source + "\n")
                target_file.write(target + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subword-model-path', type=str, required=True)
    parser.add_argument('--max-source-subwords', type=int, default=None)
    parser.add_argument('--max-target-subwords', type=int, default=None)
    parser.add_argument('--source-suffix', type=str, default='bpe.source')
    parser.add_argument('--target-suffix', type=str, default='bpe.target')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
