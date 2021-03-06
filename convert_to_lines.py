import os
import json
import argparse
import copy

from sentencepiece import SentencePieceProcessor

from parsers import MODES

def main(train_path, val_path, test_path, mode, subword_model_path, output_dir, max_source_subwords,
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

    parse = MODES.get(mode, None)
    assert parse is not None

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=MODES.keys())
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subword-model-path', type=str, required=True)
    parser.add_argument('--max-source-subwords', type=int, default=None)
    parser.add_argument('--max-target-subwords', type=int, default=None)
    parser.add_argument('--source-suffix', type=str, default='bpe.source')
    parser.add_argument('--target-suffix', type=str, default='bpe.target')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
