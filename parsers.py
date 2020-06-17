import json

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
                idx = q["idx"]
                query = fix_white_spaces(query.replace("@placeholder", "pl"))
                answers = [x["text"] for x in q["answers"]] if "answers" in q else ["pl"]
                source = fields_to_source(text, query)
                for target in answers:
                    yield {"source": source, "target": target, "entities": entities, "idx": idx}


def parse_danetqa(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            text = record["passage"]
            question = record["question"]
            label = record.get("label", True)
            idx = record["idx"]
            source = fields_to_source(question, text)
            target = danet_label(label)
            yield {"source": source, "target": target, "idx": idx}


def parse_russe(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            idx = record["idx"]
            s1 = fix_white_spaces(record["sentence1"])
            s2 = fix_white_spaces(record["sentence2"])
            word = record["word"].strip()
            source = fields_to_source(word, s1, s2)

            label = record.get("label", True)
            sense1 = str(record.get("gold_sense1", 1))
            sense2 = str(record.get("gold_sense2", 1))
            target = " ".join((danet_label(label), sense1, sense2))

            yield {"source": source, "target": target, "idx": idx}


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
                yield {"source": source, "target": span1, "idx": idx}


def parse_parus(file_name):
    with open(file_name, "r", encoding='utf-8-sig') as r:
        for line in r:
            record = json.loads(line)
            premise = record["premise"]
            c1 = record["choice1"]
            c2 = record["choice2"]
            idx = record["idx"]
            question = record["question"]
            label = record.get("label", 0)
            source = fields_to_source(question, premise, c1, c2)
            target = SEPARATOR.join((str(label), c2 if label == 1 else c1))
            yield {"source": source, "target": target, "idx": idx}


def parse_terra(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            idx = record.get("idx")
            premise = record.get("premise", record.get("sentence1", None))
            hypothesis = record.get("hypothesis", record.get("sentence2", None))
            assert premise is not None and hypothesis is not None
            label = record.get("label", "entailment")
            source = fields_to_source(premise, hypothesis)
            target = danet_label(label == "entailment")
            yield {"source": source, "target": target, "idx": idx}


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


def parse_rcb(file_name):
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            idx = record["idx"]
            premise = record["premise"]
            hypothesis = record["hypothesis"]
            verb = record["verb"]
            negation = record.get("negation", record.get("no_negation", None))
            assert negation is not None
            label = record["label"]
            source = fields_to_source(verb, negation, premise, hypothesis)
            target = label
            yield {"source": source, "target": target, "idx": idx}


MODES = {
    "rucos": parse_rucos,
    "danetqa": parse_danetqa,
    "russe": parse_russe,
    "rwsd": parse_rwsd,
    "parus": parse_parus,
    "terra": parse_terra,
    "muserc": parse_muserc,
    "rcb": parse_rcb,
    "lidirus": parse_terra
}

