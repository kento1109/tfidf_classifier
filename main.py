import csv

import hydra
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sudachipy import dictionary, tokenizer

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.A


def tokenize(text):
    return " ".join([token.surface() for token in tokenizer_obj.tokenize(text, mode)])


def read_tsv(file_path, label_dict):
    tokens_list = []
    label_list = []
    with open(file_path) as file:
        reader = csv.reader(file, delimiter="\t")
        _ = next(reader)
        for row in reader:
            text = row[0]
            tokens_list.append(tokenize(text))
            label_list.append(label_dict[row[1]] if row[1] else None)
    return (tokens_list, label_list)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    label_dict = {label: i for i, label in enumerate(cfg.labels)}

    train_tokens_list, train_label_list = read_tsv(cfg.train_file_path, label_dict)
    test_tokens_list, test_label_list = read_tsv(cfg.test_file_path, label_dict)

    vectorizer = TfidfVectorizer(
        token_pattern="(?u)\\b\\w+\\b", min_df=cfg.train_params.min_count
    )

    vectorizer.fit(train_tokens_list)

    X_train = vectorizer.transform(train_tokens_list)
    X_test = vectorizer.transform(test_tokens_list)

    y_train = train_label_list
    y_test = test_label_list

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_prob = clf.predict(X_test)

    print(classification_report(y_test, y_prob))


if __name__ == "__main__":
    main()
