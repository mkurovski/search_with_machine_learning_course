"""
Solution for Level 1 of Week 3: Train and Evaluate a Query Classifier
"""
import argparse
import logging

import fasttext
import numpy as np
import pandas as pd


dataset_filepath = "/workspace/datasets/fasttext/labeled_queries.txt"
train_filepath = "/workspace/datasets/fasttext/train_labeled_queries.txt"
test_filepath = "/workspace/datasets/fasttext/test_labeled_queries.txt"

eval_k = [1, 2, 3]
hyperparams = {
    "epoch": 25,
    "lr": 0.5,
    "wordNgrams": 2,
}


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--n_train", type=int, default=50000,  help="Number of Train Examples")
general.add_argument("--n_test", type=int, default=10000, help="Number of Test Examples")
general.add_argument("--random_seed", type=int, default=42, help="Random Seed")
args = parser.parse_args()


def load_shuffle_split(dataset_filepath: str, train_filepath: str, test_filepath: str, n_train: int, n_test: int, random_seed: int):
    with open(dataset_filepath, "r") as file:
        data = file.read()
        
    data = data.split("\n")[:-1]
    _logger.info(f"{len(data)} entries in overall dataset")

    # check non overlapping criterion
    assert (len(data) - n_train) > n_test

    print("Shuffle and Split Data")
    np.random.shuffle(data)
    train = data[:n_train]
    test = data[-n_test:]

    train_labels = pd.Series([val.split()[0] for val in train])
    test_labels = pd.Series([val.split()[0] for val in test])

    unq_train_labels = train_labels.unique()
    unq_test_labels = test_labels.unique()

    overlap = np.intersect1d(unq_train_labels, unq_test_labels)

    # assert len(unq_train_labels) == len(overlap)
    _logger.info(f"{len(overlap)} unq labels found in both sets")

    _logger.info(f"Writing train data to {train_filepath}")
    with open(train_filepath, "w") as file:
        file.write("\n".join(train))
    
    _logger.info(f"Writing train data to {test_filepath}")
    with open(test_filepath, "w") as file:
        file.write("\n".join(test))


def convert_test_file(file_path: str="cooking.test") -> list:
    with open(file_path, "r") as file:
        test_scenarios = file.read()

    test_scenarios = test_scenarios.split("\n")
    test_data = []

    for line in test_scenarios[:-1]:
        labels = line.split()
        labels = [label for label in labels if label.startswith("__label__")]
        last_label = labels[-1]
        label_start = line.find(last_label)
        question = line[label_start+len(last_label)+1:]
        test_data.append(
            {
                "question": question,
                "labels": labels
            }
        )
              
    return test_data


def get_prec_rec(model, data: list, k: int=1) -> (list, list):
    precision = []
    recall = []

    for instance in data:
        predictions, scores = model.predict(instance["question"], k=k)
        num_hits = len(set(predictions).intersection(set(instance["labels"])))
        precision.append(num_hits/len(predictions))
        recall.append(num_hits/len(instance["labels"]))
        if len(instance["labels"]) == 0:
            break
            
    return np.array(precision), np.array(recall)


if __name__ == "__main__":
    _logger.info("Loading, Shuffling, Splitting Data ...")
    load_shuffle_split(dataset_filepath, train_filepath, test_filepath, args.n_train, args.n_test, args.random_seed)

    _logger.info("Training ...")
    model = fasttext.train_supervised(input=train_filepath, **hyperparams)

    _logger.info("Evaluation ...")
    test_data = convert_test_file(test_filepath)
    for k in eval_k:
        _logger.info(f"k = {k}")
        precision, recall = get_prec_rec(model, test_data, k=k)
        _logger.info(f"P@{k}: {round(precision.mean(), 5)}")
        _logger.info(f"R@{k}: {round(recall.mean(), 5)}")

    _logger.info("Finished.")

