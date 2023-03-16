import fasttext
import nltk
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# add routine in python to load and shuffle it myself and write back
k = 1

dataset_filepath = "/workspace/datasets/fasttext/pruned_labeled_products.txt"
dataset_filepath = "/workspace/datasets/fasttext/normalized_labeled_products.txt"
# dataset_filepath = "/workspace/datasets/fasttext/labeled_products.txt"
train_filepath = "/workspace/datasets/fasttext/train_pruned.txt"
test_filepath = "/workspace/datasets/fasttext/test_pruned.txt"

n_train = 10000
n_test = 10000

random_seed = 21
np.random.seed(random_seed)

hyperparams = {
    "epoch": 25,
    "lr": 1.0,
    "wordNgrams": 2,
    "ws": 3
}

def load_shuffle_split():
    with open(dataset_filepath, "r") as file:
        data = file.read()
        
    data = data.split("\n")[:-1]
    print(f"{len(data)} entries in overall dataset")

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
    print(f"{len(overlap)} unq labels found in both sets")

    with open(train_filepath, "w") as file:
        file.write("\n".join(train))
        
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
    print("Loading Data ...")
    load_shuffle_split()
    train_data = convert_test_file(train_filepath)
    test_data = convert_test_file(test_filepath)

    train_data_labels = np.array([example["labels"][0] for example in train_data])
    test_data_labels = np.array([example["labels"][0] for example in test_data])
    unq_train_labels = np.unique(train_data_labels)
    unq_test_labels = np.unique(test_data_labels)
    label_intersection = np.intersect1d(
        unq_train_labels,
        unq_test_labels
    )

    print(f"{len(unq_train_labels)} unq. train labels")
    print(f"{len(unq_test_labels)} unq. test labels")
    print(f"{len(label_intersection)} labels in intersection")

    print("Training ...")
    model = fasttext.train_supervised(input=train_filepath, **hyperparams)

    print("Evaluation ...")
    precision, recall = get_prec_rec(model, test_data, k=k)
    print("Test Statistics:")
    print(f"P@{k}: {round(precision.mean(), 5)}")
    print(f"R@{k}: {round(recall.mean(), 5)}")

    precision, recall = get_prec_rec(model, train_data, k=k)
    print("Train Statistics:")
    print(f"P@{k}: {round(precision.mean(), 5)}")
    print(f"R@{k}: {round(recall.mean(), 5)}")

    print("\nFinished")