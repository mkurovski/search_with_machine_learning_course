import fasttext

input_data_filepath = "/workspace/datasets/fasttext/normalized_products.txt"
model_filepath = "/workspace/datasets/fasttext/title_model.bin"

hyperparams = {
    "model": "skipgram",
    "maxn": 6,
    "minCount": 100,
    "epoch": 25
}

k = 10
cosine_threshold = 0.71

seed_words = [
    "inkjet",
    "netbook",
    "headphones",
    "laptop",
    "freezer",
    "nintendo",
    "razr",
    "stratocaster",
    "holiday",
    "plasma",
    "leather"
]

sg_model = fasttext.train_unsupervised(input_data_filepath, **hyperparams)

for seed_word in seed_words:
    print(f"Seed Word: {seed_word}\n")
    for result in sg_model.get_nearest_neighbors(seed_word, k=k):
        if result[0] > cosine_threshold:
            print(result)
    print()

print(f"Saving Model to {model_filepath}")
sg_model.save_model(model_filepath)