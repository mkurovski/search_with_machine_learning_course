import fasttext

top_words_filepath = "/workspace/datasets/fasttext/top_words.txt"
model_filepath = "/workspace/datasets/fasttext/title_model.bin"
synonym_output_filepath = "/workspace/datasets/fasttext/synonyms.csv"

threshold = 0.75
k = 10

synonym_model = fasttext.load_model(model_filepath)
with open(top_words_filepath, "r") as file:
    top_words = file.read()
    top_words = top_words.split()
    
assert len(top_words) == 1000

output = []

for top_word in top_words:
    candidates = synonym_model.get_nearest_neighbors(top_word, k)
    candidates = [synonym for cosine_sim, synonym in candidates if cosine_sim >= threshold]
    if len(candidates) > 0:
        line = f"{top_word}," + ",".join(candidates)
        output.append(line)
    else:
        output.append(top_word)

with open(synonym_output_filepath, "w") as file:
    file.write("\n".join(output))
