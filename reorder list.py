import json

### I had to make a file which can reorder lists that have gaps in them. This is because I'm manually removing words from the list to clean the testing vocab pool (Things like names, etc.)
def normalize_ranks(input_file: str, output_file: str):
    ### Load the JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ### Reassign ranks sequentially from 1
    normalized = {word: idx + 1 for idx, (word, _) in enumerate(data.items())}

    ### Save new JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)

    return normalized

if __name__ == "__main__":
    result = normalize_ranks("words_by_frequency2.json", "words_by_frequency_reorder.json")
    print("Rank normalization succeeded")
    print(result)
