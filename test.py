import wordfreq
import json

with open('wordfreq-en-25000-log.json', 'r') as f:
    frequency_data = json.load(f)

with open('word_to_index.json', 'r', encoding='utf-8') as f:
    get_index_dict = json.load(f)


def fetch_list(interval: int, iterations: int):
    test_words_list = []
    for i in range(1, iterations+1):
        test_words_list.append(get_word_at_index(interval*i))
    return test_words_list

def get_word_at_index(index: int):
    if index < len(frequency_data):
        return frequency_data[index - 1][0]
    else:
        return None

def test_words(word_frequency_initial_interval: int, initial_word_count: int):
    fetched_words_list = fetch_list(word_frequency_initial_interval, initial_word_count)
    tested_words_scoring = {}
    for word in fetched_words_list:
        while True:
            user_answer = input("Do you know this word:" + word + "\n1: I don't know it   2: I recognize it, but can't define it   3: I know it\n")
            if user_answer == "1" or user_answer == "2" or  user_answer == "3":
                tested_words_scoring[get_index_dict[word]] = user_answer
                break
    print(tested_words_scoring)

test_words(5, 5)