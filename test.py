import json
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

base_scoring_dict = {
    "3": 1.0,
    "2": 0.3, ### It's important for us not to treat retrieval strength and storage strength (Bjork & Bjork, 1992) as the same thing. In this case, both are low, but storage strength is nonzero. Using a number like 0.1 risks exaggerating a falloff in knowledge, while a higher number like 0.5 might flatten the curve too much
    "1": 0.0,
}

with open('words_by_frequency.json', 'r', encoding='utf-8') as f:
    wordlist = json.load(f)

ranked_words = [w for w, r in sorted(wordlist.items(), key=lambda kv: kv[1])]

def fetch_list(interval: int, iterations: int):
    test_words_list = []
    max_iters = min(iterations, len(ranked_words) // max(1, interval))
    for i in range(1, max_iters + 1):
        w = get_word_at_index(interval * i)
        if w is not None:
            test_words_list.append(w)
    return test_words_list


def get_word_at_index(index: int):
    if 1 <= index <= len(ranked_words):
        return ranked_words[index - 1]
    return None


def test_words(word_frequency_initial_interval: int, initial_word_count: int):
    fetched_words_list = fetch_list(word_frequency_initial_interval, initial_word_count)
    tested_words_scoring = {}
    for word in fetched_words_list:
        if word is None:
            continue
        while True:
            user_answer = input("Do you know this word: " + word + "\n1: I don't know it   2: I recognize it, but can't define it   3: I know it\n")
            if user_answer == "1" or user_answer == "2" or  user_answer == "3":
                tested_words_scoring[wordlist[word]] = base_scoring_dict[user_answer]
                break
    return tested_words_scoring

def group_words(user_scores: dict, group_size: int):
    #for key, value in user_scores.items():
    grouped_data = {}

    frequency_keys = list(user_scores.keys())
    frequency_keys.sort()
    highest_tested_word = frequency_keys[-1]

    number_of_groups = math.ceil(highest_tested_word / group_size) ### Decide the number of groups by dividing the rank of the highest word scored by the desired group size, then rounding up so all words will be included
    for current_group in range(1, number_of_groups+1):
        current_group_list = []

        starting_number = group_size * (current_group - 1) ### This keeps track of what number is the bottom margin for the group we are currently computing
        current_group_name = f"Group [{starting_number + 1}-{starting_number + group_size}]"
        for i in range(1, group_size+1):
            try:
                current_group_list.append(user_scores[starting_number + i])
            except KeyError:
                pass

        grouped_data[current_group_name] = current_group_list
    return grouped_data

def average_groups(grouped_data: dict):
    averaged = {}
    for k, vals in grouped_data.items():
        if vals:
            averaged[k] = sum(vals) / len(vals)
    return averaged


def sort_and_smooth_data(test_data: dict):
    x = np.array(list(test_data.keys()), dtype=float)
    y = np.array(list(test_data.values()), dtype=float)

    smoothed_data = lowess(
        y,
        x,
        frac=0.4,
        it=3,
        return_sorted=True
    )
    return smoothed_data

def plot_data_fitted(smoothed: np.ndarray, raw: dict | None = None):
    if raw is not None:
        rx = np.array(list(raw.keys()), dtype=float)
        ry = np.array(list(raw.values()), dtype=float)
        plt.scatter(rx, ry, s=25, label="Raw", alpha=0.8)

    plt.plot(smoothed[:, 0], smoothed[:, 1], color="green", label="LOWESS (robust)")

    x_min = 1
    x_max = float(np.max(smoothed[:, 0]))
    plt.xlim(x_min, x_max)
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.tight_layout()
    plt.show()

def console_test():
    word_interval = int(input("Desired interval between tested words: "))
    word_count = int(input("Number of words to test: "))
    test_results = test_words(word_interval, word_count)
    plot_data_fitted(sort_and_smooth_data(test_results), test_results)

console_test()
quit()