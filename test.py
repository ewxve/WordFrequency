import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Scoring note (Bjork & Bjork, 1992): don't conflate retrieval vs storage strength.
# 0.3 keeps storage strength > 0 without flattening the curve.
base_scoring_dict = {
    "3": 1.0,
    "2": 0.3,  ### It's important for us not to treat retrieval strength and storage strength (Bjork & Bjork, 1992) as the same thing. In this case, both are low, but storage strength is nonzero. Using a number like 0.1 risks exaggerating a falloff in knowledge, while a higher number like 0.5 might flatten the curve too much
    "1": 0.0,
}

with open('words_by_frequency.json', 'r', encoding='utf-8') as f:
    wordlist = json.load(f)

ranked_words = [w for w, r in sorted(wordlist.items(), key=lambda kv: kv[1])]

word_history = set()  ### A list of all words that have already been given to the tester

word_scores_list = {}


def get_word_at_index(index: int):
    if 1 <= index <= len(ranked_words):
        return ranked_words[index - 1]
    return None


def ask_word(word: str):
    while True:
        user_answer = input(
            f"Do you know this word: {word}\n"
            "1: I don't know it   2: I recognize it, but can't define it   3: I know it\n"
        ).strip()
        if user_answer in ("1", "2", "3"):
            return base_scoring_dict[user_answer]
        print("Please enter 1, 2, or 3.")


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


def find_falloff():
    global word_history
    chunk_scores = {}
    chunk_list = [(1, 1000)]
    current_working_range = chunk_list[-1]

    ### FIRST PASS AT INTERVALS OF 1,000 WORDS
    list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 10)
    current_chunk_mean = test_list(list_to_test)
    chunk_scores[current_working_range] = current_chunk_mean

    while chunk_scores[chunk_list[-1]] >= 0.25:  ### Until we've found all test-worthy chunks of 1000
        chunk_list.append((chunk_list[-1][1] + 1, chunk_list[-1][1] + 1000))
        current_working_range = chunk_list[-1]
        width = current_working_range[1] - current_working_range[0]
        if chunk_scores[chunk_list[-2]] >= 0.95:
            step = width // 3  ### Speed up the testing significantly when the tester performed almost perfectly well on the last chunk
        elif chunk_scores[chunk_list[-2]] >= 0.8:
            step = width // 5  ### Speed up the testing when the tester performed incredibly well on the last chunk
        elif chunk_scores[chunk_list[-2]] <= 0.4:
            step = width // 15 ### Slow down the testing when the tester performed questionably on the last chunk
        else:
            step = width // 10
        list_to_test = list_words_within_span(current_working_range, step)
        current_chunk_mean = test_list(list_to_test)
        chunk_scores[current_working_range] = current_chunk_mean

    ### Then we do one more, to watch it get as close to zero as possible:
    chunk_list.append((chunk_list[-1][1] + 1, chunk_list[-1][1] + 1000))
    current_working_range = chunk_list[-1]
    list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 10)
    current_chunk_mean = test_list(list_to_test)
    chunk_scores[current_working_range] = current_chunk_mean

    median_chunk = ()
    median_chunk_score = min(chunk_scores.values(), key=lambda x: abs(x - 0.5))

    for key, value in chunk_scores.items():
        if value == median_chunk_score:
            median_chunk = key
            break

    if median_chunk != chunk_list[0] and median_chunk != chunk_list[-1]:
        chunk_list = [(median_chunk[0] - 1000, median_chunk[0] - 1), median_chunk, (median_chunk[1] + 1, median_chunk[1] + 1000)]  ### The median chunk along with the chunk to its immediate left and immediate right
    elif median_chunk == chunk_list[0] and median_chunk != chunk_list[-1]:
        chunk_list = [median_chunk, (median_chunk[1] + 1, median_chunk[1] + 1000)]  ### The median chunk along with the chunk to its immediate right
    else:
        chunk_list = [(median_chunk[0] - 1000, median_chunk[0] - 1), median_chunk]

    for chunk in chunk_list:
        current_working_range = chunk
        list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 35)  ### This time we want to get 35 words instead of 10
        current_chunk_mean = test_list(list_to_test)
        chunk_scores[current_working_range] = current_chunk_mean


def test_list(word_list: list):
    global word_history
    global word_scores_list
    current_chunk_scores = []
    for word in word_list:
        word_score = ask_word(word)
        word_scores_list[wordlist[word]] = word_score
        current_chunk_scores.append(word_score)
        word_history.add(word)
    current_chunk_mean = np.mean(current_chunk_scores)
    return current_chunk_mean


def list_words_within_span(span: tuple, interval: int):
    global word_history
    words_in_span = []
    current_index = max(1, span[0])
    interval = max(1, int(interval))

    while current_index <= span[1]:  ### Iterate and progress forward (by interval) until current index is past the ceiling of our range (we have gone through all words based on our interval)
        pending_word = get_word_at_index(current_index)
        if pending_word is None:
            break
        while pending_word in word_history:
            rank = wordlist[pending_word]
            next_rank = rank + 1
            if next_rank > span[1]:  ### This would mean any other words within our span have already been tested, so we just return the function
                return words_in_span
            pending_word = ranked_words[next_rank - 1]
        words_in_span.append(pending_word)
        current_index += interval
    return words_in_span


def graph_data():
    plot_data_fitted(sort_and_smooth_data(word_scores_list), word_scores_list)


find_falloff()
quit()
