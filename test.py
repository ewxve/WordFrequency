import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from bisect import bisect_right

# Scoring note (Bjork & Bjork, 1992): don't conflate retrieval vs storage strength.
# 0.3 keeps storage strength > 0 without flattening the curve.
base_scoring_dict = {
    "3": 1.0,
    "2": 0.3,  # retrieval vs storage strength
    "1": 0.0,
}

# CEFR thresholds keyed by word-count cutoffs (ranks). Must be sorted by key.
word_count_cefr_levels = {
    0: "A1 - Low",
    350: "A1 - Middle",
    650: "A1 - High",
    1000: "A2 - Low",
    1350: "A2 - Middle",
    1650: "A2 - High",
    2000: "B1 - Low",
    2650: "B1 - Middle",
    3350: "B1 - High",
    4000: "B2 - Low",
    4650: "B2 - Middle",
    5350: "B2 - High",
    6000: "C1 - Low",
    7000: "C1 - Middle",
    8000: "C1 - High",
    9000: "C2",
}

# Milestones for learner-friendly labels (approximate)
word_count_proficiency_labels = {
    0: "You're very new to learning English. You may be able to repeat basic greetings and make a few simple statements.",
    250: "You can understand greetings, numbers, and make very simple requests.",
    500: "You can order food, ask prices, and talk about yourself and your family.",
    750: "You can handle short conversations about everyday topics like weather or shopping.",
    1000: "You can manage most daily interactions and get around in English.",
    1500: "You can take part in longer conversations about familiar topics with some confidence.",
    2000: "You can follow simple stories, basic instructions, and speak in casual conversations.",
    3000: "You can live comfortably in an English-speaking country, follow TV with subtitles, and chat casually with natives.",
    4000: "You have strong conversational fluency and can discuss a wide variety of everyday topics.",
    5000: "You can study in an English-speaking school or workplace with some support.",
    6000: "You can read most novels and newspapers without much difficulty.",
    8000: "You can comfortably read magazines, news, and some academic writing on familiar topics.",
    10000: "You have professional-level fluency. You can read and write at an academic level with ease.",
    15000: "Your vocabulary is comparable to a native and educated English speaker.",
}

with open('words_by_frequency.json', 'r', encoding='utf-8') as f:
    wordlist = json.load(f)

# ranks are 1-based in get_word_at_index
ranked_words = [w for w, r in sorted(wordlist.items(), key=lambda kv: kv[1])]

word_history = set()           # words already tested
word_scores_list = {}          # {rank: score}
known_words = None


def get_word_at_index(index: int):
    if 1 <= index <= len(ranked_words):
        return ranked_words[index - 1]
    return None


def ask_word(word: str):
    while True:
        user_answer = input(
            f"Do you know this word: {word}\n"
            "1: I don't know it   2: I recognize it, but can't define/pronounce it   3: I know it\n"
        ).strip()
        if user_answer in ("1", "2", "3"):
            return base_scoring_dict[user_answer]
        print("Please enter 1, 2, or 3.")


def sort_and_smooth_data(test_data: dict):
    if not test_data:
        return np.empty((0, 2), dtype=float)
    x = np.array(list(test_data.keys()), dtype=float)
    y = np.array(list(test_data.values()), dtype=float)
    smoothed_data = lowess(y, x, frac=0.4, it=3, return_sorted=True)
    return smoothed_data


def plot_data_fitted(smoothed: np.ndarray, raw: dict | None = None, falloff_rank: float | None = None):
    if raw:
        rx = np.array(list(raw.keys()), dtype=float)
        ry = np.array(list(raw.values()), dtype=float)
        plt.scatter(rx, ry, s=25, label="Raw", alpha=0.8)

    if smoothed.size:
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="green", label="LOWESS (robust)")

    if falloff_rank is not None:
        plt.axvline(x=falloff_rank, color="red", linestyle="--", linewidth=1.5, label=f"Falloff ≈ {falloff_rank:.0f}")
        plt.annotate(f"{falloff_rank:.0f}", xy=(falloff_rank, 0.52), xytext=(5, 10),
                     textcoords="offset points", rotation=90, va="bottom", ha="left", fontsize=8)

    x_min = 1
    x_max = float(np.max(smoothed[:, 0])) if smoothed.size else 1000.0
    plt.xlim(x_min, x_max)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_list(word_list: list):
    global word_history, word_scores_list
    current_chunk_scores = []
    for word in word_list:
        word_score = ask_word(word)
        rank = wordlist[word]
        word_scores_list[rank] = float(word_score)
        current_chunk_scores.append(word_score)
        word_history.add(word)
    return float(np.mean(current_chunk_scores)) if current_chunk_scores else 0.0


def list_words_within_span(span: tuple, interval: int):
    global word_history
    words_in_span = []
    lo, hi = span
    current_index = max(1, int(lo))
    step = max(1, int(interval))

    while current_index <= hi:
        pending_word = get_word_at_index(current_index)
        if pending_word is None:
            break
        while pending_word in word_history:
            rank = wordlist[pending_word]
            next_rank = rank + 1
            if next_rank > hi or next_rank > len(ranked_words):
                return words_in_span
            pending_word = ranked_words[next_rank - 1]
        words_in_span.append(pending_word)
        current_index += step
    return words_in_span


def output_falloff_number(lowess_output):
    for r, p in reversed(lowess_output):
        if p >= 0.5: ### If the testee's estimated knowledge of the vocabulary in this area is <50%
            return r
    return None


def find_falloff():
    global word_history, known_words

    chunk_scores = {}
    chunk_list = [(1, 1000)]
    current_working_range = chunk_list[-1]

    # First pass: coarse scan by ~10 samples per 1k
    list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 10)
    current_chunk_mean = test_list(list_to_test)
    chunk_scores[current_working_range] = current_chunk_mean

    # Expand outward until performance drops below 0.25
    while chunk_scores[chunk_list[-1]] >= 0.25:
        prev = chunk_scores[chunk_list[-1]]
        last_hi = chunk_list[-1][1]
        chunk_list.append((last_hi + 1, last_hi + 1000))
        current_working_range = chunk_list[-1]
        width = current_working_range[1] - current_working_range[0]
        if prev >= 0.95:
            step = width // 3
        elif prev >= 0.8:
            step = width // 5
        elif prev <= 0.4:
            step = width // 15
        else:
            step = width // 10
        list_to_test = list_words_within_span(current_working_range, step)
        current_chunk_mean = test_list(list_to_test)
        chunk_scores[current_working_range] = current_chunk_mean

    # One more beyond the drop
    last_hi = chunk_list[-1][1]
    chunk_list.append((last_hi + 1, last_hi + 1000))
    current_working_range = chunk_list[-1]
    list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 10)
    current_chunk_mean = test_list(list_to_test)
    chunk_scores[current_working_range] = current_chunk_mean

    # Pick median-chunk w.r.t. 0.5
    median_chunk = None
    median_chunk_score = min(chunk_scores.values(), key=lambda x: abs(x - 0.5))
    for key, value in chunk_scores.items():
        if value == median_chunk_score:
            median_chunk = key
            break

    # refine around that chunk (and neighbors)
    refined = []
    if median_chunk and median_chunk != chunk_list[0] and median_chunk != chunk_list[-1]:
        refined = [(median_chunk[0] - 1000, median_chunk[0] - 1), median_chunk, (median_chunk[1] + 1, median_chunk[1] + 1000)]
    elif median_chunk == chunk_list[0] and median_chunk != chunk_list[-1]:
        refined = [median_chunk, (median_chunk[1] + 1, median_chunk[1] + 1000)]
    elif median_chunk:
        refined = [(median_chunk[0] - 1000, median_chunk[0] - 1), median_chunk]

    for chunk in refined:
        current_working_range = chunk
        list_to_test = list_words_within_span(current_working_range, (current_working_range[1] - current_working_range[0]) // 35)
        current_chunk_mean = test_list(list_to_test)
        chunk_scores[current_working_range] = current_chunk_mean

    # compute final falloff from smoothed full history
    smoothed = sort_and_smooth_data(word_scores_list)
    known_words = output_falloff_number(smoothed)


def plot_graph():
    smoothed = sort_and_smooth_data(word_scores_list)
    fall = output_falloff_number(smoothed)
    plot_data_fitted(smoothed, word_scores_list, fall)


def _map_by_thresholds(threshold_map: dict, value: int | float):
    keys = sorted(threshold_map.keys())
    idx = bisect_right(keys, value) - 1
    if idx < 0:
        return threshold_map[keys[0]]
    return threshold_map[keys[idx]]


def find_cefr(word_count):
    return _map_by_thresholds(word_count_cefr_levels, int(word_count))


def find_label(word_count):
    return _map_by_thresholds(word_count_proficiency_labels, int(word_count))


if __name__ == "__main__":
    find_falloff()
    if known_words is None:
        print("Estimated Level: (insufficient data)")
        print("Try more items at the top of the frequency list; we'll re-estimate your level.")
    else:
        print("Estimated Level:", find_cefr(known_words))
        print(find_label(known_words))
    plot_graph()
