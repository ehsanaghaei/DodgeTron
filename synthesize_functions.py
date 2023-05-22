import time

from elephant.spade import spade
# from gsp import GSP
from prefixspan import PrefixSpan

from collections import Counter


def find_common_apis(dictionary, N):
    word_count = Counter()
    word_keys = {}
    word_lists = {}

    # Count the number of occurrences of each word and track the keys and lists that include them
    for key, sub_dict in dictionary.items():
        for list_id, words_list in sub_dict.items():
            word_count.update(words_list)
            for word in set(words_list):
                if word in word_keys:
                    word_keys[word].add(key)
                else:
                    word_keys[word] = {key}
                if word in word_lists:
                    word_lists[word].add(list_id)
                else:
                    word_lists[word] = {list_id}

    # Filter words that appear in more than one key
    common_words = {word: {
        "number of times this word is appeared totally": count,
        "number of keys that include this word": len(keys),
        "number of lists containing this word": len(lists),
        "keys that contain this word": keys,
    } for word, count in word_count.items() for keys in word_keys.values() if len(keys) > 1 and (lists := word_lists.get(word, set()))}

    # Sort the common words by the number of times they appear in all keys
    sorted_common_words = dict(sorted(common_words.items(), key=lambda item: item[1]["number of keys that include this word"], reverse=True))
    # sorted_common_words = dict(sorted(sorted_common_words.items(), key=lambda item: item[1]["number of keys that include this word"], reverse=True))

    # Return the top N common words
    top_common_words = dict(list(sorted_common_words.items())[:N])

    return top_common_words

    # Example usage


my_dictionary = {
    "key1": {
        "id11": ["word1", "word2", "word3"],
        "id12": ["word4", "word5", "word6"]
    },
    "key2": {
        "id21": ["word3", "word4", "word5"],
        "id22": ["word1", "word2", "word7"]
    }
}

top_N = 3
result = find_common_apis(my_dictionary, top_N)
print(result)


def call_indexer(representations):
    calls = []
    for family in representations:
        for id_ in representations[family]:
            seq = representations[family][id_]
            calls += seq
            calls = list(set(calls))
    call2idx = {call: i for i, call in enumerate(calls)}
    udx2call = {i: call for i, call in enumerate(calls)}
    return call2idx, udx2call


def encode_sequence(call2idx, sequence):  # -> returns the patterns for a list of calls
    return [call2idx[call] for call in sequence]


def batch_encode_sequence(call2idx, sequences):
    return [encode_sequence(call2idx, sequence) for sequence in sequences]


def pattern_extractor(encoded_sequences, method):
    min_sup = 2
    max_pat = 10
    min_len = 30
    max_len = 150
    if method == "PrefixSpan":
        start_time = time.time()
        ps = PrefixSpan(encoded_sequences)
        patterns_ps = ps.frequent(min_sup, max_len=max_len, min_len=min_len)
        end_time = time.time()
        print("PrefixSpan found", len(patterns_ps), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_ps
    elif method == "GSP":
        start_time = time.time()
        gsp = GSP(encoded_sequences)
        patterns_gsp = gsp.run(min_sup=min_sup, max_pattern=max_pat, min_len=min_len, max_len=max_len)
        end_time = time.time()
        print("GSP found", len(patterns_gsp), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_gsp
    elif method == "Spade":
        start_time = time.time()
        spade1 = spade(encoded_sequences)
        patterns_spade = spade1.run_sup(max_sup=min_sup, max_pattern=max_pat, min_len=min_len, max_len=max_len)
        end_time = time.time()
        print("SPADE found", len(patterns_spade), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_spade

    else:
        print('The method has not been recognized. Use "GSP", "PrefixSpan", or "Spade"')


def synthesizer(representations, method):
    call2idx, udx2call = call_indexer(representations)
    patterns = {}
    for family in representations:
        seqs = [representations[family][id_] for id_ in representations[family]]
        seqs_encoded = [encode_sequence(call2idx, sequence) for sequence in seqs]
        patterns[family] = pattern_extractor(seqs_encoded, method)
    return patterns

# p = synthesizer(representations, 'PrefixSpan')
