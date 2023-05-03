import time

from elephant.spade import spade
from gsp import GSP
from prefixspan import PrefixSpan


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
        patterns_ps = ps.frequent(min_sup, max_len=max_len,min_len=min_len)
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
