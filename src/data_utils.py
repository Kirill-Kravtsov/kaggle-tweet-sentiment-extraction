import numpy as np


def get_word_offsets(text):
    """
    With inclusive ranges
    """
    words = text.split()
    index = text.index
    offsets = []
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = len(word)
        running_offset = word_offset + word_len
        offsets.append((word, word_offset, running_offset - 1))
    return offsets


def token2word_prob(tweet, token_probs, offsets, agg="max"):
    """
    Note: offest[1] from tokenizers is exclusive
    """
    word_offsets = get_word_offsets(tweet)
    word_probs = []

    for word, word_from, word_to in word_offsets:
        word_probs.append([])
        for (off_from, off_to), prob in zip(offsets, token_probs):
            if (off_from >= word_from) and (off_from <= word_to):
                word_probs[-1].append(prob)
    if agg == "max":
        return list(map(np.max, word_probs))
    elif agg == "mean":
        return list(map(np.average, word_probs))


def find_substr(text, substr):
    """
    Returns inclusive range
    """
    idx0 = None
    idx1 = None
    len_st = len(substr)

    for ind in (i for i, e in enumerate(text) if e == substr[0]):
        if text[ind: ind+len_st] == substr:
            idx0 = ind
            idx1 = ind + len_st - 1
            break
    return idx0, idx1
