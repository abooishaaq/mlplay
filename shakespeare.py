
def get_data():
    file = open("./data/shakespeare.txt", "r")
    data = file.read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    file.close()
    return data, chars, data_size, vocab_size, char_to_ix, ix_to_char
