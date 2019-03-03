def load_data(dtype="both", input_length=[20, 24], w2v_length=300):
    def __load_data(dtype="word", input_length=20, w2v_length=300):

        filename = model_dir + "%s_%d_%d" % (dtype, input_length, w2v_length)
        if os.path.exists(filename):
            return pd.read_pickle(filename)

        data_l_n = []
        data_r_n = []
        y = []
        for line in open(train_file, "r", encoding="utf8"):
            lineno, s1, s2, label = line.strip().split("\t")
            if dtype == "word":
                data_l_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s1))) if word in word2index])
                data_r_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s2))) if word in word2index])
            if dtype == "char":
                data_l_n.append([char2index[char] for char in s1 if char in char2index])
                data_r_n.append([char2index[char] for char in s2 if char in char2index])

            y.append(int(label))

        # 对齐语料中句子的长度
        data_l_n = pad_sequences(data_l_n, maxlen=input_length)
        data_r_n = pad_sequences(data_r_n, maxlen=input_length)
        y = np.array(y)

        pd.to_pickle((data_l_n, data_r_n, y), filename)

        return (data_l_n, data_r_n, y)

    if dtype == "both":
        ret_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_l_n, data_r_n, y = __load_data(dtype, input_length, w2v_length)
            ret_array.append(np.asarray(data_l_n))
            ret_array.append(np.asarray(data_r_n))
        ret_array.append(y)
        return ret_array
    else:
        return __load_data(dtype, input_length, w2v_length)


def input_data(sent1, sent2, dtype="both", input_length=[20, 24]):
    def __input_data(sent1, sent2, dtype="word", input_length=20):
        data_l_n = []
        data_r_n = []
        for s1, s2 in zip(sent1, sent2):
            if dtype == "word":
                data_l_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s1))) if word in word2index])
                data_r_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s2))) if word in word2index])
            if dtype == "char":
                data_l_n.append([char2index[char] for char in s1 if char in char2index])
                data_r_n.append([char2index[char] for char in s2 if char in char2index])

        # 对齐语料中句子的长度
        data_l_n = pad_sequences(data_l_n, maxlen=input_length)
        data_r_n = pad_sequences(data_r_n, maxlen=input_length)

        return [data_l_n, data_r_n]

    if dtype == "both":
        ret_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_l_n, data_r_n = __input_data(sent1, sent2, dtype, input_length)
            ret_array.append(data_l_n)
            ret_array.append(data_r_n)
        return ret_array
    else:
        return __input_data(sent1, sent2, dtype, input_length)
