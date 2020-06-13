from nltk.translate.bleu_score import corpus_bleu


def compute(ref_fpath, q_fpath, ref_no):
    refs = extract_ref(ref_fpath, ref_no)
    data = extract_data(q_fpath)

    assert len(refs) == len(data)
    return corpus_bleu(refs, data)


def extract_ref(fpath, ref_no):
    ref_corpus = list()
    with open(fpath, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            refs = list()
            for i in range(ref_no):
                tokens = tokenize(lines[idx])
                refs.append(tokens)
                idx += 1
            ref_corpus.append(refs)
    return ref_corpus


def extract_data(fpath):
    corpus = list()
    with open(fpath, 'r') as f:
        for line in f.readlines():
            tokens = tokenize(line)
            corpus.append(tokens)
    return corpus


def tokenize(text):
    line = str(text).lower().strip()
    line = line.replace('<unk>', ' ').replace('<EOS>', ' ')
    # punctuation tokenizing
    line_filtered = ''
    for c in line:
        if c in '!@$%^&*()[]{};:,\'.<>/?\|`~-=+':
            line_filtered += ' ' + c + ' '
        else:
            line_filtered += c

    return line_filtered.split(' ')
