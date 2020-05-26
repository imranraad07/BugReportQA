import re


def remove_block_quotes(text):
    modified_text = ''
    for line in text.split('\n'):
        if not line.strip().startswith('>'):
            modified_text += line + '\n'
    return modified_text


def remove_newlines(text):
    # replace new lines with space
    modified_text = text.replace("\n", " ")
    return modified_text


def remove_triple_quotes(text):
    occurrences = [m.start() for m in re.finditer('```', text)]
    idx = len(occurrences)
    for i in range(0, len(occurrences), 2):
        if idx > 0:
            text = text[:occurrences[idx - 2]] + text[occurrences[idx - 1]:]
            idx = idx - 2
    text = text.replace("`", "")
    return text


def remove_stacktrace(text):
    st_regex = re.compile('at [a-zA-Z0-9\.<>$]+\(.+\)')
    lines = list()
    for line in text.split('\n'):
        matches = st_regex.findall(line.strip())
        if len(matches) == 0:
            lines.append(line)
        else:
            for match in matches:
                line = line.replace(match, ' ')
            lines.append(line.strip(' \t'))

    lines = '\n'.join(lines)
    # hack to get rid of multiple spaces in the text
    lines = ' '.join(lines.split())
    return lines


def filter_nontext(text):
    text = remove_block_quotes(text)
    text = remove_stacktrace(text)
    text = remove_newlines(text)
    text = remove_triple_quotes(text)
    return text
