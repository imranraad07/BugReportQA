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
    modified_text = re.sub(r'```.+```', '', text)
    modified_text = re.sub(r'```.+', '', modified_text)
    return modified_text

def filter_nontext(text):
    text = remove_block_quotes(text)
    text = remove_newlines(text)
    text = remove_triple_quotes(text)
    return text