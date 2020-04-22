import re


def modify_to_remove_code(text):
    # remove new lines with space
    modified_text = text.replace("\n", " ")
    # remove codes
    modified_text = re.sub(r'```.+```', '', modified_text)
    return modified_text
