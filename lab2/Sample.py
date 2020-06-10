from typing import Dict, List

Labels = Dict[str, str]
WordList = List[str]


class Sample:
    def __init__(self, words: WordList, label: str):
        self.words = words
        self.label = label
