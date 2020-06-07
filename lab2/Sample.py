from utils import Labels, WordList


class Sample:
    def __init__(self, words: WordList, label: Labels):
        self.words = words
        self.label = label
