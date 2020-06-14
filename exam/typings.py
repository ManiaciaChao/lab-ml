from typing import Dict, List

Labels = Dict[str, str]
WordList = List[str]


class Sample:
    def __init__(self, feature: WordList, label: str):
        self.feature = feature
        self.label = label
