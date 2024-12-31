# Frasier Naive Bayes
# naiveBayes.py
# Mason Lee

import csv
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        pass

    def getData(self, filePath: str) -> dict[str, list[str]]:
        """Returns data from specified fileName in format {characterName: [allCharacter lines], ... }"""
        with open(filePath, newline='', encoding="utf8") as file:
            for row in csv.reader(file, delimiter=","):
                if row[14] == "main":
                    print(row[0], row[1]) # This is just a test
                    break
                    # Use default dict to add line to character's lines
                

    def getWordCounts() -> dict[str, dict[str, int]]:
        """This function is essentially supposed to be a tokenization function
        Returns a dictionary with each character's wordCounts"""
        pass

    def trainTestSplit():
        """This function splits data into testing and training sets"""
        pass

    def getProbabilities(line: str) -> dict[str, float]:
        """This returns a dictionary of the probability for each character"""
        pass

    def classifyLine(line: str) -> str:
        """Using the helper functions, this will take in a line and classify it as a specific character"""
        pass

    def scoreCharacter(character: str) -> float:
        """This will score how well the model does with a specific character"""
        pass


if __name__ == "__main__":
    # getData file path: "./data/cleanedTranscript.csv"
    frasierNBModel = NaiveBayes()
    frasierNBModel.getData(filePath="./data/cleanedTranscript.csv")