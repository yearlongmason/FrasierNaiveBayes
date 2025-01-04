# Frasier Naive Bayes
# naiveBayes.py
# Mason Lee

import csv
from collections import defaultdict
from random import shuffle

class NaiveBayes:
    def __init__(self):
        pass

    def getData(self, filePath: str) -> dict[str, list[str]]:
        """Sets self.characterLines equal to data from specified file in format: 
        {characterName: [allCharacter lines], ... }
        This function should also return characterLines"""
        self.characterLines = defaultdict(list)
        with open(filePath, newline='', encoding="utf8") as file:
            # Loop through each csv row
            for row in csv.reader(file, delimiter=","):
                # If the character is a main character, add their line to their respective list of lines
                if row[14] == "main":
                    character = row[0]
                    line = row[1]
                    self.characterLines[character].append(line)

        return self.characterLines

    def trainTestSplit(self, testSize: int=.25) -> tuple[dict[str, list[str]]]:
        """This function splits and returns data into testing and training sets"""
        # Creates two copies of the original character lines dictionary
        train = defaultdict(list)
        test = defaultdict(list)
        # Fill each dictionary with the respective amount of lines based on the testSize parameter
        for character in self.characterLines.keys():
            shuffledLines = self.characterLines[character].copy()
            shuffle(shuffledLines)
            # Get the split point (amount of data we're taking for the training data)
            splitPoint = int(len(shuffledLines) * (1 - testSize))
            
            # Set the training and testing lines for the character
            train[character] = shuffledLines[:splitPoint]
            test[character] = shuffledLines[splitPoint:]

        return train, test

    def getWordCounts(self, data) -> dict[str, dict[str, int]]:
        """This function is essentially supposed to be a tokenization function
        Returns a dictionary with each character's wordCounts"""
        tokenCounts = defaultdict(lambda: defaultdict(int))
        for character in data.keys():
            for line in data[character]:
                for word in line.split(" "):
                    tokenCounts[character][word] += 1
            
        return tokenCounts

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
    train, test = frasierNBModel.trainTestSplit()