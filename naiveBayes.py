# Frasier Naive Bayes
# naiveBayes.py
# Mason Lee

import csv
from collections import defaultdict
from random import shuffle

class NaiveBayes:
    def __init__(self):
        pass

    def getData(self, filePath: str, normalizeLens: bool=False, minWords: int=None, maxWords: int=None) -> dict[str, list[str]]:
        """Sets self.characterLines equal to data from specified file in format: 
        {characterName: [allCharacter lines], ... }
        This function should also return characterLines"""
        # Define the word range
        wordRange = (0 if not minWords else minWords, 9999 if not maxWords else maxWords)

        self.characterLines = defaultdict(list)
        with open(filePath, newline='', encoding="utf8") as file:
            # Loop through each csv row
            for row in csv.reader(file, delimiter=","):
                # If the character is a main character, add their line to their respective list of lines
                if row[14] == "main":
                    character = row[0]
                    line = row[1]

                    # Account for minWords and maxWords parameters
                    if len(line.split()) >= wordRange[0] and len(line.split()) <= wordRange[1]:
                        self.characterLines[character].append(line)

        # Make all characters have the same number of lines
        if normalizeLens:
            minLen = min([len(lines) for lines in self.characterLines.values()])
            for character in self.characterLines.keys():
                 shuffle(self.characterLines[character])
                 self.characterLines[character]= self.characterLines[character][:minLen]

        return self.characterLines

    def trainTestSplit(self, testSize: float=.25, strictTestSize: int=None) -> tuple[dict[str, list[str]]]:
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
            if strictTestSize:
                splitPoint = len(shuffledLines) - strictTestSize
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

    def getProbabilities(self, train: dict[str, list[str]], line: str, tokenizedLines=None) -> dict[str, float]:
        """This returns a dictionary of the probability for each character"""
        probabilies = defaultdict(float)
        totalNumLines = sum([len(lines) for lines in train.values()]) # Total number of lines in trianing set

        # Loop through each character
        for character in train.keys():
            # Get the total number of words they spoke
            totalNumWords = sum([len(charLine.split()) for charLine in train[character]])
            numCharLines = len(train[character])
            probabilies[character] = numCharLines / totalNumLines # Initial guess (initial probability)
            
            # For each word in the line, get the probability of that word being spoken by the character
            # Multiply the current probability by the new word's probability (updating our beliefs based on new data) 
            for word in line.split():
                probabilies[character] *= ((tokenizedLines[character][word] + 1) / totalNumWords)
                # Good for debugging / seeing what's going on:
                #print(f"{word}: {tokenizedLines[character][word] + 1} | New probability = {probabilies[character]}")

        return probabilies

    def classifyLine(self, train: dict[str, list[str]], line: str, wordCounts) -> str:
        """Using the helper functions, this will take in a line and classify it as a specific character"""
        return max(self.getProbabilities(train, line, wordCounts).items(), key=lambda pair: pair[1])

    def scoreCharacter(self, character: str, train: dict[str, list[str]], test: dict[str, list[str]], wordCounts) -> float:
        """This will score how well the model does with a specific character"""
        numRight = 0
        for i, line in enumerate(test[character]):
            if self.classifyLine(train, line, wordCounts)[0] == character:
                numRight += 1
                # print(f"{i}: {numRight}") # Good for checking progress

        return numRight/len(test[character])


def runFrasierNB(minWordsTest: int=None, minWordsAll: int=None, strictTestSize: int=None, printLinesPerCharacter: bool=False):
    """This function is used to run a test, it returns nothing but prints out each character's score"""
    frasierNBModel = NaiveBayes()
    frasierNBModel.getData(filePath="./data/cleanedTranscript.csv", minWords=minWordsAll)
    train, test = frasierNBModel.trainTestSplit(testSize=.25, strictTestSize=strictTestSize)

    if minWordsTest:
        # Remove lines less than or equal to minWordsTest
        for character in test.keys():
            test[character] = [line for line in test[character] if len(line.split()) >= minWordsTest]

    # Get number of lines per character
    if printLinesPerCharacter:
        for i in frasierNBModel.characterLines.keys():
            print(f"{i}: {len(frasierNBModel.characterLines[i])}", end=" ")
            print(f"Train: {len(train[i])} | Test: {len(test[i])}")

    wordCounts = frasierNBModel.getWordCounts(train)
    for character in train.keys():
        print(f"{character} score: {frasierNBModel.scoreCharacter(character, train, test, wordCounts)}")

if __name__ == "__main__":
    runFrasierNB(strictTestSize=500, printLinesPerCharacter=True, minWordsTest=10)
