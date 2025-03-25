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
        This function also returns characterLines"""
        # Define the word range (default max is 9999 because it will never exceed that)
        wordRange = (0 if not minWords else minWords, 9999 if not maxWords else maxWords)

        self.characterLines = defaultdict(list)
        with open(filePath, newline='', encoding="utf8") as file:
            for row in csv.reader(file, delimiter=","):
                # If the character is a main character, add their line to their respective list of lines
                if row[14] == "main":
                    character = row[0]
                    line = row[1]

                    # Make sure the line is within the valid range
                    numWords = len(line.split())
                    if numWords >= wordRange[0] and numWords <= wordRange[1]:
                        self.characterLines[character].append(line)

        # Make all characters have the same number of lines if the user chooses
        if normalizeLens:
            minLen = min([len(lines) for lines in self.characterLines.values()])
            for character in self.characterLines.keys():
                 shuffle(self.characterLines[character])
                 self.characterLines[character]= self.characterLines[character][:minLen]

        return self.characterLines

    def trainTestSplit(self, testSize: float=.25, strictTestSize: int=None, minWords: int=None) -> tuple[dict[str, list[str]]]:
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

            # Get all lines with valid length and get subset of it
            # Set train equal to the rest of the valid lengths and the invalid lengths
            if minWords:
                validTestLines = [line for line in shuffledLines if len(line.split()) >= minWords]
                invalidTestLines = [line for line in shuffledLines if len(line.split()) < minWords]
                splitPoint = len(validTestLines) - strictTestSize
                train[character] = validTestLines[:splitPoint]
                test[character] = validTestLines[splitPoint:]
                train[character].extend(invalidTestLines)
            # If minWords not specefied then just split data as normal
            else:
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
                # Good for debugging / seeing what's going on (probabilities changing):
                #print(f"{word}: {tokenizedLines[character][word] + 1} | New probability = {probabilies[character]}")

        return probabilies

    def classifyLine(self, train: dict[str, list[str]], line: str, wordCounts) -> str:
        """Using the helper functions, take in a line and classify it as a specific character"""
        return max(self.getProbabilities(train, line, wordCounts).items(), key=lambda pair: pair[1])

    def scoreCharacter(self, character: str, train: dict[str, list[str]], test: dict[str, list[str]], wordCounts) -> float:
        """This will score how well the model does with a specific character"""
        predictions = defaultdict(int)
        for i, line in enumerate(test[character]):
            predictions[self.classifyLine(train, line, wordCounts)[0]] += 1

        return predictions[character]/len(test[character]), dict(predictions)


def runFrasierNB(minWordsTest: int=None, minWordsAll: int=None, strictTestSize: int=None, printLinesPerCharacter: bool=False):
    """This function is used to run a test, it returns nothing but prints out each character's score"""
    frasierNBModel = NaiveBayes()
    frasierNBModel.getData(filePath="./data/cleanedTranscript.csv", minWords=minWordsAll)
    train, test = frasierNBModel.trainTestSplit(testSize=.25, strictTestSize=strictTestSize, minWords=minWordsTest)

    # Get number of lines per character
    if printLinesPerCharacter:
        for i in frasierNBModel.characterLines.keys():
            print(f"{i}: {len(frasierNBModel.characterLines[i])}", end=" ")
            print(f"Train: {len(train[i])} | Test: {len(test[i])}")

    characterConfusionMatrix = dict()
    # Print out each character's score
    wordCounts = frasierNBModel.getWordCounts(train)
    for character in train.keys():
        score, predictions = frasierNBModel.scoreCharacter(character, train, test, wordCounts)
        characterConfusionMatrix[character] = predictions
        print(f"{character}: {score}")
    
    print(characterConfusionMatrix)
    return characterConfusionMatrix

if __name__ == "__main__":
    # Run through multiple tests using different minWord values
    for minWords in [0, 5, 10, 15, 20, 25]:
        print(f"Testing with lines that have a minimum of {minWords} words:")
        runFrasierNB(strictTestSize=250, minWordsTest=minWords, printLinesPerCharacter=False)
        print()
