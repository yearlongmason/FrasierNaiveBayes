# Frasier Naive Bayes
# naiveBayes.py
# Mason Lee

class NaiveBayes:
    def __init__(self):
        pass

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
    print("Hello world")