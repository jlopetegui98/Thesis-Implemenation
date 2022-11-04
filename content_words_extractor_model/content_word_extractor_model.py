import nltk
import contractions
import re

# this class will be used to extract content words from a given text
# it will get part of the speech of each word and then filter out the words
# that are not content words, which are those that are not nouns, verbs or adjectives
class ContentWordsExtractor:
    def __init__(self):
        pass

    def get_content_words(self, text):
        # expand constractions
        text = contractions.fix(text)
        # to lower case
        text = text.lower()
        text += "$"
        # replace "h" when it is after a number by hour using regex
        text = re.sub(r"(\d)\s*h[^a-zA-Z0-9_]", r"\1 hour ", text)
        # replace "F" when it is after a number by minute using regex
        text = re.sub(r"(\d)\s*F[^a-zA-Z0-9_]", r"\1 fahrenheit ", text)
        # replace "C" when it is after a number by minute using regex
        text = re.sub(r"(\d)\s*C[^a-zA-Z0-9_]", r"\1 celcius ", text)
        # replace "%" when it is after a number by minute using regex
        text = re.sub(r"(\d)\s*\%[^a-zA-Z0-9_]", r"\1 percent ", text)
        # replace special characters by space
        text = re.sub(r"[^.,a-zA-Z0-9 ]", r" ", text)
        # to lower case
        text = text.lower()
        # remove special character from end of sentence
        text = text[:-1]
        # get part of speech of each word
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        # filter out content words
        content_words = list(
            filter(
                lambda x: x[1]
                in [
                    "NN",
                    "NNS",
                    "NNP",
                    "NNPS",
                    "VB",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                    "JJ",
                    "JJR",
                    "JJZ",
                ],
                pos_tags,
            )
        )
        # return content words
        return [word for word, _ in content_words]
