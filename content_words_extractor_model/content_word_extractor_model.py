import nltk
import contractions

# this class will be used to extract content words from a given text
# it will get part of the speech of each word and then filter out the words
# that are not content words, which are those that are not nouns, verbs or adjectives
class ContentWordsExtractor:
    def __init__(self):
        pass

    def get_content_words(self, text):
        # expand constractions
        text = contractions.fix(text)
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
