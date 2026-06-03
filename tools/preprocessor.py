# import os
# os.environ['GENSIM_DATA_DIR'] = '/home/isel-har/goinfre/gensim'
import nltk
from transformers import AutoTokenizer
import contractions
import string
from nltk.stem     import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.corpus   import wordnet
import numpy as np
import gensim.downloader as api
# import re


# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("stopwords")


class NLProcessor:
    punct = string.punctuation.replace('-', '')
    punct_translator = str.maketrans('', '', punct)
    digit_translator = str.maketrans("", "", string.digits)

    def __init__(
        self,
        use_stopwords=False,
        normalize=True,
        lower=True,
        use_clean=True,
        remove_punc=True,
        sub_word_tokenizer=False,
        embedder=api.load("fasttext-wiki-news-subwords-300")
    ):
        self.use_clean     = use_clean
        self.use_stopwords = use_stopwords
        self.use_normalize = normalize
        self.padding_len   = 0
        self.unfound_words = []
        self.remove_punc   = remove_punc
        self.lower         = lower

        self.stop_words = set(stopwords.words("english")) if use_stopwords else None
        self.normalizer = WordNetLemmatizer() if normalize else None
        self.sub_word_tokenizer = sub_word_tokenizer
        self.embedder = embedder


    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def clean(self, sentences):
        cleaned_tweets = []

        for sentence in sentences:
            sentence = contractions.fix(sentence)
            if self.lower:
                sentence = sentence.lower()
            if self.remove_punc:
                sentence = sentence.translate(self.punct_translator)
            sentence = sentence.translate(self.digit_translator)
            sentence = sentence.strip() # remove leading/trailing spaces
            sentence = " ".join(sentence.split()) # remove duplicate spaces

            cleaned_tweets.append(sentence)

        return cleaned_tweets
    

    def tokenization(self, sentences):
        tokens_list = list()

        if self.sub_word_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
        else:
            for text in sentences:
                tokens = word_tokenize(text)
                tokens_list.append(tokens)

        return tokens_list


    def filter_stopwords(self, tokens_list):
        filtered_tokens = []

        for tokens in tokens_list:
            filtered = [word for word in tokens if word not in self.stop_words]
            filtered_tokens.append(filtered)

        return filtered_tokens

    # @staticmethod
    # def remove_ed_(word):
    #     return re.sub(r'ed$', '', word)

    # def remove_ed_suf(self, tokens_list: list):
    #     stems_tokens = []

    #     for tokens in tokens_list:
    #         stems_tokens.append([self.remove_ed_(token) for token in tokens])

    #     return stems_tokens


    def normalize(self, tokens_list):

        normalized_tokens_list = list()

        for tokens in tokens_list:
            lemmas = list()
            pos_tags = nltk.pos_tag(tokens)
            for word, tag in pos_tags:
                wn_tag = self.get_wordnet_pos(tag)
                lemma = self.normalizer.lemmatize(word, wn_tag)
                lemmas.append(lemma)
    
            normalized_tokens_list.append(lemmas)

        return normalized_tokens_list


    def add_padding(self, embedded_tokens):
        
        padding_vec = np.zeros(embedded_tokens[0][0].shape[0])
        padded_list = list()
        for embedded_token in  embedded_tokens:
  
            for _ in range(len(embedded_token), self.padding_len):
                embedded_token.append(padding_vec)

            padded_list.append(np.array(embedded_token))

        return np.array(padded_list)



    def word2vec_embedding(self, tokens_list):
        embedded_tokens = list()

        for i, tokens in enumerate(iterable=tokens_list):
            
            embeddings = list()
            for token in tokens:

                if token in self.embedder:
                    embedding_vec = self.embedder[token]
                    embeddings.append(embedding_vec)
                else:
                    print(f"token not found : {token}")

            if len(embeddings) > self.padding_len:
                self.padding_len = len(embeddings)


            if len(embeddings) > 0:
                embedded_tokens.append(embeddings)
            else:
                self.unfound_words.append(i)
        return embedded_tokens



    def transform(self, raw_sentences):

        sentences = raw_sentences
        if self.use_clean:
            sentences = self.clean(sentences)

        tokens  = self.tokenization(sentences)
        if self.use_stopwords:
            tokens  = self.filter_stopwords(tokens)
        if self.use_normalize:
            tokens  = self.normalize(tokens)

        # if self.remove_ed:
        #     tokens = self.remove_ed_suf(tokens)

        if self.sub_word_tokenizer:
            return tokens
    

        embedded_tokens = self.word2vec_embedding(tokens)
        return self.add_padding(embedded_tokens)
