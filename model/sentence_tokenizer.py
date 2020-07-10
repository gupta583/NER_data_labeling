# Each sentences is tokenized and labels are extended for tokenized words.

from transformers import BertTokenizer, BertConfig
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def tokenize_sentences_labels(sentence, sent_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, sent_labels):

        # Tokenizing the words
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        # Tokenizer split the words into n_subwords with preface '##' if word not in dictionary 
        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

