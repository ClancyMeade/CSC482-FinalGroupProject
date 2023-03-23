import re 
import nltk
import get_data
import embeddings
from nltk.corpus import stopwords 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten 
from keras.layers import Embedding 
from sklearn.model_selection import KFold
from numpy import array 
import numpy as np

# Number of dimensions in embeddings 
VECTOR_LENGTH = 100 

class SentimentAnalizer: 
    def __init__(self, brand: str): 
        self.brand = brand # the brand name 
        self.word_indexes = {} # Unique indexes of each word: key = word, val = unique index
        self.embeddings = {} # Embeddings of each word: key = word, val = vector 
        self.index = 0 # Counter used to create unique indexes for words 
        self.max_seq_len = 0 # Maximum length of a tweet in the training set 
        self.NN = None

    # Returns cleaned text 
    def get_cleaned_tweet(self, text: str) -> str: 
        # Remove urls 
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove escapes
        text = re.sub(r'\\/', '', text)
        # Strip quotes 
        text = re.sub(r'"', '', text)
        # Remove emojis 
        emoji_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        # Remove punctuation and numbers 
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    
    # Input: data, a list of labeled tweets [[tweet, label], ...]        
    # Output: data in same format, but tweets are tokenized: [[[tokens], label], ...]
    # While tokenizing, also: 
    #     - Stores unique words and their encoding index in self.word_indexes 
    #     - Keeps track of the longest tweet and stores in self.max_seq_len
    def get_tokens(self, data: list, training: bool) -> list: 
        tokenized_data = []
        for entry in data: 
            # Look at nonempy tweets
            if len(entry) > 1:
                tweet = entry[0]
                label = entry[1]
                tokens = []
                # Tokenize tweet
                for word in nltk.word_tokenize(tweet): 
                    to_add = word.lower() # Lowercase 
                    if to_add not in stopwords.words(): # Remove stopwords 
                        tokens.append(to_add) 
                        # Keep track of unique words as we go so we don't have to do it later 
                        if training and to_add not in self.word_indexes:
                            self.word_indexes[to_add] = self.index
                            self.index += 1 # Unique index of unique word
                tokenized_data.append([tokens, label])
                # Keep track of the length of the longest tweet
                if training: 
                    self.max_seq_len = max(self.max_seq_len, len(tokens))
        return tokenized_data
    
    
    # Returns a padded sequence (since the NN needs the len of all tweets to be the same) 
    # Pads the encoded tweets (just their indexes) with zeroes 
    def get_padded_sequence(self, sequence: list, max_len: int) -> list:
        padded_sequence = sequence
        num_to_pad = max_len - len(sequence)
        for i in range(num_to_pad): 
            padded_sequence.append(0)
        return padded_sequence
    
    # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
    # Return padded data and their corresponding labels
    def get_padded_data(self, tokenized_tweets: list, training: bool):
        padded_data = []
        labels = []
        for training_tweet in tokenized_tweets:
            tweet_words = training_tweet[0]
            tweet_label = training_tweet[1]
            index_sequence = []
            # Convert sequence of tokens to its encoded sequence of number (unique indexes stored in dict)
            for word in tweet_words:
                if training or word in self.word_indexes:
                    word_index = self.word_indexes[word]
                    index_sequence.append(word_index)
            # no words overlap with training data
            if (not training) and len(index_sequence) == 0:
                print(tweet_words)
                continue
            # Pad the sequence of numbers
            padded_index_sequence = self.get_padded_sequence(
                index_sequence, self.max_seq_len)
            padded_data.append(padded_index_sequence)
            # Convert labels to numbers
            if tweet_label == "+":
                labels.append(1)
            elif tweet_label == "-":
                labels.append(2)
            else:
                labels.append(0)

        return padded_data, labels
    
    # Trains a Convolutional Neural Network on tweets about the brand 
    def train(self): 
        print()
        print(
            '------------------------------------------------------------------------')
        print('Brand: ' + self.brand.upper())
        print(
            '------------------------------------------------------------------------')
        print()

        # Get training and testing data and clean all the tweets
        training_data = get_data.get_training_data(self.brand)
        testing_data = get_data.get_testing_data(self.brand)
        for tweet in training_data: 
            tweet[0] = self.get_cleaned_tweet(tweet[0])
        for tweet in testing_data: 
            tweet[0] = self.get_cleaned_tweet(tweet[0])            

        # Tokenize the training tweets and get their embeddings 
        tokenized_training_tweets = self.get_tokens(training_data, training=True)
        self.embeddings = embeddings.get_embedding_dict(
            self.word_indexes, 
            vector_length=VECTOR_LENGTH
        ) 
        
        # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
        # and store them in padded_training_data. Add their corresponding labels to training_labels array 
        padded_training_data, training_labels = self.get_padded_data(tokenized_training_tweets, training=True)
        
        # Get embedding matrix 
        total_words = len(self.word_indexes)
        embedding_matrix = np.zeros((total_words, VECTOR_LENGTH)) # Creates matrix of zeros of size total_words x vectorlen
        
        # Store the embeddings for each word at their corresponding index in the embedding_matrix  
        # This matrix is used as initial weights in the NN 
        # The indexes of each vector correspond to the sequence of numbers (corresponding to words) in padded_training_data
        for word, index in self.word_indexes.items(): 
            embedding_vector = self.embeddings[word]
            embedding_matrix[index] = embedding_vector
        
        # Build the model: Convolutional Neural Net 
        training_labels = np.array(training_labels)
        padded_training_data = np.array(padded_training_data)


        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=10, shuffle=True)

        fold_idx = 1
        for train, test in kfold.split(padded_training_data, training_labels):
            model = Sequential()
            # Embedding layer -> word embeddings are initial weights
            embedding_layer = Embedding(total_words, VECTOR_LENGTH,
                                        weights=[embedding_matrix],
                                        input_length=self.max_seq_len)
            model.add(embedding_layer)
            model.add(Conv1D(VECTOR_LENGTH, 5, activation='relu'))
            model.add(MaxPooling1D(5))
            model.add(Flatten())
            model.add(Dense(units=VECTOR_LENGTH, activation='relu'))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam',
                        loss='binary_crossentropy', metrics=['accuracy'])
            # print(model.summary())
            model.fit(padded_training_data[train], training_labels[train], epochs=4, verbose=0)
            # self.NN = model

            loss, accuracy = model.evaluate(
                padded_training_data[test], training_labels[test], verbose=0)
            
            acc_per_fold.append(accuracy * 100)
            loss_per_fold.append(loss)

            print(
                '------------------------------------------------------------------------')
            print('FOLD #' + str(fold_idx) + ':')
            print('Accuracy: ' + str(acc_per_fold[fold_idx - 1]) + '%')
            print('Loss: ' + str(loss_per_fold[fold_idx - 1]))

            fold_idx += 1

        print('------------------------------------------------------------------------')
        print('AVERAGE SCORES:')
        print('Accuracy: ' + str(np.mean(acc_per_fold)) + '%')
        print('Loss: ' + str(np.mean(loss_per_fold)))
        print('------------------------------------------------------------------------')

        

        # Tokenize the testing tweets
        tokenized_testing_tweets = self.get_tokens(testing_data, training=False)

        # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
        # and store them in padded_testing_data. Add their corresponding labels to testing_labels array
        padded_testing_data, testing_labels = self.get_padded_data(tokenized_testing_tweets, training = False)

        loss, accuracy = model.evaluate(
            padded_testing_data, testing_labels, verbose=0)
        print(
            '------------------------------------------------------------------------')
        print('Testing Data:')
        print('Accuracy: ' + str(accuracy * 100) + '%')
        print('Loss: ' + str(loss))
        
        # OTHER TO DO ****
        # Try tuning parameters (maybe): 
            # number of layers 
            # activation functions 
        # Try other types of networks: 
            # Feed Forward NN (the code will be different though, so maybe we just use this one)
        # Create functions to use the models to classify tweets 
        # Create functions to store the trained models to files and functions to restore them 
        
        # NOTE: Make sure to dowload the embeddings before or this won't run 
        # To download them, run: python3 embeddings.py (the get_embeddings() function will download them into a text file 
        # It might take a while since the file is 1 gig) 
        # Also, changing the number of epochs will increase the accuracy, but I think it will lead to overfitting if we do too many 
            
        
        
        
                

def main():
    nvidia_sa = SentimentAnalizer("nvidia")
    nvidia_sa.train()
    microsoft_sa = SentimentAnalizer("microsoft")
    microsoft_sa.train()

if __name__ == "__main__": 
    main()
