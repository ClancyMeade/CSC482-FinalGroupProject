import re 
import nltk
import get_data
import embeddings
import numpy as np
import sys
from keras.models import Sequential, model_from_json
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Embedding 
from keras.layers import Dropout
from keras.initializers import Constant
from sklearn.model_selection import KFold
from keras import utils
from get_data import get_recent_tweets


VECTOR_LENGTH = 100 # Number of dimensions in embeddings 
POSITIVE_SENTIMENT = 1
NEGATIVE_SENTIMENT = 2
NEUTRAL_SENTIMENT = 0


class SentimentAnalyzer: 
    def __init__(self, brand: str, k_fold: bool):
        self.k_fold = k_fold
        self.brand = brand # the brand name 
        self.word_indexes = {} # Unique indexes of each word: key = word, val = unique index
        self.embeddings = {} # Embeddings of each word: key = word, val = vector 
        self.index = 1 # Counter used to create unique indexes for words 
        self.seq_len = 0 # Num nodes in first layer 
        self.neural_net = None # Trained model 
        self.vocab_size = 0 
        self.count_positive = 0
        self.count_negative = 0 
        self.count_neutral = 0


    # Returns cleaned text 
    def get_cleaned_tweet(self, text: str) -> str: 
        # Remove urls 
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove escapes
        text = re.sub(r'\\/', '', text)
        # Strip quotes 
        text = re.sub(r'"', '', text)
        # Remove emojis 
        #emoji_pattern = re.compile(pattern = "["
           # u"\U0001F600-\U0001F64F"  # emoticons
           # u"\U0001F300-\U0001F5FF"  # symbols & pictographs
           # u"\U0001F680-\U0001F6FF"  # transport & map symbols
           # u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              # "]+", flags = re.UNICODE)
        #text = emoji_pattern.sub(r'', text)
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
                    #if to_add not in stopwords.words(): # Remove stopwords 
                    tokens.append(to_add) 
                    # Keep track of unique words as we go so we don't have to do it later 
                    if training and to_add not in self.word_indexes:
                        self.word_indexes[to_add] = self.index
                        self.index += 1 # Unique index of unique word
                tokenized_data.append([tokens, label])
        return tokenized_data
    
        
    # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
    # Put them all in the same order (so features match up in first layer of NN)
    # Returns the encoded data and its labels 
    def convert_to_sequences(self, tokenized_tweets: list, training: bool):
        sequences = []
        labels = []
        for training_tweet in tokenized_tweets:
            tweet_words = training_tweet[0]
            tweet_label = training_tweet[1]
            index_sequence = np.zeros(self.seq_len)
            # Convert sequence of tokens to its encoded sequence of number (unique indexes stored in dict)
            for word in tweet_words:
                if training or word in self.word_indexes:
                    word_index = self.word_indexes[word]
                    index_sequence[word_index] = word_index
            # no words overlap with training data
            if (not training) and len(index_sequence) == 0:
                #print(tweet_words)
                continue
            # Pad the sequence of numbers
            #padded_index_sequence = self.get_padded_sequence(
                #index_sequence, self.max_seq_len)
            sequences.append(index_sequence)
            # Convert labels to numbers
            if tweet_label == "+":
                labels.append(POSITIVE_SENTIMENT)
                self.count_positive += 1
            elif tweet_label == "-":
                labels.append(NEGATIVE_SENTIMENT)
                self.count_negative += 1
            else:
                labels.append(NEUTRAL_SENTIMENT)
                self.count_neutral += 1

        return sequences, labels
    
        
    # Trains a Feed Forward neural network on tweets about the brand 
    def train(self): 
        print()
        print(
            '------------------------------------------------------------------------')
        print('Training Brand: ' + self.brand.upper())
        print(
            '------------------------------------------------------------------------')
        print()

        # Get training data and clean all the tweets
        training_data = get_data.get_training_data(self.brand)
        for tweet in training_data: 
            tweet[0] = self.get_cleaned_tweet(tweet[0])

        # Tokenize the training tweets and get their embeddings 
        tokenized_training_tweets = self.get_tokens(training_data, training=True)
        self.embeddings = embeddings.get_embedding_dict(
            self.word_indexes, 
            vector_length=VECTOR_LENGTH
        ) 
        
        self.vocab_size = len(self.word_indexes) 
        self.seq_len = self.vocab_size + 1
        # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
        # and store them in sequences_training_data. Add their corresponding labels to training_labels array 
        sequence_training_data, training_labels = self.convert_to_sequences(tokenized_training_tweets, training=True)
        
        
        # Get embedding matrix 
        embedding_matrix = np.zeros((self.vocab_size + 1, VECTOR_LENGTH)) # Creates matrix of zeros of size total_words x vectorlen
        # Store the embeddings for each word at their corresponding index in the embedding_matrix  
        # This matrix is used as initial weights in the NN 
        # The indexes of each vector correspond to the sequence of numbers (corresponding to words) in padded_training_data
        for word, index in self.word_indexes.items(): 
            embedding_vector = self.embeddings[word]
            embedding_matrix[index] = embedding_vector
            
        training_labels = np.array(training_labels)
        sequence_training_data = np.array(sequence_training_data)
        # Convert training labels to one hot encodings 
        training_labels = utils.to_categorical(training_labels, 3)  
        
        print("Num Positive: " + str(self.count_positive))
        print("Num Negative: " + str(self.count_negative))
        print("Num Neutral: " + str(self.count_neutral))
        print("Sequence Length: " + str(self.seq_len))
        print("Training Data Shape: " + str(sequence_training_data.shape))
        print("Embedding Matrix Shape: " + str(embedding_matrix.shape))
        print()
        
        # K-FOLD Cross validation 
        if self.k_fold == True: 
            # Define per-fold score containers\
            print('DOING K-FOLD VALIDATION\n\n')
            acc_per_fold = []
            loss_per_fold = []

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=10, shuffle=True)

            fold_idx = 1
            for train, test in kfold.split(sequence_training_data, training_labels):
                model = Sequential() 
                # Embedding layer -> word embeddings are initial weights 
                embedding_layer = Embedding(self.vocab_size + 1, VECTOR_LENGTH, 
                                            weights=[embedding_matrix],
                                            input_length=self.seq_len, 
                                            trainable=False)            
                model.add(embedding_layer)
                model.add(Dense(units=512, activation='relu'))
                model.add(Flatten())        
                model.add(Dropout(0.5))
                model.add(Dense(units=3, activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                model.fit(sequence_training_data[train], training_labels[train], epochs=3, verbose=0, batch_size=12)
                #self.neural_net = final_model

                loss, accuracy = model.evaluate(
                    sequence_training_data[test], training_labels[test], verbose=0)
                
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
            print('------------------------------------------------------------------------\n')
        
        print('TRAINING FINAL MODEL\n')
        final_model = Sequential() 
        # Embedding layer -> word embeddings are initial weights 
        embedding_layer = Embedding(self.vocab_size + 1, VECTOR_LENGTH, 
                                    weights=[embedding_matrix], 
                                    input_length=self.seq_len, 
                                    trainable = False)   
        final_model.add(embedding_layer) 
        final_model.add(Dense(units=512, activation='relu')) # Hidden layer 
        final_model.add(Flatten()) # Flattens into one dimension                 
        final_model.add(Dropout(0.5)) # Randomly sets half the inputs to zero at each update (avoids overfitting)
        final_model.add(Dense(units=3, activation='softmax')) # Output layer (3 classes)
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        final_model.fit(sequence_training_data, training_labels, epochs=3, verbose=1, batch_size=12)
        self.neural_net = final_model       

       
    # Tests each of the models
    def test(self): 
        print()
        print(
            '------------------------------------------------------------------------')
        print('Testing Brand: ' + self.brand.upper())
        print(
            '------------------------------------------------------------------------')
        print()

        # Get testing data and clean all the tweets 
        testing_data = get_data.get_testing_data(self.brand)
        for tweet in testing_data: 
            tweet[0] = self.get_cleaned_tweet(tweet[0])            

        # Tokenize the testing tweets
        tokenized_testing_tweets = self.get_tokens(testing_data, training=False)

        # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
        # and store them in padded_testing_data. Add their corresponding labels to testing_labels array
        sequence_testing_data, testing_labels = self.convert_to_sequences(tokenized_testing_tweets, training = False)
        testing_labels = np.array(testing_labels)
        # Convert labels to one-hot encodings
        labels = utils.to_categorical(testing_labels, 3)
        padded_testing_data = np.array(sequence_testing_data)
        
        loss, accuracy = self.neural_net.evaluate(
            padded_testing_data, labels, verbose=0)
        print('Testing Data:')
        print('Accuracy: ' + str(accuracy * 100) + '%')
        print('Loss: ' + str(loss))

    
    # Saves model to a folder (in models)
    def save_model(self): 
        model_json = self.neural_net.to_json()
        with open("./models/" + self.brand + "/" + self.brand + '_model.json', 'w') as json_file: 
            json_file.write(model_json)
        self.neural_net.save_weights("./models/" + self.brand + "/" + self.brand + '_model.h5')
        print("Saved model for " + self.brand + " to disk.")
        with open("./models/" + self.brand + "/" + self.brand + '_indexes.txt', 'w') as index_file: 
            index_file.write(str(self.seq_len)+"\n")
            for word in self.word_indexes: 
                to_write = word + " " + str(self.word_indexes[word]) + "\n"
                index_file.write(to_write)       
        index_file.close()         


    # Restores model from folder (in models)
    def load_model(self): 
        json_file = open("./models/" + self.brand + "/" + self.brand + '_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.neural_net = model_from_json(loaded_model_json)
        self.neural_net.load_weights("./models/" + self.brand + "/" + self.brand + '_model.h5')
        self.neural_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Loaded model for " + self.brand + " from disk.")
        index_file = open("./models/" + self.brand + "/" + self.brand + '_indexes.txt', 'r')
        self.seq_len = int(index_file.readline().strip())
        for line in index_file: 
            line_list = line.split()
            word = line_list[0].strip()
            index = int(line_list[1].strip())
            self.word_indexes[word] = index
        index_file.close()
    
    
    # Input: list of tweets to be classified 
    # Output: array containing counts in each class [countPos, countNeg, countNeutral]
    def classify_tweets(self, tweets):
        counts = [0, 0, 0]
        cleaned_tweets = []
        for tweet in tweets: 
            cleaned_tweets.append([self.get_cleaned_tweet(tweet), None])                            
        # Tokenize the testing tweets
        tokenized_tweets = self.get_tokens(cleaned_tweets, training=False)
        # Convert tokenized tweets to their encoded sequences (sequences of unique indexes)
        sequence_data, tweet_labels = self.convert_to_sequences(tokenized_tweets, training = False)
        sequence_data = np.array(sequence_data)
        # Get probabilitiy of each class 
        probabilities = self.neural_net.predict(sequence_data)
        # Convert probability into class 
        classes = np.argmax(probabilities, axis=-1)
        for c in classes: 
            if c == POSITIVE_SENTIMENT: 
                counts[0] += 1
            elif c == NEGATIVE_SENTIMENT: 
                counts[1] += 1
            else: 
                counts[2] += 1
        return counts 
            
        
 # Trains all the models and tests them 
def analyze(brandName):
    sa = SentimentAnalyzer(brandName, True)
    sa.load_model()
    tweets = get_recent_tweets(brandName, 1000)
    all_tweets = []
    for tweet in tweets: 
        all_tweets.append(tweet[0])
    return sa.classify_tweets(all_tweets)


# Trains all the models and tests them 
def train_all(with_k_fold):
    nvidia_sa = SentimentAnalyzer("nvidia", with_k_fold)
    nvidia_sa.train()
    nvidia_sa.save_model()
    microsoft_sa = SentimentAnalyzer("microsoft", with_k_fold)
    microsoft_sa.train()
    microsoft_sa.save_model()
    adobe_sa = SentimentAnalyzer("adobe", with_k_fold)
    adobe_sa.train()
    adobe_sa.save_model()
    ifixit_sa = SentimentAnalyzer("ifixit", with_k_fold)
    ifixit_sa.train()
    ifixit_sa.save_model()


# Tests all the models
def test_all(): 
    nvidia_sa = SentimentAnalyzer("nvidia", False)
    nvidia_sa.load_model()
    microsoft_sa = SentimentAnalyzer("microsoft", False)
    microsoft_sa.load_model()
    adobe_sa = SentimentAnalyzer("adobe", False)
    adobe_sa.load_model()
    ifixit_sa = SentimentAnalyzer("ifixit", False)
    ifixit_sa.load_model()
    print("\nTESTING")
    nvidia_sa.test()
    microsoft_sa.test()
    adobe_sa.test()
    ifixit_sa.test()
    
        
def main():
    print(len(sys.argv))
    if len(sys.argv) <= 1: 
        print("Too Few Arguments. Use -test or -train [-k]")
    else: 
        if sys.argv[1] == "-test":
            test_all() # DON'T NEED TO RUN THIS AGAIN, THEY ARE ALL SAVED
        elif sys.argv[1] == "-train" and len(sys.argv) == 3 and sys.argv[2] == "-k": 
            train_all(True)
        elif sys.argv[1] == "-train": 
            train_all(False)
    
if __name__ == "__main__": 
    main()