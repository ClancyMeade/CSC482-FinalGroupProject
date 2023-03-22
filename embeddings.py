import os 
import tqdm 
import requests 
import zipfile 

# Downloads file of embeddings that was trained using tweets 
# Run this first 
def get_embeddings():
    URL = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    out_file = 'word_embeddings.zip'
    if os.path.isfile(out_file): 
        print("Embeddings already downloaded.")
    else: 
        # Download zip file of embeddings trained on tweets 
        # Embeddings from GloVe
        # Jeffrey Pennington, Richard Socher, Christopher D. Manning at 
        # https://nlp.stanford.edu/projects/glove/
        print("**********************************************")
        print("   Downloading zip file of word embeddings   ")
        print("**********************************************")
        response = requests.get(URL, stream=True)
        handle = open(out_file, 'wb')
        # Creates progress bar, and writes 512 byte chunks at a time 
        for chunk in tqdm.tqdm(response.iter_content(chunk_size=512)): 
            if chunk: 
                handle.write(chunk)
        handle.close()
        print("Download complete.")
    if os.path.isfile('glove.twitter.27B.100d.txt'): 
        print("File already extracted.")
    else: 
        # Extract the specific file with 100 dimension vectors (about 1 gig)
        zf = zipfile.ZipFile(out_file)
        print("Extracting glove.twitter.27B.100d.txt from word_embeddings.zip")
        zf.extract("glove.twitter.27B.100d.txt")
    # Delete zip file
    os.remove('word_embeddings.zip')
    

# Input: 
#    words: dictionary of words and their unique indexes: key = word, value = index 
# Output: Returns a dictionary of embeddings for each word in words 
#    key = word, value = vector (list of nums)
def get_embedding_dict(words, vector_length): 
    emb_file = 'glove.twitter.27B.100d.txt'
    embedding_dict = {}
    # Go through file and get embeddings for words that we are looking for 
    with open(emb_file, 'r') as embedding_file: 
        for line in embedding_file: 
            line_list = line.split()
            word = line_list[0]
            # Check if we are looking for word, if we are get the vector 
            if word in words: 
                vector = []
                for value in line_list[1:]: 
                    vector.append(float(value))
                embedding_dict[word] = vector                
    # Check if all the words were found using counts 
    num_desired_words = len(words.keys())
    num_found_words = len(embedding_dict.keys())
    num_unknown_words = num_desired_words - num_found_words

    if num_unknown_words > 0: 
        # There are unknown words (don't have their embeddings, just use zero vector)
        for word in words: 
            if word not in embedding_dict: 
                # Use a vector of zeroes for unknown words 
                embedding_dict[word] = [0 for i in range(vector_length)]
    return embedding_dict


if __name__ == "__main__": 
    get_embeddings()