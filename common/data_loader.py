from __future__ import annotations
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical
import random
import numpy as np
import pickle




def file_loader(filename: str):
    """This function creates a generator for the data in given document 
    
    Parameters
    ----------
    filename: str
        A file name we want to read from
    """
    file = open(filename, 'r')
    entry: dict
    entry = {}
    for line in file:
        line = line.strip()
        colonPos = line.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        elem_name = line[:colonPos]
        elem_val = line[colonPos+2:]
        entry[elem_name] = elem_val
    yield entry

    




def load_n_data(gen: generator, n: int, attr_data: str,
                attr_labels: str) -> tuple(list, list):
    """This function creates data and labels using generator
    
    This function creates data and labels using generator
    
    Parameters
    ----------
    gen: generator
        We will generate values from this generator
    attr_data: str
        A dictionary key defining a value we want to append to the output list
    n: int
        How many data to be loaded
    attr_labels: str
        A dictionary key defining a value we want to append to the output list
    Returns
    ----------
    list
        A list containing data we were asking for
    list
        A list containing labels we were asking for
    
    """
    out_data: list = []
    out_labels: list = []
    
    for i in range(n):
        next_dict = next(gen)
        if next_dict == {}:
            # Out of data in generator
            return (out_data, out_labels)
        next_value_data: dict = next_dict.get(attr_data)
        # Get the data from our dict
        
        next_value_labels = next_dict.get(attr_labels)    
        while(next_value_data is None or next_value_labels is None):
            # Given attributes not found. Check next
            next_dict = next(gen)
            if next_dict == {}:
                # We are out of data!Just return what we have# Just return what we have
                return (out_data, out_labels)
                
            next_value_data = next_dict.get(attr_data)
            next_value_labels = next_dict.get(attr_labels)-1
        out_data.append(next_value_data)
        out_labels.append(next_value_labels)
    return (out_data, out_labels)

def save_numpy_stars(gen: generator, attr_data:str, attr_labels:str, max_words:int=20000, 
               max_len_review:int=500)-> None:
    """Converts data do numpy arrays and saves them
    
    Parameters
    ----------
    gen: generator
        We will generate values from this generator
    attr_data: str
        A dictionary key defining a value we want to append to the output list
    attr_labels: str
        A dictionary key defining a value we want to append to the output list
    max_words: int
        Vocabulary size (default 20000)
    max_len_review: int
        Review padding size
    """
    str_all_data: list = []
    str_all_labels: list = []

    out_all_data: np.array
    out_all_labels: np.array


    # Load all possible data
    (str_all_data, str_all_labels) = load_n_data(gen, 400000, attr_data, attr_labels)
    tokenizer = Tokenizer(num_words=max_words)
    # Fit using all the data
    tokenizer.fit_on_texts(str_all_data)
    str_all_data = tokenizer.texts_to_sequences(str_all_data)

    out_all_data: np.array = pad_sequences(str_all_data, maxlen = max_len_review)
    
    # Cast to int and substract 1 to get range (0-1)
    str_all_labels = [(int(x.replace(".0", ""))-1)/4 for x in str_all_labels]
    out_all_labels: np.array = np.array(str_all_labels)

    with open('source/stars/tokenizer.pickle', 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    
    for i in range(len(out_all_data)):
        np.save("source/stars/data/" + str(i) + ".npy", out_all_data[i])
    np.save("source/stars/labels.npy", out_all_labels)
    
    return



def save_numpy_titles(gen: generator, attr_data:str, attr_labels:str, max_words:int=20000, 
               max_len_review:int=500, max_len_title:int=10) -> None:
    """Converts data do numpy arrays and saves them
    
    Parameters
    ----------
    gen: generator
        We will generate values from this generator
    file_in: str
        A file name we want to read from
    attr_data: str
        A dictionary key defining a value we want to append to the output list
    attr_labels: str
        A dictionary key defining a value we want to append to the output list
    max_words: int
        Vocabulary size (default 20000)
    max_len_review: int
        Review padding size
    max_len_title: int
        Title padding
    """
    str_all_data: list = []
    str_all_labels: list = []

    out_all_review_data: list = []
    out_all_title_data: list = []
    out_all_labels: list = []


    # Load all possible data
    (str_all_data, str_all_labels) = load_n_data(gen, 400000, attr_data, attr_labels)
    tokenizer = Tokenizer(num_words=max_words)
    # Fit using all the data
    tokenizer.fit_on_texts(str_all_data)
    temp_all_data = tokenizer.texts_to_sequences(str_all_data)
    temp_all_labels = tokenizer.texts_to_sequences(str_all_labels)
    # Split data for different inputs and generate correct output
    for i, review in enumerate(temp_all_data):
        for j, word in enumerate(temp_all_labels[i]):
            out_all_labels.append(word)
            out_all_review_data.append(review)
            if j == 0:
                out_all_title_data.append([])
            else:
                new_item: list = out_all_title_data[-1] + [temp_all_labels[i][j-1]]
                out_all_title_data.append(new_item)

    pad_all_review_data: np.array = pad_sequences(out_all_review_data, maxlen = max_len_review)
    pad_all_title_data: np.array = pad_sequences(out_all_title_data, maxlen = max_len_title)
    
    pad_all_labels: np.array = np.array(out_all_labels)

    with open('source/titles/tokenizer.pickle', 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(len(pad_all_review_data)):
        np.save("source/titles/reviews/" + str(i) + ".npy", pad_all_review_data[i])
        np.save("source/titles/titles/"  + str(i) + ".npy", pad_all_title_data[i])
    np.save("source/titles/labels.npy", pad_all_labels)

    return



def generate_dataset_stars(gen: generator, attr_data: str, attr_labels: str,train_n: int = 64000,
                           test_n: int = 8000, max_words: int = 20000) -> tuple[tuple[list, list],
                                                  tuple[list, list],
                                                  tuple[list, list]]:   
    """This function creates a dataset by using a generator
    
    This function creates a dataset by using a generator
    
    Parameters
    ----------
    gen: generator
        We will generate values from this generator
    attr_data: str
        A dictionary key defining a data we want to append to the output list
    attr_labels: str
        A dictionary key defining a labels we want to append to the output list
    train_n: int
        Defines a size of the data lists
    test_n: int
        Defines a size of the test lists
    max_words: int
        Number of unique words in dataset
    Returns
    ----------
    tokenizer:
        Used tokenizer
    tuple[list, list]
        A tuple of train_data and train_labels
    tuple[list, list]
        A tuple of test_data and test_labels
    tuple[list, list]
        A tuple of val_data and val_labels. It uses the rest of data from generator
        (total_n-train_n-test_m)
    
    """
    # Train data and labels
    str_train_labels: list
    out_train_data: list
    out_train_labels: list
    
    # Test data and labels
    str_test_labels: list
    out_test_data: list
    out_test_labels: list
    
    
    str_all_labels: list
    out_all_data: list
    
    # Load all possible data
    (str_all_data, str_all_labels) = load_n_data(gen, 100000, attr_data, attr_labels)
    
    # Just to randomize set
    list_all = list(zip(str_all_data, str_all_labels))
    random.shuffle(list_all)
    
    str_all_data, str_all_labels = zip(*list_all)
    tokenizer = Tokenizer(num_words=max_words)
    # Fit using all the data
    tokenizer.fit_on_texts(str_all_data)
    out_all_data = tokenizer.texts_to_sequences(str_all_data)
    
     # Generate train raw data
    out_train_data = out_all_data[:train_n]
    str_train_labels = str_all_labels[:train_n]
    # Generate test raw data
    out_test_data = out_all_data[train_n:train_n+test_n] 
    str_test_labels = str_all_labels[train_n:train_n+test_n] 
    # Generate validation data. It contains the rest of data
    out_val_data = out_all_data[train_n+test_n:]
    str_val_labels = str_all_labels[train_n+test_n:]
    
    # Cast to int and substract 1 to get range (0-1)
    out_train_labels = [(int(x.replace(".0", ""))-1)/4 for x in str_train_labels]
    out_test_labels = [(int(x.replace(".0", ""))-1)/4 for x in str_test_labels]
    out_val_labels = [(int(x.replace(".0", ""))-1)/4 for x in str_val_labels]
    
    return tokenizer, (out_train_data, out_train_labels), (out_test_data, out_test_labels), (out_val_data, out_val_labels)



def generate_dataset_titles(gen: generator, attr_data: str, attr_labels: str,
                            train_n: int = 230000, 
                            max_words: int = 20000) -> tuple[tuple[list, list],
                                                  tuple[list, list],
                                                  tuple[list, list]]:   
    """This function creates a dataset for title creation by using a generator
    
    This function creates a dataset for title creation by using a generator
    
    Parameters
    ----------
    gen: generator
        We will generate values from this generator
    attr_data: str
        A dictionary key defining a data we want to append to the output list
    attr_labels: str
        A dictionary key defining a labels we want to append to the output list
    train_n: int
        Defines a size of the data lists
    max_words: int
        Number of unique words in dataset
    Returns
    ----------
    tokenizer:
        A tokenizer used to tokenize reviews
    tuple[list, list, list]
        A tuple of train_review_data, train_title_data, train_labels
    tuple[list, list, list]
        A tuple of test_review_data, test_title_data, test_labels

    """
    str_all_data: list = []
    str_all_labels: list = []

    out_all_review_data: list = []
    out_all_title_data: list = []
    out_all_labels: list = []
    
    # Load all possible data
    (str_all_data, str_all_labels) = load_n_data(gen, 100000, attr_data, attr_labels)
    
    
    tokenizer = Tokenizer(num_words=max_words)
    # Fit using all the data
    tokenizer.fit_on_texts(str_all_data)
    temp_all_data = tokenizer.texts_to_sequences(str_all_data)
    temp_all_labels = tokenizer.texts_to_sequences(str_all_labels)
    
    # Split data for different inputs and generate correnc output
    for i, review in enumerate(temp_all_data):
        for j, word in enumerate(temp_all_labels[i]):
            out_all_labels.append(word)
            out_all_review_data.append(review)
            if j == 0:
                out_all_title_data.append([])
            else:
                new_item: list = out_all_title_data[-1] + [temp_all_labels[i][j-1]]
                out_all_title_data.append(new_item)
            
    # Just to randomize set
    list_all = list(zip(out_all_review_data, out_all_title_data, out_all_labels))
    random.shuffle(list_all)  
    out_all_review_data, out_all_title_data, out_all_labels = zip(*list_all)
    
    # Generate train raw data
    out_train_review_data = out_all_review_data[:train_n]
    out_train_title_data = out_all_title_data[:train_n]
    out_train_labels = out_all_labels[:train_n]
    # Generate test raw data
    out_test_review_data = out_all_review_data[train_n:]
    out_test_title_data = out_all_title_data[train_n:] 
    out_test_labels = out_all_labels[train_n:] 
        
    return tokenizer, (out_train_review_data, out_train_title_data, out_train_labels), (out_test_review_data, out_test_title_data, out_test_labels)


class DataGeneratorStars(Sequence):
    """A class used to represent data sequence generator for stars prediction
    
    A class used to represent data sequence generator for stars prediction

    Attributes
    ----------
    batch_size: int
        Size of generated batch
    it_begin: int
        Start of generation range
    it_end int
        End of generation range
    max_words: int
        Vocabulary size
    max_review_size: int
        Size of each review
    labels: np.array
        Contains all labels
    Methods
    -------
    
    """
    def __init__(self, it_begin: int, it_end: int, batch_size: int = 64, max_words: int = 20000, max_review_size: int = 500):
        """init function

        init function

        Parameters
        -------
        batch_size: int
            Size of generated batch
        it_begin: int
            Start of generation range
        it_end int
            End of generation range
        max_words: int
            Vocabulary size
        max_review_size: int
            Size of each review
        """
        self.batch_size = batch_size
        self.it_begin = it_begin
        self.it_end = it_end
        self.max_words = max_words
        self.max_review_size = max_review_size
        self.labels = np.load("source/stars/labels.npy")
        self.on_epoch_end()

    def __len__(self) -> int:
        """Returns number of batches
        
        Parameters
        ------- 
        Returns
        -------
        int:
            Number of batches
        
        """
        return int(np.floor((self.it_end-self.it_begin)/self.batch_size))

    def __getitem__(self, id: int) -> (np.array, np.array) :
        """Returns one batch of data
        
        Parameters
        -------
        id: int
            Id of batch data

        Returns
        -------
        np.array:
            Number of batches
        """
        # Generate indexes of the batch
        temp_ids = self.ids[id*self.batch_size:(id+1)*self.batch_size]
        # Generate data
        (data, label) = self.data_generation(temp_ids)
        return data, label

    def on_epoch_end(self):
        """Called when epoch ends

        Shuffles data
        """
        self.ids = np.arange(self.it_begin, self.it_end)
        np.random.shuffle(self.ids)

    def data_generation(self, ids):
        """Generate a batch given sizes

        Parameters
        -------
        ids: list
            List of ids used to generate data
        Returns
        -------
        np.array:
            Number of batches
        """
        data = np.zeros((self.batch_size, self.max_review_size))
        labels = np.zeros((self.batch_size), dtype=float)

        for i, id in enumerate(ids):       
            data[i] = np.load('source/stars/data/' + str(id) + '.npy')
            labels[i] = self.labels[id]

        return data, labels


class DataGeneratorTitles(Sequence):
    """A class used to represent data sequence generator for summary predictions
    
    A class used to represent data sequence generator for summary predictions

    Attributes
    ----------
    batch_size: int
        Size of generated batch
    it_begin: int
        Start of generation range
    it_end int
        End of generation range
    max_words: int
        Vocabulary size
    max_review_size: int
        Size of each review
    max_titles_size: int
        Size of each title
    labels: np.array
        Contains all labels
    Methods
    -------
    
    """
    def __init__(self, it_begin: int, it_end: int, batch_size: int = 64, max_words: int = 20000, max_review_size: int = 500, max_title_size: int = 10):
        """init function

        init function

        Parameters
        -------
        batch_size: int
            Size of generated batch
        it_begin: int
            Start of generation range
        it_end int
            End of generation range
        max_words: int
            Vocabulary size
        max_review_size: int
            Size of each review
        """
        self.batch_size = batch_size
        self.it_begin = it_begin
        self.it_end = it_end
        self.max_words = max_words
        self.max_review_size = max_review_size
        self.max_title_size = max_title_size
        self.labels = np.load("source/titles/labels.npy")
        self.on_epoch_end()

    def __len__(self) -> int:
        """Returns number of batches
        
        Parameters
        ------- 
        Returns
        -------
        int:
            Number of batches
        
        """
        return int(np.floor((self.it_end-self.it_begin)/self.batch_size))

    def __getitem__(self, id: int) -> (np.array, np.array) :
        """Returns one batch of data
        
        Parameters
        -------
        id: int
            Id of batch data

        Returns
        -------
        np.array:
            Number of batches
        """
        # Generate indexes of the batch
        temp_ids = self.ids[id*self.batch_size:(id+1)*self.batch_size]
        # Generate data
        (data, label) = self.data_generation(temp_ids)
        return data, label

    def on_epoch_end(self):
        """Called when epoch ends

        Shuffles data
        """
        self.ids = np.arange(self.it_begin, self.it_end)
        np.random.shuffle(self.ids)

    def data_generation(self, ids) -> (dict, np.array):
        """Generate a batch given sizes

        Parameters
        -------
        ids: list
            List of ids used to generate data
        Returns
        -------
        dict:
            Dictionary containing reviews and titles
        np.array:
            Labels for training
        """
        reviews_data: np.array = np.zeros((self.batch_size, self.max_review_size))
        titles_data: np.array = np.zeros((self.batch_size, self.max_title_size))
        labels: np.array = np.zeros((self.batch_size), dtype=float)

        for i, id in enumerate(ids):
            reviews_data[i] = np.load("source/titles/reviews/" + str(id) + ".npy")
            titles_data[i] = np.load("source/titles/titles/" + str(id) + ".npy")
            labels[i] = self.labels[id]

        return {"review": reviews_data, "title": titles_data}, to_categorical(labels, num_classes=self.max_words)