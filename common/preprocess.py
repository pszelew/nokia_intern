from common.data_loader import save_numpy_titles, save_numpy_stars, file_loader
import os
import shutil

def preprocess_stars(file_in: str) -> None:
    """This function prepares data to be used by stars prediction
    
    Parameters
    ----------
    file_in: str
        Input file name. In our case probably Cell_Phones_&_Accessories.txt
    Returns
    ----------
    None
    """

    try:
        shutil.rmtree("source/stars")
    except:
        pass
    os.mkdir("source/stars")
    os.mkdir("source/stars/data")

    gen = file_loader(file_in)
    save_numpy_stars(gen, "review/text", "review/score")
    


def preprocess_titles(file_in: str) -> None:
    """This function prepares data to be used by summaries prediction
    
    Parameters
    ----------
    file_in: str
        Input file name. In our case probably Cell_Phones_&_Accessories.txt
    Returns
    ----------
    None
    """
    try:
        shutil.rmtree("source/titles")
    except:
        pass
    os.mkdir("source/titles")
    os.mkdir("source/titles/reviews")
    os.mkdir("source/titles/titles")
    
    gen = file_loader(file_in)
    save_numpy_titles(gen, "review/text", "review/summary")


def preprocess_dict(d: dict):
    """This function preprocesses dictionary.

    This function preprocesses dictionary. It basically converst data to proper data types. Changes origial dictionary!
    
    Parameters
    ----------
    d: str
        Dictionary to be preprocessed
    Returns
    ----------
    None
    """
    for entry in d:
        # Preprocess price
        if entry["product/price"] == "unknown":
            entry["product/price"] = None
        else:
            entry["product/price"] = float(entry["product/price"])
        # Preprocess userId
        if entry["review/userId"] == "unknown":
            entry["review/userId"] = None
        # Preprocess helpfulness
        sl: int = entry["review/helpfulness"].find("/")
        entry["review/helpfulPositiveVotes"] = int(entry["review/helpfulness"][0:sl])
        entry["review/helpfulAllVotes"] = int(entry["review/helpfulness"][sl+1:])
        # Preprocess score
        entry["review/score"] = int(entry["review/score"].replace(".0", ""))
        # Preprocess time
        entry["review/time"] = int(entry["review/time"])
    

if __name__ == "__main__":
    preprocess_stars("source/Cell_Phones_&_Accessories.txt")
    preprocess_titles("source/Cell_Phones_&_Accessories.txt")