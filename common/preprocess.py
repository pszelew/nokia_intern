from common.data_loader import save_numpy_titles, save_numpy_stars, file_loader
import os
import shutil

def preprocess_stars(file_in: str):
    try:
        shutil.rmtree("source/stars")
    except:
        pass
    os.mkdir("source/stars")
    os.mkdir("source/stars/data")

    gen = file_loader(file_in)
    save_numpy_stars(gen, "review/text", "review/score")
    


def preprocess_titles(file_in: str):
    try:
        shutil.rmtree("source/titles")
    except:
        pass
    os.mkdir("source/titles")
    os.mkdir("source/titles/reviews")
    os.mkdir("source/titles/titles")
    
    gen = file_loader(file_in)
    save_numpy_titles(gen, "review/text", "review/summary")


if __name__ == "__main__":
    preprocess_stars("source/Cell_Phones_&_Accessories.txt")
    preprocess_titles("source/Cell_Phones_&_Accessories.txt")