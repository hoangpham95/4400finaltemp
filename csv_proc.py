import pandas as pd
import csv
from os.path import join


class CSVProcessor:
    def __init__(self, root_dir, file_name):
        self.root_dir = root_dir
        self.file_name = file_name

    def process(self):
        df = pd.read_csv(filepath_or_buffer=join(
            self.root_dir, self.file_name))
        return (df['Id'], df['Class'])
