
from os import path
import re
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""
fills vector database with provided file
"""

DATA_PATH = "data/newsarticles_tiny.tsv"
assert path.exists(DATA_PATH), "Missing data file"

# load file
# use commas delimiter, double quotes and replace None with empty string
# to not cause parsing problems with the CSVLoader from langchain
CSV_OPTIONS = { "delimiter": "\t", "quotechar": '"'} #, "restval": ''}
import sys
import csv
csv.field_size_limit(sys.maxsize)
loader = CSVLoader(file_path=DATA_PATH, csv_args=CSV_OPTIONS)
documents = loader.load()

# convert to text snippets
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts. passing persist_directory will persist the database
persist_directory = 'db'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
