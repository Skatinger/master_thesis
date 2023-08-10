
import  os
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# import openai embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.qa import RetrievalQA
import pandas as pd

persist_directory =  'db'
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# load the rulings
rulings = pd.read_csv('manually_reidentified.csv')

# for every ruling do a prediction

for ruling in rulings.iterrows():
    print(ruling["file_number"])
    # prompt the qa system with the ruling
    query = "Who is the person referred to as <mask> in the following text?\n\n" + ruling["text"]
    answer = qa.run(query)
    print(answer)
    import pdb; pdb.set_trace()
