
import  os
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
# import openai embeddings
from langchain.embeddings import OpenAIEmbeddings

persist_directory =  'db'
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)