
import  os
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# import openai embeddings
import openai
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

persist_directory =  'db'
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# retriever = vectordb.as_retriever()

# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# load the rulings
rulings = pd.read_csv('data/manually_reidentified.csv')

# for every ruling do a prediction
for index, ruling in rulings.iterrows():
    import pdb; pdb.set_trace()
    print(ruling["file_number"])
    # prompt the qa system with the ruling
    ruling_text = ruling["full_text"][:10000]
    # TODO: use gpt to paraphrase the ruling

    # get the top 5 documents
    documents = vectordb.similarity_search(ruling_text, k=5)

    input = "Who is the person referred to as <mask> in the following text?\n\n" + ruling_text + "\n\n"
    input += "Use the following articles to find the correct name of the person:\n\n"
    for document in documents:
        input += document.page_content + "\n\n"


    response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    { "role": "user", "content": input },
                ],
                # temperature=0.5,
                max_tokens=10,
                top_p=0.1,
                n=5,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )

    print(response)
