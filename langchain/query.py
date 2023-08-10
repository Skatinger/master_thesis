
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

def paraphrase_ruling(text):
    prompt = """Paraphrase the following court ruling. Take care to not lose any details, focus on things that could
                be interesting to read in a news article or specific details such as names, locations, money and costs,
                and most importantly the verdict. It does not have to be very short, it's more important to retain
                all information. The ruling:\n\n""" + text + "\n\n"
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613", # use gpt-3.5-turbo-0613 for less strict rate limits
                messages=[
                    { "role": "user", "content": prompt },
                ],
                # temperature=0.5,
                max_tokens=2000,
                top_p=0.5,
                n=1,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )
    return response.choices[0]["message"]["content"]

# load the rulings
rulings = pd.read_csv('data/manually_reidentified.csv')

# for every ruling do a prediction
for index, ruling in rulings.iterrows():
    import pdb; pdb.set_trace()
    print(ruling["file_number"])
    # prompt the qa system with the ruling
    ruling_text = ruling["full_text"] # [:10000]
    
    # paraphrase the rulings text to make it fit into the token limit
    paraphrased_ruling = paraphrase_ruling(ruling_text)

    # get the top 5 documents
    documents = vectordb.similarity_search(ruling_text, k=5)

    input = """Who is the person referred to as <mask> in the following text? Give the name
               and other relevant information about the person if you can. The name is the imporant thing.\n\n"""
    input + "The texts:\n\n" + paraphrased_ruling + "\n\n"
    input += "Use the following articles to find the correct name of the person:\n\n"
    for document in documents:
        input += document.page_content + "\n\n"


    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613", # use gpt-3.5-turbo-0613 for less strict rate limits
                messages=[
                    { "role": "user", "content": input },
                ],
                # temperature=0.5,
                max_tokens=200,
                top_p=0.1,
                n=5,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )

    print(response)
