# integrate code with openAI

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st


#initialze the llm from open ai

os.environ["OPENAI_API_KEY"]=openai_key

#stramlit framework

st.title('Search about cricketers')
input_text=st.text_input("search")



#memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
stat_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history ')

#prompt template 


first_input_prompt= PromptTemplate(
   input_variables=['name'],
   template="tell me about {name}"
)

#openai llms
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)


second_input_prompt= PromptTemplate(
   input_variables=['person'],
   template=" what are the stats of {person} by 2020"
)

#memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
stat_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history ')


#openai llms
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='stats',memory=stat_memory)

parent_chain=SequentialChain(chains=[chain,chain2],
                             input_variables=['name'],
                             output_variables=['person','stats'],
                             verbose=True)



if input_text:
   st.write(parent_chain({'name':input_text}))
   
   with st.expander('stats'):
      st.info(stat_memory.buffer)
   
   with st.expander('person name'):
      st.info(person_memory.buffer)
    


