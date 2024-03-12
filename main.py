from langchain_community.llms import Cohere
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

st.set_page_config("wide")
# Set your API key in an environment variable before running the app, not in the code.
# os.environ["cohere_api_key"] = "your_api_key_here"

st.header("Truth Or Dare")
st.subheader("Musician 9DX")

input_text = st.radio(options=["Truth", "Dare"], label="Select")

llm = Cohere()

memory1=ConversationBufferMemory(input_key="choice",memory_key="unique")
memory2=ConversationBufferMemory(input_key="truth_or_dare_task",memory_key="unique2")


prompt_1 = PromptTemplate(
    input_variables=["choice"],
    template="PLaying Truth or Dare: Generate one {choice} question. No Explanations. ",
    output_key="truth_or_dare_task"
)

prompt_2 = PromptTemplate(
    input_variables=["choice","truth_or_dare_task"],
    template="The choice was {choice} generate a follow-up question to {truth_or_dare_task}. No Explanations. ",
    output_key="follow_up"
)

llmchain1 = LLMChain(llm=llm, prompt=prompt_1, output_key="truth_or_dare_task",memory=memory1)
llmchain2 = LLMChain(llm=llm, prompt=prompt_2, output_key="follow_up",memory=memory2)

seqChain = SequentialChain(
    chains=[llmchain1, llmchain2],
    input_variables=["choice"],
    output_variables=["truth_or_dare_task", "follow_up"]
)

btn=st.button("Generate")

st.divider()

if btn:
    result = seqChain({"choice": input_text})
    st.snow()
    st.info(result["choice"])
    st.divider()
    st.error(result["truth_or_dare_task"])
    st.divider()
    st.error(result["follow_up"])
