import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


## streamlit app
st.set_page_config(page_title="Langchain: Summarize text From YT or Website",page_icon="🦜")
st.title("🦜 Langchain: Summarize text From YT or Website")
st.subheader("Summarize URL")

## Get the Groq API Key and url(YT or Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",type="password")

generic_url = st.text_input("URL",label_visibility="collapsed")
prompt_template="""
Provide summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    #validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
        
    elif not validators.url(generic_url):
        st.error("Please provide a valid URL")
    else:
        try:
            with st.spinner("waiting..."):
                llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
                ##loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
                docs = loader.load()

                #chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception Occurred: {e}")
