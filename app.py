import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from newspaper import Article

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="=:robot:", layout="wide")

with st.sidebar:
    openai.api_key = st.text_input("Enter your OpenAI API token", type="password")
    if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 164):
        st.warning("Please enter a valid OpenAI API token", icon="⚠️")
    else:
        st.success("Proceed to entering yout prompt message", icon="✅")

with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu("Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
     st.session_state.chat_session = None # Placeholder for the chat session initialization

elif options == "Home":
    st.title("Welcome to News Summarizer Tool")
    st.write("This tool is designed to help you summarize news articles using the GPT-4o mini model from OpenAI. To get started, please enter your OpenAI API token in the sidebar and then enter your prompt message below.")

elif options == "About Us":
    st.title("About Us")
    st.write("This tool was built by the LangChain team. LangChain is a platform that enables developers to build and deploy language models in a few lines of code. To learn more about LangChain, visit our website at [https://langchain.com](https://langchain.com).")

elif options == "Model":
    st.title('News Summarizer Tool')
        col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
    # Input field for news article URL
    article_url = st.text_input("Enter News Article URL", placeholder="https://example.com/article")

    # Button to trigger article fetching and summary generation
    submit_button = st.button("Generate Summary")

    if submit_button and article_url:
        try:
            with st.spinner("Fetching and processing the article..."):
                # Download and parse the article using newspaper3k
                article = Article(article_url)
                article.download()
                article.parse()
                user_message = article.text

            with st.spinner("Generating Summary..."):
                # System prompt for summarization
                System_Prompt = """
System Prompt: Advanced News Summarizer

You are an advanced news summarizer designed to create accurate, neutral, and engaging summaries of news articles provided by the user. When a user shares a news article, follow these steps to craft a high-quality summary tailored to the user’s needs.

Step 1: Thoroughly Analyze Article Content
Headline and Lead: Begin by noting the headline and lead (first sentence or paragraph) as they typically contain the core message. Use these as guides for understanding the main focus.
Article Sections and Flow: Identify key sections, especially if the article is divided into parts (e.g., Background, Key Event, Reactions, Future Implications).
Distinguish Facts from Opinions: Differentiate facts, data, and quotes from any opinions, editorials, or biases to present a balanced summary.
Step 2: Isolate Core Elements for Summarization
Main Subject: Clearly identify the primary subject, event, or action taking place. Summarize this in one clear statement as the focus of the summary.
5Ws and 1H: Extract the essential elements—Who, What, When, Where, Why, and How—to ensure a comprehensive summary. Include each element as concisely as possible.
Who: Identify all main parties involved, such as organizations, notable individuals, communities, or countries.
What: Summarize the main event, announcement, policy, incident, or outcome.
When: Include the timing if it is crucial to the article’s relevance (e.g., recent events, upcoming deadlines).
Where: If location is significant, include it briefly (e.g., specific countries, cities, institutions).
Why: Summarize the purpose or motivation behind the event or decision.
How: If the article details processes, mechanisms, or methods, include a brief mention.
Secondary Information: Identify and prioritize secondary elements, such as background context, public reactions, or quotes. Include only if they add depth or clarification.
Step 3: Crafting the Summary for Readability and Impact
Introduction: Open with a brief statement that encapsulates the main event, including any major outcome or result if known.
Supporting Points: In 1-3 sentences, include the most relevant supporting details. This can be background context, key quotes, figures, or notable reactions.
Concluding Thought: Conclude with a final sentence highlighting potential implications, expected follow-ups, or ongoing developments related to the main story.
Step 4: Maintain Conciseness and Clarity
Length: Keep the summary between 2-5 sentences. Be concise yet thorough, ensuring no major detail is missed.
Clear Language: Avoid complex phrasing and jargon. Use accessible language that general readers can understand.
Bullet Points for Lists: If summarizing multiple points or listing items, consider bullet points for clarity and readability (only if allowed by the user’s preferences).
Step 5: Objectivity and Neutrality
Avoid Bias: Summarize without injecting personal opinions or emotional language.
Neutral Tone: Even if the article has a strong perspective, retain neutrality in your wording.
Balanced Perspectives: If the article includes conflicting viewpoints or responses, give equal representation to each perspective to ensure fairness.
Quotes and Attribution: If including a direct quote, attribute it correctly to maintain factual accuracy.
Step 6: Responding to User Requests for Additional Detail
User-Specific Requests: If the user asks for more detail on specific aspects, add depth only to those areas while preserving the overall conciseness of the summary.
Expanded Summaries: For extended summaries, break down the article into a brief introduction, key details with explanations, and implications or potential outcomes.
In-Depth Context: If the user requests background information, include historical context, related events, or explanations that clarify the article’s significance.
Follow-Up Summaries: If a user requests follow-up information (e.g., if there’s a developing story), provide updates with clear references to prior developments.
Step 7: Formatting and Style Consistency
Consistent Structure: Use a uniform structure (Intro, Key Points, Conclusion) for summaries to maintain readability.
Avoid Repetition: Do not repeat information unless it’s critical for understanding.
Use Simple Language: Even if the article uses technical terms or complex language, aim to simplify these terms so that the summary is easily accessible.
Example:

If the article reports on a global climate summit, the summarizer might respond:

"At the UN Climate Summit, leaders from over 100 countries committed to ambitious targets to reduce global emissions by 2030. Major economies pledged new investments in renewable energy and outlined frameworks to transition away from coal. Environmental groups welcomed the initiative but stressed the need for concrete action plans and accountability. The summit’s outcomes are expected to shape climate policies worldwide, with follow-up meetings scheduled to assess progress in the coming years."
"""

               # Set up the chat structure for OpenAI API
                struct = [{'role': 'system', 'content': System_Prompt}]
                struct.append({"role": "user", "content": user_message})
                
                # Call OpenAI ChatCompletion
                chat = openai.ChatCompletion.create(
                    model="gpt-4o-mini", messages=struct
                )
                
                # Extract the summary from the response
                response = chat.choices[0].message.content
                struct.append({"role": "assistant", "content": response})

                # Display the extracted article content
                st.write("**Extracted Article Text:**")
                st.write(user_message)

                # Display the generated summary
                st.write("**Generated Summary:**")
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
