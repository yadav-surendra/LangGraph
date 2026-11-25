
import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessageChunk

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.caption("Chat with Google Gemini")

CONFIG = {
    'configurable': {
        'thread_id': 'thread-1'
    }
}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])


# User input
user_input = st.chat_input("Type here...")

if user_input:

    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.write(user_input)

    # Stream AI response
    with st.chat_message('assistant'):

        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):

                if isinstance(message_chunk, AIMessageChunk):

                    content = message_chunk.content

                    if isinstance(content, str):
                        yield content

                    elif isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'text':
                                yield item.get('text', '')

        ai_message = st.write_stream(generate_response())

    # Save complete response
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message
    })
