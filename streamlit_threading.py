
import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid


# ****************************************
# Utility Functions
# ****************************************

def generate_thread_id():
    return str(uuid.uuid4())


def generate_title(user_input):
    words = user_input.split()
    title = " ".join(words[:5])

    if len(words) > 5:
        title += "..."

    return title


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

    add_thread(thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )

    return state.values.get('messages', [])


def format_message(content):

    if isinstance(content, str):
        return content

    elif isinstance(content, list):
        text = ""

        for item in content:

            if isinstance(item, dict) and item.get('type') == 'text':
                text += item.get('text', '')

            elif isinstance(item, str):
                text += item

        return text

    return str(content)


# ****************************************
# Session Setup
# ****************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}


add_thread(st.session_state['thread_id'])


# ****************************************
# Sidebar UI
# ****************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:

    title = st.session_state['thread_titles'].get(
        thread_id, "New Conversation"
    )

    if st.sidebar.button(title, key=thread_id):

        # Switch to selected conversation
        st.session_state['thread_id'] = thread_id

        # Load conversation messages
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = 'user'

            elif isinstance(msg, (AIMessage, AIMessageChunk)):
                role = 'assistant'

            else:
                continue

            temp_messages.append({
                'role': role,
                'content': format_message(msg.content)
            })

        st.session_state['message_history'] = temp_messages

        st.rerun()


# ****************************************
# Main UI
# ****************************************

st.title("AI Chatbot")
st.caption("Powered by LangGraph")


# Display conversation history
for message in st.session_state['message_history']:

    with st.chat_message(message['role']):
        st.markdown(format_message(message['content']))


# ****************************************
# User Input
# ****************************************

user_input = st.chat_input('Type here...')

if user_input:

    # Generate title from the first prompt only
    if not st.session_state['message_history']:

        st.session_state['thread_titles'][
            st.session_state['thread_id']
        ] = generate_title(user_input)

    # Display user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.markdown(user_input)


    # LangGraph configuration
    CONFIG = {
        'configurable': {
            'thread_id': st.session_state['thread_id']
        }
    }


    # Stream AI response
    with st.chat_message('assistant'):

        def ai_only_stream():

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

                            if isinstance(item, dict) and item.get('type') == 'text':
                                yield item.get('text', '')

        ai_message = st.write_stream(ai_only_stream())


    # Save AI response
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message
    })
