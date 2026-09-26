import streamlit as st

from rag import create_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "Employee Policy Assistant",

    page_icon=
        "📘",

    layout=
        "centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📘 Employee Policy Assistant"
)


st.caption(
    "Ask questions about Acme Global Services employee policies."
)


# ============================================================
# LOAD RAG ONCE
# ============================================================

@st.cache_resource
def load_rag():

    return create_rag()


rag = load_rag()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a policy question..."
)


if question:

    # ----------------------------
    # Show user message
    # ----------------------------

    st.session_state.messages.append({

        "role":
            "user",

        "content":
            question
    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # ----------------------------
    # Generate answer
    # ----------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching policies..."
        ):

            try:

                result = rag.ask(
                    question
                )


                answer = (
                    result["answer"]
                )


                st.markdown(
                    answer
                )


                # --------------------
                # Sources
                # --------------------

                if result["sources"]:

                    with st.expander(
                        "Retrieved sources"
                    ):

                        for source in (
                            result["sources"]
                        ):

                            st.markdown(
                                f"""
### Source {source['number']}

**Section:** {source['section'] or 'Unknown'}

**Similarity:** {source['score']}

{source['text']}

---
"""
                            )


                st.session_state.messages.append({

                    "role":
                        "assistant",

                    "content":
                        answer
                })


            except Exception as error:

                st.error(
                    "The request failed."
                )

                st.exception(
                    error
                )