import streamlit as st

from rag import qa_bot


def main():
    st.title("DTC Q&A System")

    courses = [
        "data-engineering-zoomcamp",
        "machine-learning-zoomcamp",
        "mlops-zoomcamp"
    ]

    with st.form(key='rag_form'):
        zoomcamp_option = st.selectbox("Select a zoomcamp", courses)
        prompt = st.text_input("Enter your prompt")
        response_placeholder = st.empty()
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        response_placeholder.markdown("Loading...")
        response = qa_bot(prompt, course=zoomcamp_option)
        response_placeholder.markdown(response)

if __name__ == "__main__":
    main()