import streamlit as st
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

# Model and tokenizer loading
checkpoint = "./model/LaMini-Flan-T5-248M"  
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# LLM pipeline
def llm_pipeline(pdf_contents):
    # Extract text from the PDF contents
    pdf_document = fitz.open(stream=pdf_contents, filetype="pdf")
    pdf_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text()

    # Use the pipeline to generate the summary
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )

    result = pipe_sum(pdf_text)
    summary = result[0]['summary_text']
    return summary

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using a Smaller Model")

    # Button to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Summarize"):
            # Check if the uploaded file is a PDF
            if uploaded_file.type == "application/pdf":
                summary = llm_pipeline(uploaded_file.read())

                # Display the summary
                st.info("Summarization Complete")
                st.success(summary)
            else:
                st.error("Please upload a valid PDF file.")

if __name__ == "__main__":
    main()
