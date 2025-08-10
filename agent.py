import fitz  
import google.generativeai as genai
import gradio as gr
GEMINI_API_KEY = "AIzaSyCWfIosHYmPC_j28VQWiqu6fiCxRQQXfLA" 
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

pdf_text_global = ""

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file.name) as pdf_doc:
        for page in pdf_doc:
            text += page.get_text()
    return text


def chunk_text(text, max_chars=3000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_pdf(pdf_file):
    global pdf_text_global
    pdf_text_global = extract_text_from_pdf(pdf_file)

    chunks = chunk_text(pdf_text_global, max_chars=3000)
    summaries = []

    for idx, chunk in enumerate(chunks, start=1):
        prompt = f"""
        Summarize the following text clearly and concisely.
        Use bullet points and highlight key facts.

        Text:
        {chunk}
        """
        response = model.generate_content(prompt)
        summaries.append(f"--- Summary Part {idx} ---\n" + response.text)

    
    combined_summary = "\n".join(summaries)
    return combined_summary

def answer_question(question):
    if not pdf_text_global:
        return "Please upload and summarize a PDF first."
    
    prompt = f"""
    You are an AI assistant. Use ONLY the following PDF content to answer the question accurately.

    PDF Content:
    {pdf_text_global}

    Question: {question}
    """
    response = model.generate_content(prompt)
    return response.text

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF Summarizer + Q&A (Gemini AI) — Large PDF Ready 🚀")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            summarize_btn = gr.Button("Summarize PDF")
            summary_output = gr.Textbox(label="PDF Summary", lines=20)
        
        with gr.Column():
            question_input = gr.Textbox(label="Ask a follow-up question")
            ask_btn = gr.Button("Ask Question")
            answer_output = gr.Textbox(label="Answer", lines=10)

    summarize_btn.click(summarize_pdf, inputs=[pdf_input], outputs=[summary_output])
    ask_btn.click(answer_question, inputs=[question_input], outputs=[answer_output])

demo.launch()
