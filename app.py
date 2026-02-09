# # #MAIN MAIN CODE HERE
# # import streamlit as st
# # import os
# # import tempfile
# # import time
# # from pathlib import Path
# # import base64
# # from PIL import Image
# # import io
# # import fitz  # PyMuPDF
# # import pandas as pd
# # import numpy as np
# # import google.generativeai as genai
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_core.messages import HumanMessage, AIMessage
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # # Set page configuration
# # st.set_page_config(page_title="PDF Summarizer with Gemini Pro", layout="wide")

# # # Sidebar for API key input
# # with st.sidebar:
# #     st.title("PDF Summarizer with Gemini Pro")
# #     api_key = api_key = st.text_input("Enter your Google API Key:AIzaSyCMFd8JSDYzEpPaQnQzt6yV8wkaP8q4q9U", type="password")
# #     st.info("Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
# #     if api_key:
# #         os.environ["GOOGLE_API_KEY"] = api_key
# #         genai.configure(api_key=api_key)

# # # Initialize session state variables if they don't exist
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []
# # if "bookmarks" not in st.session_state:
# #     st.session_state.bookmarks = {}
# # if "current_pdf" not in st.session_state:
# #     st.session_state.current_pdf = None
# # if "pdf_content" not in st.session_state:
# #     st.session_state.pdf_content = ""
# # if "vector_store" not in st.session_state:
# #     st.session_state.vector_store = None
# # if "qa_chain" not in st.session_state:
# #     st.session_state.qa_chain = None
# # if "images" not in st.session_state:
# #     st.session_state.images = []
# # if "tables" not in st.session_state:
# #     st.session_state.tables = []

# # def extract_text_and_images_from_pdf(pdf_file):
# #     """Extract text, images, and tables from a PDF file."""
# #     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
# #         tmp_file.write(pdf_file.getvalue())
# #         tmp_path = tmp_file.name
    
# #     text_content = ""
# #     images = []
# #     tables = []
# #     doc = None
    
# #     try:
# #         # Open the document
# #         doc = fitz.open(tmp_path)
        
# #         for page_num, page in enumerate(doc):
# #             # Extract text
# #             text_content += page.get_text()
            
# #             # Extract images
# #             image_list = page.get_images(full=True)
# #             for img_index, img in enumerate(image_list):
# #                 xref = img[0]
# #                 base_img = doc.extract_image(xref)
# #                 image_bytes = base_img["image"]
                
# #                 # Convert image bytes to a format suitable for display
# #                 try:
# #                     image = Image.open(io.BytesIO(image_bytes))
# #                     images.append({
# #                         "page": page_num + 1,
# #                         "image": image,
# #                         "caption": f"Image {img_index+1} from page {page_num+1}"
# #                     })
# #                 except Exception as e:
# #                     print(f"Error processing image: {e}")
            
# #             # Extract tables (simplified approach - detecting table-like structures)
# #             try:
# #                 tables_on_page = page.find_tables()
# #                 if tables_on_page:
# #                     for table_idx, table in enumerate(tables_on_page):
# #                         if table.rows > 1:  # Only include if it has more than one row
# #                             try:
# #                                 df = pd.DataFrame(table.extract())
# #                                 tables.append({
# #                                     "page": page_num + 1,
# #                                     "table": df,
# #                                     "caption": f"Table {table_idx+1} from page {page_num+1}"
# #                                 })
# #                             except Exception as e:
# #                                 print(f"Error processing table: {e}")
# #             except Exception as e:
# #                 print(f"Error finding tables on page {page_num + 1}: {e}")
    
# #     except Exception as e:
# #         st.error(f"Error processing PDF: {e}")
    
# #     finally:
# #         # Important: Close the document
# #         if doc:
# #             doc.close()
        
# #         # Add a small delay to ensure file handles are released
# #         time.sleep(0.5)
        
# #         # Try to delete the file, but don't crash if we can't
# #         try:
# #             os.unlink(tmp_path)
# #         except PermissionError:
# #             print(f"Could not delete temporary file {tmp_path} - it will be cleaned up later")
    
# #     return text_content, images, tables

# # def process_pdf_with_langchain(text_content):
# #     """Process PDF content using LangChain and create vector store."""
# #     # Split text into chunks
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
# #     chunks = text_splitter.split_text(text_content)
    
# #     if not chunks:
# #         st.error("Could not extract meaningful text from the PDF.")
# #         return None, None
    
# #     # Create embeddings and vector store
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
# #     # Create QA chain
# #     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
# #     qa_chain = ConversationalRetrievalChain.from_llm(
# #         llm=llm,
# #         retriever=vector_store.as_retriever(),
# #         return_source_documents=True
# #     )
    
# #     return vector_store, qa_chain

# # def get_gemini_response(prompt):
# #     """Get response from Gemini Pro model."""
# #     try:
# #         model = genai.GenerativeModel('gemini-2.0-flash')
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         return f"Error: {str(e)}"

# # def get_gemini_vision_response(prompt, image):
# #     """Get response from Gemini Pro Vision model for image analysis."""
# #     try:
# #         model = genai.GenerativeModel('gemini-2.0-flash-vision')
        
# #         # Convert PIL image to bytes for Gemini
# #         buffered = io.BytesIO()
# #         image.save(buffered, format="PNG")
# #         img_bytes = buffered.getvalue()
        
# #         # Create the content parts properly
# #         image_part = {
# #             "inline_data": {
# #                 "mime_type": "image/png",
# #                 "data": base64.b64encode(img_bytes).decode('utf-8')
# #             }
# #         }
        
# #         # Use the correct API format for multimodal inputs
# #         response = model.generate_content([prompt, image_part])
# #         return response.text
# #     except Exception as e:
# #         return f"Error analyzing image: {str(e)}"

# # def summarize_pdf(text_content):
# #     """Generate a summary of the PDF content."""
# #     prompt = f"""
# #     Please provide a comprehensive summary of the following document:
    
# #     {text_content[:15000]}  # Limiting to first 15000 chars for API limits
    
# #     Include:
# #     1. Main topic and purpose
# #     2. Key points and arguments
# #     3. Major findings or conclusions
# #     4. Structure of the document
    
# #     Make the summary detailed enough to capture the essence but concise enough to be quickly digestible.
# #     """
    
# #     return get_gemini_response(prompt)

# # def analyze_image(image):
# #     """Analyze and describe an image using Gemini Pro Vision."""
# #     prompt = """
# #     Please analyze this image from a PDF document and provide:
# #     1. A detailed description of what you see
# #     2. Any text content visible in the image
# #     3. The likely purpose of this image in the document
# #     4. Any notable features or elements
    
# #     Be detailed but concise.
# #     """
    
# #     return get_gemini_vision_response(prompt, image)

# # def analyze_table(df):
# #     """Analyze and describe a table using Gemini Pro."""
# #     # Convert DataFrame to string representation
# #     table_str = df.to_string()
    
# #     prompt = f"""
# #     Please analyze this table extracted from a PDF document:
    
# #     {table_str}
    
# #     Provide:
# #     1. A clear description of what this table represents
# #     2. The key information it conveys
# #     3. Any patterns or notable data points
# #     4. How this table might relate to the document's content
    
# #     Be detailed but concise.
# #     """
    
# #     return get_gemini_response(prompt)

# # def add_bookmark(page_number, description):
# #     """Add a bookmark for a specific page."""
# #     if st.session_state.current_pdf:
# #         if page_number not in st.session_state.bookmarks:
# #             st.session_state.bookmarks[page_number] = description
# #             return True
# #     return False

# # # Main application interface
# # st.title("PDF Summarizer with Gemini Pro")

# # # API key check
# # if not api_key:
# #     st.warning("Please enter your Google API Key in the sidebar to continue.")

# # # PDF upload section
# # uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# # if uploaded_file is not None and api_key:
# #     # Check if this is a new PDF file
# #     if st.session_state.current_pdf != uploaded_file.name:
# #         with st.spinner("Processing PDF..."):
# #             # Reset state for new PDF
# #             st.session_state.current_pdf = uploaded_file.name
# #             st.session_state.chat_history = []
            
# #             try:
# #                 # Extract content from PDF
# #                 text_content, images, tables = extract_text_and_images_from_pdf(uploaded_file)
# #                 st.session_state.pdf_content = text_content
# #                 st.session_state.images = images
# #                 st.session_state.tables = tables
                
# #                 # Process with LangChain
# #                 st.session_state.vector_store, st.session_state.qa_chain = process_pdf_with_langchain(text_content)
                
# #                 st.success(f"PDF processed successfully: {uploaded_file.name}")
# #             except Exception as e:
# #                 st.error(f"Error processing PDF: {str(e)}")
# #                 st.session_state.current_pdf = None
    
# #     # Only proceed if PDF was processed successfully
# #     if st.session_state.current_pdf:
# #         # Create tabs for different features
# #         tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Q&A", "Images & Tables", "Chat History", "Bookmarks"])
        
# #         with tab1:
# #             st.header("PDF Summary")
# #             if st.button("Generate Summary"):
# #                 with st.spinner("Generating summary..."):
# #                     summary = summarize_pdf(st.session_state.pdf_content)
# #                     st.markdown(summary)
# #                     # Add to chat history
# #                     st.session_state.chat_history.append(
# #                         (HumanMessage(content="Generate a summary of this PDF."), 
# #                          AIMessage(content=summary))
# #                     )
        
# #         with tab2:
# #             st.header("Question & Answer")
# #             question = st.text_input("Ask a question about the PDF:")
# #             if question and st.button("Get Answer"):
# #                 if st.session_state.qa_chain:
# #                     with st.spinner("Finding answer..."):
# #                         try:
# #                             chat_history = [(msg[0].content, msg[1].content) for msg in st.session_state.chat_history]
# #                             result = st.session_state.qa_chain({"question": question, "chat_history": chat_history})
# #                             answer = result["answer"]
# #                             st.markdown(answer)
                            
# #                             # Add to chat history
# #                             st.session_state.chat_history.append(
# #                                 (HumanMessage(content=question), 
# #                                  AIMessage(content=answer))
# #                             )
# #                         except Exception as e:
# #                             st.error(f"Error getting answer: {str(e)}")
# #                 else:
# #                     st.warning("Q&A system not initialized. Please ensure the PDF was processed correctly.")
        
# #         with tab3:
# #             st.header("Images & Tables")
# #             col1, col2 = st.columns(2)
            
# #             with col1:
# #                 st.subheader("Images")
# #                 if st.session_state.images:
# #                     selected_image_idx = st.selectbox(
# #                         "Select an image to analyze:", 
# #                         range(len(st.session_state.images)),
# #                         format_func=lambda i: f"Image {i+1} (Page {st.session_state.images[i]['page']})"
# #                     )
                    
# #                     if selected_image_idx is not None:
# #                         img_data = st.session_state.images[selected_image_idx]
# #                         st.image(img_data["image"], caption=img_data["caption"])
                        
# #                         if st.button("Analyze Image"):
# #                             with st.spinner("Analyzing image..."):
# #                                 analysis = analyze_image(img_data["image"])
# #                                 st.markdown("### Image Analysis")
# #                                 st.markdown(analysis)
                                
# #                                 # Add bookmark option
# #                                 if st.button("Bookmark this image"):
# #                                     page_num = img_data["page"]
# #                                     success = add_bookmark(
# #                                         page_num, 
# #                                         f"Image {selected_image_idx+1} on page {page_num}"
# #                                     )
# #                                     if success:
# #                                         st.success(f"Bookmark added for page {page_num}")
# #                 else:
# #                     st.info("No images found in the PDF.")
            
# #             with col2:
# #                 st.subheader("Tables")
# #                 if st.session_state.tables:
# #                     selected_table_idx = st.selectbox(
# #                         "Select a table to analyze:", 
# #                         range(len(st.session_state.tables)),
# #                         format_func=lambda i: f"Table {i+1} (Page {st.session_state.tables[i]['page']})"
# #                     )
                    
# #                     if selected_table_idx is not None:
# #                         table_data = st.session_state.tables[selected_table_idx]
# #                         st.dataframe(table_data["table"])
                        
# #                         if st.button("Analyze Table"):
# #                             with st.spinner("Analyzing table..."):
# #                                 analysis = analyze_table(table_data["table"])
# #                                 st.markdown("### Table Analysis")
# #                                 st.markdown(analysis)
                                
# #                                 # Add bookmark option
# #                                 if st.button("Bookmark this table"):
# #                                     page_num = table_data["page"]
# #                                     success = add_bookmark(
# #                                         page_num, 
# #                                         f"Table {selected_table_idx+1} on page {page_num}"
# #                                     )
# #                                     if success:
# #                                         st.success(f"Bookmark added for page {page_num}")
# #                 else:
# #                     st.info("No tables found in the PDF.")
        
# #         with tab4:
# #             st.header("Chat History")
# #             if st.session_state.chat_history:
# #                 for i, (human_msg, ai_msg) in enumerate(st.session_state.chat_history):
# #                     st.markdown(f"**Question {i+1}**: {human_msg.content}")
# #                     st.markdown(f"**Answer {i+1}**: {ai_msg.content}")
# #                     st.divider()
# #             else:
# #                 st.info("No chat history yet. Ask questions in the Q&A tab.")
        
# #         with tab5:
# #             st.header("Bookmarks")
# #             if st.session_state.bookmarks:
# #                 st.write("Your bookmarks:")
# #                 for page_num, description in sorted(st.session_state.bookmarks.items()):
# #                     col1, col2 = st.columns([3, 1])
# #                     with col1:
# #                         st.markdown(f"**Page {page_num}**: {description}")
# #                     with col2:
# #                         # Option to remove bookmark
# #                         if st.button(f"Remove", key=f"remove_{page_num}"):
# #                             del st.session_state.bookmarks[page_num]
# #                             st.experimental_rerun()
# #             else:
# #                 st.info("No bookmarks yet. You can add bookmarks when viewing images or tables.")
            
# #             # Manual bookmark addition
# #             st.subheader("Add a new bookmark")
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 page_num = st.number_input("Page number", min_value=1, step=1)
# #             with col2:
# #                 desc = st.text_input("Description")
            
# #             if st.button("Add Bookmark"):
# #                 if page_num and desc:
# #                     success = add_bookmark(page_num, desc)
# #                     if success:
# #                         st.success(f"Bookmark added for page {page_num}")
# #                     else:
# #                         st.error("Failed to add bookmark. Make sure PDF is loaded.")

# # elif not api_key:
# #     st.info("Please enter your Google API Key in the sidebar to get started.")
# # else:
# #     st.info("Please upload a PDF document to get started.")

# # # Add sidebar information about features
# # with st.sidebar:
# #     st.subheader("Features")
# #     st.markdown("""
# #     - **Summary**: Get a comprehensive summary of the PDF
# #     - **Q&A**: Ask questions about the PDF content
# #     - **Image & Table Analysis**: View and analyze images and tables
# #     - **Chat History**: Review your previous questions and answers
# #     - **Bookmarks**: Save important pages for quick reference
# #     """)
    
# #     st.subheader("About")
# #     st.markdown("""
# #     This app uses:
# #     - **Google Gemini Pro** for text understanding and generation
# #     - **Gemini Pro Vision** for image analysis
# #     - **LangChain** for document processing and Q&A
# #     - **FAISS** for vector search and retrieval
# #     """)





















# # import streamlit as st
# # import os
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import FAISS
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # import google.generativeai as genai
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.decomposition import LatentDirichletAllocation

# # # ✅ Set API key as environment variable for LangChain
# # os.environ["GOOGLE_API_KEY"] = "AIzaSyCD-Ev3pRRqE3-35-F9P9zF_50PfOftuUU"  # 🔐 Replace with your actual API key
# # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# # # 🧠 App title
# # st.set_page_config(page_title="Chat with PDF", layout="wide")
# # st.markdown(""" 
# #     <style>
# #         .block-container {
# #             padding: 2rem 2rem 2rem 2rem;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # st.header("📄 Chat with PDF using Gemini 1.5 Pro")

# # # Session state for chat history
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # 📥 Upload PDF
# # pdf = st.file_uploader("Upload your PDF", type="pdf")

# # if pdf is not None:
# #     # 📖 Read PDF
# #     reader = PdfReader(pdf)
# #     raw_text = ""
# #     for page in reader.pages:
# #         content = page.extract_text()
# #         if content:
# #             raw_text += content

# #     # 📋 Display summary of PDF
# #     with st.expander("📑 PDF Summary"):
# #         llm_summary = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.2)
# #         summary_prompt = f"Summarize the following document content in a few bullet points:\n{raw_text[:3000]}"
# #         summary = llm_summary.invoke(summary_prompt)
# #         st.write(summary.content)

# #     # 🔍 Split text into chunks
# #     splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
# #     chunks = splitter.split_text(raw_text)

# #     # Step 1: Convert text chunks into a term frequency matrix (document-term matrix)
# #     vectorizer = CountVectorizer(stop_words='english')  # Using 'english' stop words to filter common words
# #     X = vectorizer.fit_transform(chunks)  # Convert the chunks into numerical vectors

# #     # Step 2: Apply LDA
# #     lda = LatentDirichletAllocation(n_components=2, random_state=42)  # You can change the number of topics (n_components)
# #     lda.fit(X)

# #     # Step 3: Print the topics discovered by LDA
# #     st.write("### Topics discovered by LDA:")
# #     for index, topic in enumerate(lda.components_):
# #         st.write(f"**Topic #{index + 1}:**")
# #         st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]])
# #         st.write("\n")

# #     # 🧬 Gemini Embeddings
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# #     # 📚 Create FAISS vector store
# #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

# #     # 🤖 Load Gemini LLM (1.5 Pro)
# #     llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

# #     # 🔗 Create QA chain
# #     chain = load_qa_chain(llm, chain_type="stuff")

# #     # ❓ Get user question
# #     question = st.text_input("Ask a question about the PDF:")

# #     if question:
# #         with st.spinner("Thinking..."):
# #             docs = vectorstore.similarity_search(question)
# #             response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
# #             answer = response["output_text"]
# #             st.subheader("💬 Answer")
# #             st.write(answer)

# #             # Save in chat history
# #             st.session_state.chat_history.append((question, answer))

# #             # Download option
# #             st.download_button("📥 Download Answer", answer, file_name="response.txt")

# #     # Display chat history in sidebar
# #     with st.sidebar:
# #         st.title("🕘 Chat History")
# #         if st.session_state.chat_history:
# #             for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
# #                 st.markdown(f"**Q{i}:** {q}")
# #                 st.markdown(f"**A{i}:** {a}")
# #                 st.markdown("---")  # Add separation

# #         # Clear chat button
# #         if st.button("🗑️ Clear Chat"):
# #             st.session_state.chat_history = []








# # # MAIN CODE STARTS HERE
# # import streamlit as st
# # import os
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import FAISS
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # import google.generativeai as genai

# # # ✅ Set API key as environment variable for LangChain
# # os.environ["GOOGLE_API_KEY"] = "AIzaSyBRaaHxynRERnmeSnN_-PZpHl9E06Swth4"  # 🔐 Replace with your actual API key
# # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# # # 🧠 App title
# # st.set_page_config(page_title="Chat with PDF", layout="wide")
# # st.markdown("""
# #     <style>
# #         .block-container {
# #             padding: 2rem 2rem 2rem 2rem;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # st.header("📄 Chat with PDF using Gemini 1.5 Pro")

# # # Session state for chat history
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # 📥 Upload PDF
# # pdf = st.file_uploader("Upload your PDF", type="pdf")

# # if pdf is not None:
# #     # 📖 Read PDF
# #     reader = PdfReader(pdf)
# #     raw_text = ""
# #     for page in reader.pages:
# #         content = page.extract_text()
# #         if content:
# #             raw_text += content

# #     # 📋 Display summary of PDF
# #     with st.expander("📑 PDF Summary"):
# #         llm_summary = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.2)
# #         summary_prompt = f"Summarize the following document content in a few bullet points:\n{raw_text[:3000]}"
# #         summary = llm_summary.invoke(summary_prompt)
# #         st.write(summary.content)

# #     # 🔍 Split text into chunks
# #     splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
# #     chunks = splitter.split_text(raw_text)

# #     # 🧬 Gemini Embeddings
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# #     # 📚 Create FAISS vector store
# #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

# #     # 🤖 Load Gemini LLM (1.5 Pro)
# #     llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

# #     # 🔗 Create QA chain
# #     chain = load_qa_chain(llm, chain_type="stuff")

# #     # ❓ Get user question
# #     question = st.text_input("Ask a question about the PDF:")

# #     if question:
# #         with st.spinner("Thinking..."):
# #             docs = vectorstore.similarity_search(question)
# #             response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
# #             answer = response["output_text"]
# #             st.subheader("💬 Answer")
# #             st.write(answer)

# #             # # 🔎 Show relevant snippets
# #             # with st.expander("🔍 Source Document Snippets"):
# #             #     for doc in docs:
# #             #         st.markdown(
# #             #             f"""
# #             #             <div style='background-color:#f0f0f0; color:#000000; padding:10px; border-radius:8px; margin-bottom:10px; max-height:200px; overflow-y:auto; white-space: pre-wrap;'>
# #             #             {doc.page_content}
# #             #             </div>
# #             #             """,
# #             #             unsafe_allow_html=True
# #             #         )

# #             # Save in chat history
# #             st.session_state.chat_history.append((question, answer))

# #             # Download option
# #             st.download_button("📥 Download Answer", answer, file_name="response.txt")

# #     # Display chat history in sidebar
# #     with st.sidebar:
# #         st.title("🕘 Chat History")
# #         if st.session_state.chat_history:
# #             for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
# #                 st.markdown(f"**Q{i}:** {q}")
# #                 st.markdown(f"**A{i}:** {a}")
# #                 st.markdown("---")  # Add separation

# #         # Clear chat button
# #         if st.button("🗑️ Clear Chat"):
# #             st.session_state.chat_history = []


# # MAIN CODE STARTS HERE
# import streamlit as st
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from collections import Counter
# import tempfile
# import base64
# import io

# # ✅ Set API key as environment variable for LangChain
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCMFd8JSDYzEpPaQnQzt6yV8wkaP8q4q9U"  # 🔐 Replace with your actual API key
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# # 🧠 App title
# st.set_page_config(page_title="Chat with PDF", layout="wide")
# st.markdown("""
#     <style>
#         .block-container {
#             padding: 2rem 2rem 2rem 2rem;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.header("📄 Chat with PDF using Gemini 1.5 Pro")

# # Session state for chat history and bookmarks
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "bookmarks" not in st.session_state:
#     st.session_state.bookmarks = []

# # 📥 Upload PDF
# pdf = st.file_uploader("Upload your PDF", type="pdf")

# if pdf is not None:
#     # 📖 Read PDF
#     reader = PdfReader(pdf)
#     pages = reader.pages
#     raw_text = ""
#     page_texts = []
#     for page in pages:
#         content = page.extract_text()
#         if content:
#             raw_text += content
#             page_texts.append(content)

#     # 📋 PDF Summary
#     with st.expander("📑 PDF Summary"):
#         llm_summary = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.2)
#         summary_prompt = f"Summarize the following document content in a few bullet points:\n{raw_text[:3000]}"
#         summary = llm_summary.invoke(summary_prompt)
#         st.write(summary.content)

#     # 📊 Word Cloud and Most Frequent Words
#     with st.expander("📈 Text Analysis - Word Cloud & Frequency"):
#         words = raw_text.split()
#         freq = Counter(words)
#         most_common = freq.most_common(10)
#         st.write("**Top 10 Frequent Words:**")
#         st.table(most_common)

#         wc = WordCloud(width=800, height=400, background_color='white').generate(raw_text)
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.imshow(wc, interpolation='bilinear')
#         ax.axis("off")
#         st.pyplot(fig)

#     # 🔍 Split text into chunks
#     splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
#     chunks = splitter.split_text(raw_text)

#     # 🧬 Gemini Embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # 📚 Create FAISS vector store
#     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

#     # 🤖 Load Gemini LLM (1.5 Pro)
#     llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

#     # 🔗 Create QA chain
#     chain = load_qa_chain(llm, chain_type="stuff")

#     # 🗂️ Page-wise view
#     with st.expander("📄 View PDF by Page"):
#         selected_page = st.slider("Select Page", 1, len(page_texts), 1)
#         st.text_area("Page Content", value=page_texts[selected_page-1], height=200)

#     # ❓ Multi-Question Input
#     st.subheader("❓ Ask One or More Questions")
#     questions_input = st.text_area("Ask your questions (one per line):")

#     if questions_input:
#         questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
#         with st.spinner("Thinking..."):
#             for question in questions:
#                 docs = vectorstore.similarity_search(question)
#                 response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
#                 answer = response["output_text"]
#                 st.markdown(f"**Q: {question}**")
#                 st.write(answer)

#                 # Bookmark Option
#                 if st.button(f"⭐ Bookmark this answer", key=question):
#                     st.session_state.bookmarks.append((question, answer))

#                 # Save in chat history
#                 st.session_state.chat_history.append((question, answer))

#     # 📥 Download chat history
#     if st.session_state.chat_history:
#         history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
#         st.download_button("📤 Download Full Chat", history_text, file_name="chat_history.txt")

#     # 📚 Bookmarked Answers
#     with st.sidebar:
#         st.title("⭐ Bookmarked Answers")
#         if st.session_state.bookmarks:
#             for i, (q, a) in enumerate(reversed(st.session_state.bookmarks), 1):
#                 st.markdown(f"**Q{i}:** {q}")
#                 st.markdown(f"**A{i}:** {a}")
#                 st.markdown("---")

#         st.title("🕘 Chat History")
#         if st.session_state.chat_history:
#             for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
#                 st.markdown(f"**Q{i}:** {q}")
#                 st.markdown(f"**A{i}:** {a}")
#                 st.markdown("---")

#         # Clear chat button
#         if st.button("🗑️ Clear Chat"):
#             st.session_state.chat_history = []
#             st.session_state.bookmarks = []

#new code gemini
# import streamlit as st
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from collections import Counter
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(page_title="Chat with PDF", layout="wide")
# st.header("📄 Chat with PDF using Gemini 1.5 Pro")

# # ---------------- SIDEBAR (API KEY) ----------------
# with st.sidebar:
#     st.title("🔑 Configuration")
#     api_key = st.text_input("Enter Google API Key", type="password")
#     st.info("Get your API key from Google AI Studio")

# if not api_key:
#     st.warning("Please enter your Google API key to continue.")
#     st.stop()

# os.environ["GOOGLE_API_KEY"] = api_key
# genai.configure(api_key=api_key)

# # ---------------- SESSION STATE ----------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "bookmarks" not in st.session_state:
#     st.session_state.bookmarks = []

# # ---------------- UPLOAD PDF ----------------
# pdf = st.file_uploader("Upload a PDF", type="pdf")

# if pdf:
#     reader = PdfReader(pdf)
#     raw_text = ""
#     page_texts = []

#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             raw_text += text
#             page_texts.append(text)

#     # ---------------- PDF SUMMARY ----------------
#     with st.expander("📑 PDF Summary"):
#         llm_summary = ChatGoogleGenerativeAI(
#             model="models/gemini-1.5-pro",
#             temperature=0.2
#         )
#         prompt = f"Summarize the following document in bullet points:\n{raw_text[:3000]}"
#         summary = llm_summary.invoke(prompt)
#         st.write(summary.content)

#     # ---------------- WORD CLOUD ----------------
#     with st.expander("📊 Text Analysis"):
#         words = raw_text.split()
#         freq = Counter(words)
#         st.write("**Top 10 frequent words:**")
#         st.table(freq.most_common(10))

#         wc = WordCloud(width=800, height=400, background_color="white").generate(raw_text)
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.imshow(wc, interpolation="bilinear")
#         ax.axis("off")
#         st.pyplot(fig)

#     # ---------------- TEXT SPLITTING ----------------
#     splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = splitter.split_text(raw_text)

#     # ---------------- EMBEDDINGS + VECTOR STORE ----------------
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

#     # ---------------- QA LLM ----------------
#     llm = ChatGoogleGenerativeAI(
#         model="models/gemini-1.5-pro",
#         temperature=0.3
#     )

#     chain = load_qa_chain(llm, chain_type="stuff")

#     # ---------------- PAGE VIEW ----------------
#     with st.expander("📄 View PDF by Page"):
#         page_no = st.slider("Select Page", 1, len(page_texts), 1)
#         st.text_area("Page Content", page_texts[page_no - 1], height=200)

#     # ---------------- QUESTIONS ----------------
#     st.subheader("❓ Ask Questions")
#     questions = st.text_area("One question per line")

#     if questions:
#         for q in questions.split("\n"):
#             q = q.strip()
#             if not q:
#                 continue

#             docs = vectorstore.similarity_search(q)
#             response = chain(
#                 {"input_documents": docs, "question": q},
#                 return_only_outputs=True
#             )

#             answer = response["output_text"]
#             st.markdown(f"**Q:** {q}")
#             st.write(answer)

#             if st.button("⭐ Bookmark", key=q):
#                 st.session_state.bookmarks.append((q, answer))

#             st.session_state.chat_history.append((q, answer))

#     # ---------------- DOWNLOAD CHAT ----------------
#     if st.session_state.chat_history:
#         chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
#         st.download_button("📥 Download Chat", chat_text, "chat_history.txt")

# # ---------------- SIDEBAR HISTORY ----------------
# with st.sidebar:
#     st.title("⭐ Bookmarks")
#     for q, a in st.session_state.bookmarks[::-1]:
#         st.markdown(f"**Q:** {q}")
#         st.markdown(f"**A:** {a}")
#         st.markdown("---")

#     st.title("🕘 Chat History")
#     for q, a in st.session_state.chat_history[::-1]:
#         st.markdown(f"**Q:** {q}")
#         st.markdown(f"**A:** {a}")
#         st.markdown("---")

#     if st.button("🗑️ Clear All"):
#         st.session_state.chat_history = []
#         st.session_state.bookmarks = []


# groq code 
import streamlit as st
import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

from collections import Counter


# ---------------- LOAD ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please add it to .env file")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with PDF (Free RAG)", layout="wide")
st.header("📄 Chat with PDF using Free RAG (Groq + Open Models)")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

# ---------------- UPLOAD PDF ----------------
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    raw_text = ""
    page_texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
            page_texts.append(text)

    # ---------------- SUMMARY ----------------
    with st.expander("📑 PDF Summary"):
        summary_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.2
        )

        prompt = f"Summarize the following document in bullet points:\n{raw_text[:3000]}"
        summary = summary_llm.invoke(prompt)
        st.write(summary.content)

    # ---------------- TEXT ANALYSIS ----------------
    # with st.expander("📊 Text Analysis"):
    #     words = raw_text.split()
    #     freq = Counter(words)

    #     st.write("**Top 10 Frequent Words**")
    #     st.table(freq.most_common(10))

    #     wc = WordCloud(
    #         width=800,
    #         height=400,
    #         background_color="white"
    #     ).generate(raw_text)

    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     ax.imshow(wc, interpolation="bilinear")
    #     ax.axis("off")
    #     st.pyplot(fig)

    # ---------------- SPLITTING ----------------
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    # ---------------- EMBEDDINGS (FREE + OFFLINE) ----------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # ---------------- QA MODEL ----------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    # ---------------- PAGE VIEW ----------------
    with st.expander("📄 View PDF by Page"):
        page_no = st.slider("Select Page", 1, len(page_texts), 1)
        st.text_area(
            "Page Content",
            page_texts[page_no - 1],
            height=250
        )

    # ---------------- QUESTIONS ----------------
    st.subheader("❓ Ask Questions (one per line)")
    questions = st.text_area("Enter your questions")

    if questions:
        for q in questions.split("\n"):
            q = q.strip()
            if not q:
                continue

            docs = vectorstore.similarity_search(q)
            response = chain(
                {"input_documents": docs, "question": q},
                return_only_outputs=True
            )

            answer = response["output_text"]

            st.markdown(f"**Q:** {q}")
            st.write(answer)

            if st.button("⭐ Bookmark", key=q):
                st.session_state.bookmarks.append((q, answer))

            st.session_state.chat_history.append((q, answer))

    # ---------------- DOWNLOAD CHAT ----------------
    if st.session_state.chat_history:
        chat_text = "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]
        )
        st.download_button(
            "📥 Download Chat History",
            chat_text,
            "chat_history.txt"
        )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⭐ Bookmarks")
    for q, a in st.session_state.bookmarks[::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    st.title("🕘 Chat History")
    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    if st.button("🗑️ Clear All"):
        st.session_state.chat_history = []
        st.session_state.bookmarks = []
