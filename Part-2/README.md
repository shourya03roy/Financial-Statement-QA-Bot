### Documentation for Financial QA Bot Application  

---

## **Introduction**  
The Financial QA Bot is a Streamlit-based interactive web application designed to help users query financial data from PDF documents containing Profit & Loss (P&L) statements. The application uses advanced natural language processing (NLP) and machine learning models to process the document, extract meaningful insights, and provide answers to user queries in real-time.

---

## **Features**  
- Upload PDF documents containing financial data.  
- Extract and preprocess P&L data for embedding.  
- Perform interactive financial queries in natural language.  
- Use state-of-the-art models for accurate and detailed responses.  

---

## **User Guide**  

### **Uploading Documents**  
1. Open the application in your browser (e.g., using the Streamlit URL or deployed link).  
2. Click the "Upload a P&L Statement PDF" section.  
3. Upload a PDF file containing the Profit & Loss data.  
   - **Supported Formats**: `.pdf`  
   - Ensure the document includes clear tabular data or structured financial content.  

### **Asking Questions**  
1. Once the PDF is uploaded, the application will extract and preprocess the P&L data.  
2. You will see the extracted data displayed in a table for your reference.  
3. Enter your query in the "Enter your query" input box.  
   - Examples of valid queries:
     - "What is the revenue from operations for 2024?"
     - "How much profit was made in 2023?"
     - "Compare the revenue from operations for 2024 and 2023."
4. The bot will respond with the answer based on the uploaded document.  

### **Interpreting the Bot’s Responses**  
- The bot generates responses by retrieving and processing relevant information from the uploaded document.  
- For example:  
  - **Query**: "What is the revenue from operations for 2024?"  
    **Response**: "The revenue from operations for 2024 is 37,923."  
  - **Query**: "What is the profit for the period in 2023?"  
    **Response**: "The profit for the period in 2023 is 6,134."  

---

## **Examples of Interactions**  

### **Example 1**  
**Query**: "What are the financial metrics for 2024?"  
**Response**:  
- Revenue from operations: 37,923  
- Other income, net: 2,729  
- Profit for the period: 7,975  

---

### **Example 2**  
**Query**: "How much other income was earned in 2024?"  
**Response**: "The other income earned in 2024 is 2,729."  

---

### **Example 3**  
**Query**: "What is the percentage increase in profit for the period from 2023 to 2024?"  
**Response**: "The profit for the period increased from 6,134 in 2023 to 7,975 in 2024, which is an approximate increase of 30%."  

---

## **Approach Explanation**  

### **Data Extraction**  
- The application uses the `pdfplumber` library to extract text from uploaded PDF documents.  
- The extracted text is manually parsed to extract key financial metrics, which are converted into a structured format (e.g., tabular data).  

### **Preprocessing and Vectorization**  
- Each financial metric is formatted into descriptive text (e.g., "Revenue from operations: 2024: 37,923, 2023: 37,441").  
- The `sentence-transformers/all-MiniLM-L6-v2` model is used to embed these descriptions into high-dimensional vectors.

### **Retrieval and QA**  
- **FAISS (Facebook AI Similarity Search)**:  
  - Stores and retrieves embeddings efficiently using cosine similarity or L2 distance.  
- **LangChain Integration**:  
  - Uses a retrieval-based QA chain to match user queries with relevant financial metrics.  
  - The `google/flan-t5-base` model generates natural language responses based on retrieved embeddings.  

---

## **Deployment Instructions**  

1. **Environment Setup**  
   - Install Python (3.8 or higher).  
   - Install required libraries using the command:  
     pip install streamlit pandas pdfplumber sentence-transformers faiss-cpu langchain
     
2. **Hugging Face API Token**  
   - Create an account on [Hugging Face](https://huggingface.co/) and generate an API token.  
   - Set the token as an environment variable:  
     ```python
     os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<your_huggingface_api_token>"

3. **Run the Application**  
   - Save the provided code into a Python file (e.g., `financial_qa_bot_part2.py`).  
   - Run the Streamlit app:  
     streamlit run financial_qa_bot_part2.py
   - Access the app in your browser at `http://localhost:8501`.

4. **Deployment Options**  
   - **Cloud Hosting**: Deploy the app on platforms like Streamlit Community Cloud, AWS, or Heroku.  
   - **Docker**: Package the app into a Docker container for easy deployment.

---

## **Challenges and Solutions**  

### **Challenge 1**: Handling complex or unstructured financial data.  
**Solution**: Focus on structured data extraction for key metrics. Improve parsing by manually simulating P&L extraction during initial phases.  

### **Challenge 2**: Embedding high-dimensional data for retrieval.  
**Solution**: Used the `sentence-transformers/all-MiniLM-L6-v2` model for efficient embeddings and FAISS for scalable retrieval.  

### **Challenge 3**: Generating accurate, context-aware responses.  
**Solution**: Integrated LangChain’s retrieval-based QA pipeline with `google/flan-t5-base` for high-quality generative responses.  

---

## **Usage Instructions**  

1. Upload a PDF document containing structured P&L data.  
2. Ask financial queries in natural language.  
3. View extracted financial data and bot responses on the app interface.  
4. Use the examples provided for inspiration or to test the bot’s functionality.

---
