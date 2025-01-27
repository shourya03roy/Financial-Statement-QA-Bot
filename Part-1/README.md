# Financial Terms QA System with Retrieval-Augmented Generation (RAG)

## Overview

This project develops a Question Answering (QA) system to process Profit & Loss (P&L) data extracted from PDF documents. The model integrates multiple tools such as `pdfplumber` for PDF data extraction, `SentenceTransformers` for document embedding, `FAISS` for efficient retrieval, and `LangChain` for generative responses. The system answers user queries regarding financial terms and metrics present in the P&L data.

## Model Architecture

### 1. **Data Extraction**
   - **Tool**: `pdfplumber`
   - **Approach**: P&L data is extracted from a PDF file by parsing the pages containing relevant financial information. The function `extract_pl_data(pdf_path, start_page, end_page)` is used to retrieve the text from the specified range of pages.
   - **Manual Parsing**: A sample financial statement is parsed to extract metrics like "Revenue from operations," "Other income," and "Profit for the period." This data is manually mapped and stored as a DataFrame.

### 2. **Data Preprocessing and Embedding**
   - **Tool**: `SentenceTransformers` for embeddings
   - **Approach**: The financial metrics are preprocessed into a suitable text format (`Metric: 2024: value, 2023: value`). These processed texts are then embedded using the pre-trained model `"sentence-transformers/all-MiniLM-L6-v2"`. The embeddings represent the textual data in a high-dimensional vector space, capturing the semantic meaning of the financial terms.
   
### 3. **Vector Storage and Retrieval**
   - **Tool**: `FAISS`
   - **Approach**: A FAISS index is created using the generated embeddings. This allows for efficient and scalable similarity search, enabling the retrieval of the most relevant documents for a given query. The embeddings are added to a `FAISS` index, which allows retrieval by vector similarity.
   - **Document Storage**: Each document's embedding is linked to its original text via a simple index-to-docstore mapping, where the text can be retrieved by its index.

### 4. **Generative Response with LangChain**
   - **Tool**: `LangChain` with Hugging Face integration
   - **Approach**: LangChain is used to integrate a pre-trained generative model (`google/flan-t5-base`) from Hugging Face. The `RetrievalQA` chain is used to combine the retrieval process with the generative model. Given a query, the system first retrieves the most relevant documents from the FAISS vector store and then uses the `HuggingFaceHub` LLM to generate a response.
   - **QA Process**: The user’s query is passed to the `qa_chain.run(query)` function, which performs retrieval and generates the corresponding answer based on the retrieved financial data.

## Challenges and Solutions

### 1. **Data Extraction from PDFs**
   - **Challenge**: The structure of P&L data in PDFs can vary significantly, requiring robust parsing techniques.
   - **Solution**: A simple but effective text extraction method was used to pull content from specific pages. However, for more structured documents, more advanced parsing techniques such as regular expressions could be applied for better accuracy.

### 2. **Embeddings and Vectorization**
   - **Challenge**: Ensuring that financial terms and metrics are properly embedded for accurate retrieval.
   - **Solution**: The `sentence-transformers/all-MiniLM-L6-v2` model was chosen due to its efficiency in embedding short texts while retaining semantic meaning. The data was preprocessed to include relevant details for each metric to ensure the embeddings were meaningful.

### 3. **Vector Retrieval Efficiency**
   - **Challenge**: Efficient retrieval from a growing set of documents.
   - **Solution**: FAISS was selected to index the embeddings, which optimizes search queries by utilizing vector similarity. The use of `FAISS.IndexFlatL2` ensures fast similarity search even for a large number of documents.

### 4. **Integrating Generative Model for Responses**
   - **Challenge**: Providing accurate and context-aware responses based on retrieved financial data.
   - **Solution**: The retrieval process was combined with a generative model (`google/flan-t5-base`), which was fine-tuned to generate concise and relevant answers to the financial queries.

## Deployment Instructions

1. **Setting Up the Environment**:
   - Install the required dependencies using `pip`:
     pip install pdfplumber pandas sentence-transformers faiss-cpu langchain

2. **Set Hugging Face API Token**:
   - Obtain a Hugging Face API token by creating an account on [Hugging Face](https://huggingface.co/).
   - Set the token in your environment:
     import getpass
     API_token = getpass.getpass("Enter Hugging Face API token: ")
     os.environ["HUGGINGFACEHUB_API_TOKEN"] = API_token

3. **Upload and Extract Data from PDFs**:
   - Upload your PDF file containing the Profit & Loss statement.
   - Use the `extract_pl_data` function to extract the text from the desired pages of the document.

4. **Preprocess Data**:
   - Convert the extracted P&L data into a structured format (e.g., DataFrame) and preprocess it for embedding.

5. **Set Up the FAISS Vector Store**:
   - Use the `SentenceTransformer` model to embed the data, and then create a FAISS index for fast retrieval.

6. **Deploy the QA System**:
   - Initialize the `HuggingFaceHub` model and use the `RetrievalQA` chain to set up the QA system.
   - Test with queries to ensure the system responds appropriately.

## Usage Instructions

1. **Run Queries**: 
   Once the system is set up, you can run queries like:
   - "What is the revenue from operations for 2024?"
   - "What is the profit for the period in 2023?"
   - "How much other income was earned in 2024?"

   The system will retrieve the most relevant documents and generate answers based on the retrieved information.

2. **Expand Dataset**:
   You can expand the dataset by adding more financial metrics to the DataFrame and re-processing the data through the same pipeline to update the FAISS index.

## Future Improvements

1. **Advanced PDF Parsing**:
   - Implement a more robust parser for structured data extraction (e.g., tables) using libraries like `camelot-py` or custom heuristics.
   
2. **Model Fine-tuning**:
   - Fine-tune the generative model to better handle financial-specific queries.

3. **Scalability**:
   - Consider scaling the system for larger documents and datasets by optimizing FAISS indexing techniques and using more powerful hardware.

4. **Integration with Other Data Sources**:
   - Expand the system to work with other types of documents or data sources, such as income statements, balance sheets, or other financial reports.
