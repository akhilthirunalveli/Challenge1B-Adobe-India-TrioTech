<div align="left">
  <img src="https://github.com/akhilthirunalveli/akhilthirunalveli/blob/main/assets/1%20(2).png" alt="App Demo" width="1000"/>
  <img src="https://github.com/akhilthirunalveli/akhilthirunalveli/blob/main/3(1).png" alt="App Demo" width="1000"/>
</div>

-----

## Our Approach

Our system tackles the challenge of information overload in dense documents. Instead of relying on simple keyword matching, which can be brittle and miss context, we employ a *semantic understanding* approach.

At its core, the system uses a powerful *Sentence-BERT (S-BERT)* model to convert both the user's query (defined by their persona and job_to_be_done) and sections of the source documents into high-dimensional vectors (embeddings). By calculating the *cosine similarity* between these vectors, we can find document sections that are most semantically relevant to the user's goal, even if they don't share the exact same words. This allows for a more intuitive and context-aware information retrieval process.

-----

## Step-by-step Strategy

The pipeline executes a clear, multi-stage strategy to ensure accurate and relevant results:

1.  *🧠 Configuration & Query Formulation:* The system first reads input.json to understand the user's role and the specific task. This information is combined into a focused query that guides the search.
2.  *📄 Logical Document Sectioning:* Each PDF is processed using the PyMuPDF library. Instead of treating pages as the primary unit, the script analyzes document structure (font size, bolding) to identify headings and group the text into complete, logical sections.
3.  *🔍 Semantic Encoding:* The pre-trained all-MiniLM-L6-v2 model is loaded. This model then encodes the user's query and every extracted document section into meaningful numerical embeddings.
4.  *🥇 Relevance Ranking:* The system computes the cosine similarity score between the query embedding and all section embeddings. Sections are then ranked from highest to lowest score, indicating their relevance to the task.
5.  *⚙ Filtering & Selection:* To provide diverse and concise results, the top sections are selected based on parameters like top_k_sections and max_sections_per_document, preventing a single document from dominating the results.
6.  *📝 Output Generation:* The final ranked list of section titles, along with their page numbers and full text content, is compiled and saved to output.json for review and downstream use.

-----

## Libraries & Technologies Used

  * *Backend:* Python
  * *NLP / Semantic Search:* sentence-transformers
  * *AI Model:* all-MiniLM-L6-v2
  * *PDF Processing:* PyMuPDF (fitz)
  * *Containerization:* Docker

-----

## Build & Run Instructions

Follow these steps to run the application locally.

1.  *Clone the repository:*
    bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    
2.  *Install dependencies:*
    bash
    pip install -r requirements.txt
    
3.  *Download the AI Model:* (This step requires internet)
    bash
    python download_model.py
    
4.  *Add your documents:* Place your PDF files inside the /pdfs folder.
5.  *Configure your task:* Edit the input.json file to define your persona, task, and target filenames.
6.  *Run the script:*
    bash
    python main_docker.py
    
7.  *Check the output:* The results will be saved in output.json.

-----

## Docker Instructions

🐳 For a more robust and portable solution, run the application inside a Docker container. This is the recommended approach as it works offline once the image is built.

Step 1: Build Your Docker Image

From your Challenge1B-Adobe-India-TrioTech directory, build image first. Let's call it adobe-challenge.
Bash
```
docker build -t adobe-challenge .
```
Step 2: For running container with volumes
```
docker run --rm -v "$(pwd)/pdfs:/app/pdfs:ro" -v "$(pwd)/output:/app/output" adobe-challenge
```    

-----

<div align="left">
  <img src="https://github.com/akhilthirunalveli/akhilthirunalveli/blob/main/assets/Adobe%20India%20Hackathon%202025.png" alt="App Demo" width="1000"/>
</div>
