# Our Methodology

At the heart of this project lies a sophisticated approach designed to mimic and enhance human-like reading and comprehension. We move beyond the limitations of traditional keyword searching (like CTRL+F), which often fails to grasp the user's true intent, and instead embrace a semantic understanding of text. Our methodology is a multi-stage process that intelligently deconstructs, analyzes, and ranks information to deliver highly relevant insights.

## 1. Intelligent Document Segmentation

First, we recognize that documents are not just a flat sequence of words; they have a deliberate structure. Instead of arbitrarily splitting text by page or paragraph, our system employs the PyMuPDF library to perform intelligent segmentation. It analyzes the visual and stylistic properties of the text, such as font size, weight (bolding), and layout, to identify natural section headings. This allows us to break down a document into coherent, self-contained logical sections—much like a person would distinguish between different chapters or topics. This ensures that the context of the information is preserved within each chunk we analyze.

## 2. Translating Words into Meaning (Embeddings)

Once the document is segmented, we get to the core of our semantic engine. We use a state-of-the-art Sentence-BERT (S-BERT) model called all-MiniLM-L6-v2. This model's job is to act as a universal translator, converting any piece of text—whether it's the user's query or a document section—into a rich numerical representation known as an embedding.

Think of an embedding as a digital fingerprint for meaning. This "fingerprint" is a high-dimensional vector that captures the deep semantic properties of the text. Because the model is trained on a massive amount of language, texts with similar meanings will have mathematically similar embeddings, regardless of whether they use the exact same words.

## 3. Finding the Perfect Match with Cosine Similarity

With embeddings generated for both the user's query and every document section, the final step is to find the best matches. We achieve this using a mathematical measure called Cosine Similarity. This technique calculates the angle between two embedding vectors. A smaller angle (resulting in a cosine similarity score closer to 1.0) indicates that the vectors are closely aligned, meaning their underlying texts share a strong contextual and semantic relationship.

By computing this score for every section against the query, we can quantitatively rank how relevant each section is to the user's task. The highest-scoring sections are the ones that the model has determined are the most meaningful matches, providing a precise and reliable way to pinpoint critical information.
