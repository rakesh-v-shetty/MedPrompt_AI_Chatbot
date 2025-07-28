## MedPrompt AI Chatbot: Advanced Medical Assistant

MedPrompt AI is an advanced medical assistant chatbot designed to provide information based on medical contexts. It leverages Retrieval-Augmented Generation (RAG) by combining a FAISS index of medical textbooks with two powerful Large Language Models (LLMs), Google's Gemini and Mistral AI, to generate comprehensive, accurate, and empathetic responses.

### ✨ Features

  * **Dual LLM Integration:** Utilizes both Gemini 1.5 Flash and Mistral Tiny models for robust and diverse medical insights.
  * **Retrieval-Augmented Generation (RAG):** Integrates with a FAISS vector database to retrieve relevant information from pre-processed medical textbooks, ensuring responses are grounded in factual knowledge.
  * **Context Management:** Employs conversation summarization to maintain context over long dialogues, preventing token limit issues while retaining critical medical details.
  * **Intuitive Web Interface:** A modern, responsive chat interface built with Flask and Tailwind CSS, featuring:
      * Real-time message display.
      * Typing indicators for a dynamic user experience.
      * Voice input via Web Speech API.
      * Text-to-Speech (TTS) output for AI responses.
      * Quick action suggestions for guided conversations.
  * **Structured Medical Responses:** AI responses are formatted with headings, bolded key terms, and lists for enhanced readability and clarity.

### 🚀 Setup and Installation

Follow these steps to get the MedPrompt AI Chatbot up and running on your local machine.

#### 1\. Prerequisites

  * Python 3.8+
  * `pip` (Python package installer)
  * Access to Google Gemini API Key
  * Access to Mistral AI API Key
  * Google Colab (recommended for data processing)
  * Google Drive (for storing pre-processed data and textbooks)

#### 2\. Environment Variables

Create a `.env` file in the root directory of your project and add your API keys:

```dotenv
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
```

**Note:** The provided `.env` file contains placeholder keys. Replace them with your actual API keys.

#### 3\. Data Preparation

The chatbot relies on pre-processed medical textbook data and a FAISS index. You can generate these files using the `model.ipynb` Jupyter notebook, preferably on Google Colab with a T4 GPU for faster processing.

1.  **Download NLTK Data:** The `model.ipynb` notebook handles the download of necessary NLTK tokenizers.
2.  **Mount Google Drive:** The notebook will guide you to mount your Google Drive. Ensure your medical textbooks (`.txt` files) are placed in a folder named `Medical_Textbooks` within your Google Drive's `MyDrive` directory (`/content/drive/MyDrive/Medical_Textbooks`).
3.  **Run `model.ipynb`:** Execute all cells in `model.ipynb` to:
      * Install required libraries (`sentence-transformers`, `faiss-cpu`, `nltk`, `numpy`, `spicy`).
      * Process the text files from your `Medical_Textbooks` directory into chunks.
      * Generate embeddings for these chunks using `BAAI/bge-base-en-v1.5` model.
      * Build a FAISS index from the embeddings.
      * Save the processed chunks (`medical_text_chunks.pkl`) and the FAISS index (`medical_faiss_index.bin`) to `/content/drive/MyDrive/medprompt_ai_data/`.

Alternatively, you can obtain the dataset and pre-processed files from the following sources:

  * **Dataset (MedQA-USMLE):** [Kaggle - MedQA-USMLE](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle)
  * **Pre-processed Files & Textbooks:** [Google Drive Folder](https://drive.google.com/drive/folders/1qzZlyx77mZ5Dq64Dwgz_7GUkgqQMsyg-?usp=drive_link) (Look for `medprompt_ai_data` folder for the `.bin` and `.pkl` files and `env.zip` for `.env` file).

Ensure that `medical_text_chunks.pkl` and `medical_faiss_index.bin` are located in a `medprompt_ai_data` directory in the root of your project (where `app.py` resides).

#### 4\. Install Dependencies

Install the Python packages listed in `app.py`:

```bash
pip install -r requirements.txt
# (If you don't have a requirements.txt, you can create one by running:
# pip freeze > requirements.txt
# Or manually install:
pip install Flask flask-cors python-dotenv google-generativeai sentence-transformers faiss-cpu numpy requests logging
```

#### 5\. Run the Application

```bash
python app.py
```

The application will typically run on `http://127.0.0.1:5000`. Open this URL in your web browser.

### 💡 Usage

Once the server is running, navigate to `http://127.0.0.1:5000` in your web browser.

  * Type your medical questions into the input field at the bottom.
  * Click the "Send" button or press `Enter` to send your question.
  * The AI will respond with information based on the medical context.
  * Use the microphone icon to input questions via voice.
  * Toggle the speaker icon to enable/disable text-to-speech for AI responses.
  * Click "New Chat" to start a fresh conversation.
  * Explore quick action suggestions at the bottom of the chat for guided prompts.

### 📂 Project Structure

```
MedPrompt AI Chatbot/
├── .env                  # Environment variables for API keys
├── app.py                # Main Flask application logic
├── Drive.txt             # Information about Google Drive assets
├── imp.md                # Markdown file detailing future improvements
├── model.ipynb           # Jupyter notebook for data preprocessing and FAISS index creation
└── templates/
    └── index.html        # Frontend HTML for the chat interface
```

### 🔮 Future Improvements

Several areas have been identified for future enhancements to improve the chatbot's functionality, user experience, and production readiness:

#### Category 1: Conversational Intelligence & Accuracy

1.  **User Feedback Mechanism:** Implement "Thumbs Up/Down" icons on AI responses to gather direct feedback on answer quality. This data can be used to refine prompts, RAG context retrieval, and model performance.
2.  **Advanced Retrieval-Augmented Generation (RAG):**
      * **Re-ranking:** Use a sophisticated model (e.g., cross-encoder) to re-rank the initial FAISS results for more semantic relevance.
      * **Query Expansion:** Employ an LLM to rewrite user queries for broader and more effective document retrieval.
3.  **Proactive "Next Step" Suggestions:** Modify the final synthesis prompt to generate and display relevant follow-up questions at the end of each AI response, creating a more guided conversation flow.

#### Category 2: UI/UX and Frontend Polish

1.  **Full Markdown and Table Support:** Replace the custom `formatAIMessage` function with a robust third-party library like `marked.js` to render complex Markdown, especially tables, which are crucial for presenting medical information clearly.
2.  **Persistent Conversation History Sidebar:** Add a sidebar to list past conversations, allowing users to easily access and continue previous discussions. This requires backend changes to store conversation titles and histories in a database.

#### Category 3: Security & Production Readiness

1.  **Containerize the Application with Docker:** Create a `Dockerfile` to ensure consistent deployment across various environments, simplifying dependency management and scaling.
2.  **PII (Personally Identifiable Information) Redaction:** Implement automatic detection and removal of personal information from user input before processing or logging, crucial for privacy and HIPAA compliance. This can be done using regular expressions or dedicated services like AWS Comprehend PII Detection or Google Cloud DLP API.
3.  **Implement API Rate Limiting:** Utilize a Flask extension like `Flask-Limiter` to protect the `/ask` endpoint from abuse, control API costs, and ensure fair usage.
4.  **Persistent and Scalable Conversation History:** Replace the in-memory Python dictionary for conversation histories with a persistent database like Redis or MongoDB to prevent data loss on server restarts and ensure scalability for multiple users.
5.  **Asynchronous Task Handling:** Refactor the `/ask` endpoint to make external API calls (Gemini, Mistral) asynchronous using `asyncio` and `aiohttp` to reduce user wait times and prevent request timeouts. Running with an ASGI server (e.g., Gunicorn with Uvicorn worker) would be necessary for this.

### ⚠️ Disclaimer

**DISCLAIMER:** This AI tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider.

### 📝 License

This project is open-sourced under the [MIT License](https://www.google.com/search?q=LICENSE) (placeholder, consider adding a https://www.google.com/search?q=LICENSE file).

### 📞 Contact

For any questions or suggestions, please open an issue in the GitHub repository or contact [Your Name/Email].
