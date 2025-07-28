# MedPrompt AI Chatbot: Advanced Medical Assistant ✨

MedPrompt AI is an advanced medical assistant chatbot designed to provide information based on medical contexts. It leverages cutting-edge large language models (LLMs) and a robust Retrieval-Augmented Generation (RAG) system to offer **comprehensive, empathetic, and accurate responses** to your medical questions.

-----

## 🚀 Features

  * **Dual LLM Integration:** Utilizes both Google's **Gemini-1.5-Flash** and **Mistral AI** models for robust and diverse medical insights.
  * **Retrieval-Augmented Generation (RAG):** Integrates with a **FAISS vector index** and pre-processed medical text chunks to provide **context-aware and highly accurate** responses.
  * **Smart Conversation History:** Manages chat history efficiently, intelligently summarizing long conversations to **maintain crucial context**.
  * **Streamed Responses:** Delivers AI responses in a **streaming fashion** for a more engaging and real-time user experience.
  * **Intuitive User Interface:** A responsive and modern web interface built with Flask, HTML, CSS (Tailwind CSS), and JavaScript, ensuring **seamless interaction**.
  * **Voice Capabilities (STT & TTS):** Includes both **speech-to-text** (microphone input) and optional **text-to-speech** (voice output) for enhanced accessibility.
  * **Crystal Clear Formatting:** AI responses are beautifully formatted using markdown, including **bold text, headings, and lists**, for superior readability.
  * **Quick Action Buttons:** Offers pre-defined quick action buttons for common queries, making it **easier to start conversations** and explore topics.

-----

## 🛠️ Technologies Used

  * **Backend Framework:** Flask (Python)
  * **Large Language Models:**
      * Google Gemini (`gemini-1.5-flash`)
      * Mistral AI (`mistral-tiny`)
  * **Embeddings:** `sentence-transformers` (model: `BAAI/bge-base-en-v1.5`)
  * **Vector Database:** FAISS (`faiss-cpu`)
  * **Data Processing:** `nltk`, `numpy`, `pickle`
  * **Frontend:** HTML, CSS (Tailwind CSS), JavaScript, Font Awesome
  * **Environment Variables:** `python-dotenv`

-----

## ⚙️ Setup and Installation

Follow these steps to get the MedPrompt AI Chatbot up and running on your local machine.

### 1\. Obtain API Keys 🔑

The application requires API keys for Google Gemini and Mistral AI.

  * **Gemini API Key:** Obtain yours from the [Google AI Studio](https://ai.google.dev/).
  * **Mistral API Key:** Get your key from the [Mistral AI Platform](https://console.mistral.ai/).

Once you have your keys, create a `.env` file in the root directory of your project (e.g., `MedPrompt AI Chatbot/.env`) with the following content, replacing the placeholder values with your actual API keys:

```dotenv
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
```

### 2\. Set Up Pre-processed Data 📚

The chatbot relies on pre-processed medical text chunks and a FAISS index for its RAG capabilities. You have two convenient options to obtain these:

#### Option A: Download Pre-built Data (Recommended) 👇

You can directly download the `medical_faiss_index.bin` and `medical_text_chunks.pkl` files from the following Google Drive link. After downloading, **place these files inside a new directory named `medprompt_ai_data`** in the root of your project:

[MedPrompt AI Data on Google Drive](https://drive.google.com/drive/folders/1qzZlyx77mZ5Dq64Dwgz_7GUkgqQMsyg-?usp=drive_link)

#### Option B: Generate Data Using `model.ipynb` 🧑‍💻

If you prefer to generate the data yourself, you can run the `model.ipynb` notebook on [Google Colab](https://colab.research.google.com/).

  * ***Recommendation:*** Use a **T4 GPU runtime** in Colab for significantly faster processing.
  * This notebook will guide you through:
    1.  Installing necessary libraries.
    2.  Downloading NLTK data.
    3.  Mounting your Google Drive to access your medical textbooks (you'll need to provide your own `.txt` medical textbook files in a folder like `Medical_Textbooks` on your Google Drive).
    4.  Processing the textbooks, generating embeddings, and building the FAISS index.
    5.  Saving `medical_faiss_index.bin` and `medical_text_chunks.pkl` to your specified `SAVE_DIR` (e.g., `/content/drive/MyDrive/medprompt_ai_data`).

Ensure these generated files are then accessible to your `app.py` (e.g., by placing them in a `medprompt_ai_data` folder in your project directory).

### 3\. Install Dependencies 📦

Install all the required Python packages by running the following command in your terminal from the project's root directory:

```bash
pip install -r requirements.txt
```

### 4\. Run the Application ▶️

Once you have set up your API keys and the data files, you can launch the Flask application:

```bash
python app.py
```

The application will start, and you can usually access it by opening your web browser and navigating to `http://127.0.0.1:5000/`.

-----

## 💬 Usage

1.  **Access the Chatbot:** Open your web browser and go to the address where the Flask app is running (e.g., `http://127.0.0.1:5000/`).
2.  **Ask a Question:** Type your medical question into the input box at the bottom of the chat interface.
3.  **Send Message:** Press `Enter` or click the send button (the paper plane icon) to submit your question.
4.  **Receive Response:** The AI will swiftly process your question, retrieve relevant context, and synthesize a comprehensive answer, which will be streamed to the chat interface.
5.  **Voice Input/Output:**
      * Click the **microphone icon** to speak your question directly.
      * Click the **volume icon** to toggle voice output for the AI's responses on or off.
6.  **New Chat:** Click the **"New Chat" button** to start a fresh conversation at any time.
