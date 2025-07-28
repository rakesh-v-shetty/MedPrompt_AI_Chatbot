# MedPrompt AI Chatbot: Advanced Medical Assistant ✨🩺

MedPrompt AI is an innovative medical assistant chatbot engineered to deliver **accurate, empathetic, and comprehensive information** based on a robust medical knowledge base. It harnesses the power of cutting-edge large language models (LLMs) and an advanced Retrieval-Augmented Generation (RAG) system, designed to be your intelligent companion for medical inquiries.

-----

## 🚀 Features

  * **Dual LLM Integration:** Seamlessly integrates with both Google's **Gemini-1.5-Flash** and **Mistral AI** models, ensuring diverse perspectives and robust medical insights.
  * **Retrieval-Augmented Generation (RAG):** Built upon a sophisticated RAG architecture that queries a **FAISS vector index** of pre-processed medical text chunks, providing **context-aware and highly accurate** responses grounded in factual data.
  * **Smart Conversation History:** Efficiently manages and intelligently summarizes long chat histories to **maintain crucial conversational context** throughout your interactions.
  * **Streamed Responses:** Delivers AI-generated content in a **real-time streaming fashion**, enhancing user engagement and providing a dynamic chat experience.
  * **Intuitive User Interface:** Features a responsive and modern web interface, developed with Flask, HTML, CSS (Tailwind CSS), and JavaScript, ensuring **seamless and visually appealing interactions**.
  * **Voice Capabilities (STT & TTS):** Enhances accessibility with both **speech-to-text (STT)** for microphone input and optional **text-to-speech (TTS)** for voice output of AI responses.
  * **Crystal Clear Formatting:** AI responses are meticulously formatted using markdown, incorporating **bold text, clear headings (`##`), and organized lists (`*`)** for superior readability and information retention.
  * **Quick Action Buttons:** Offers pre-defined quick action buttons for common medical queries, simplifying the process of **initiating conversations** and exploring topics.

-----

## 🛠️ Technologies Used

  * **Backend Framework:** Flask (Python)
  * **Large Language Models:**
      * Google Gemini (`gemini-1.5-flash`)
      * Mistral AI (`mistral-tiny`)
  * **Embeddings:** `sentence-transformers` (model: `BAAI/bge-base-en-v1.5`)
  * **Vector Database:** FAISS (`faiss-cpu`)
  * **Natural Language Toolkit (NLTK):** For text processing utilities like sentence tokenization.
  * **Data Handling:** `numpy`, `pickle`, `requests`.
  * **Frontend:** HTML, CSS (Tailwind CSS), JavaScript, Font Awesome.
  * **Environment Variables:** `python-dotenv`.

-----

## ⚙️ Setup and Installation

Follow these steps to get the MedPrompt AI Chatbot up and running on your local machine.

### Prerequisites

  * Python 3.9+
  * `pip` (Python package installer)

### 1\. Obtain API Keys 🔑

The application requires API keys for Google Gemini and Mistral AI to function.

  * **Gemini API Key:** Obtain yours from the [Google AI Studio](https://ai.google.dev/).
  * **Mistral API Key:** Get your key from the [Mistral AI Platform](https://console.mistral.ai/).

Once you have your keys, create a `.env` file in the root directory of your project (e.g., `MedPrompt AI Chatbot/.env`) with the following content, replacing the placeholder values with your actual API keys:

```dotenv
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
```

**Security Note:** Remember to add `.env` to your `.gitignore` file to prevent your API keys from being accidentally committed to version control.

### 2\. Dataset Information 📚

The MedPrompt AI Chatbot's RAG system is built upon a knowledge base derived from medical textbooks. While this project uses a custom dataset, the [MedQA-USMLE Dataset on Kaggle](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle) is a relevant public medical dataset that can be referenced or used for similar applications.

### 3\. Set Up Pre-processed Data 🗃️

The chatbot relies on pre-processed medical text chunks and a FAISS index for its RAG capabilities. You have two convenient options to obtain these:

#### Option A: Download Pre-built Data (Recommended) 👇

You can directly download the `medical_faiss_index.bin` and `medical_text_chunks.pkl` files from the following Google Drive link. After downloading, **create a new directory named `medprompt_ai_data` in the root of your project and place these files inside it**:

[MedPrompt AI Data on Google Drive](https://drive.google.com/drive/folders/1qzZlyx77mZ5Dq64Dwgz_7GUkgqQMsyg-?usp=drive_link)

#### Option B: Generate Data Using `model.ipynb` 🧑‍💻

If you prefer to generate the data yourself, you can run the `model.ipynb` notebook on [Google Colab](https://colab.research.google.com/).

  * ***Recommendation:*** Utilize a **T4 GPU runtime** in Colab for significantly faster processing.
  * This notebook will guide you through:
    1.  Installing necessary libraries.
    2.  Downloading NLTK data (handled automatically by the notebook).
    3.  Mounting your Google Drive to access your medical textbooks (you'll need to provide your own `.txt` medical textbook files in a folder like `Medical_Textbooks` on your Google Drive).
    4.  Processing the textbooks, generating embeddings, and building the FAISS index.
    5.  Saving `medical_faiss_index.bin` and `medical_text_chunks.pkl` to your specified `SAVE_DIR` (e.g., `/content/drive/MyDrive/medprompt_ai_data`).

Ensure these generated files are then accessible to your `app.py` (e.g., by placing them in a `medprompt_ai_data` folder in your project directory).

### 4\. Install Dependencies 📦

You can install all the required Python packages using `pip`.

#### Method 1: Using `requirements.txt` (Standard)

If you have a `requirements.txt` file (which lists all project dependencies), simply run:

```bash
pip install -r requirements.txt
```

#### Method 2: Installing Directly via `pip` (Alternate)

Alternatively, you can install each core dependency by listing them in a single `pip install` command:

```bash
pip install flask python-dotenv google-generativeai sentence-transformers faiss-cpu nltk numpy requests scipy
```

#### Method 3: Using a Virtual Environment (Recommended Best Practice) 🐍

It's highly recommended to install dependencies within a virtual environment. This isolates your project's dependencies, preventing conflicts with other Python projects on your system.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**
      * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
      * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
3.  **Install the dependencies within the activated environment:**
    ```bash
    pip install flask python-dotenv google-generativeai sentence-transformers faiss-cpu nltk numpy requests scipy
    ```
    Once installed, you can run your application. To exit the virtual environment, simply type `deactivate`.

### 5\. Run the Application ▶️

Once you have set up your API keys and the data files, you can launch the Flask application:

```bash
python app.py
```

The application will start, and you can usually access it by opening your web browser and navigating to `http://127.0.0.1:5000/`.

**Note:** For production deployment, it is recommended to use a production-ready WSGI server like Gunicorn instead of Flask's built-in development server.

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

-----

**DISCLAIMER:** This AI tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider.
