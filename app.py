import os
from dotenv import load_dotenv
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import logging
import requests
import uuid

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

if not GEMINI_API_KEY or not MISTRAL_API_KEY:
    logging.error("API keys for Gemini or Mistral are not set. Please check your .env file.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# --- Mistral AI Model Configuration ---
MISTRAL_MODEL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_GENERATIVE_MODEL_NAME = "mistral-tiny"
MISTRAL_HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# Data and Model Configuration
DATA_DIRECTORY = 'medprompt_ai_data'
CHUNKS_FILE_PATH = os.path.join(DATA_DIRECTORY, 'medical_text_chunks.pkl')
FAISS_INDEX_FILE_PATH = os.path.join(DATA_DIRECTORY, 'medical_faiss_index.bin')
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# Global variables
text_chunks_db = []
faiss_index = None
embedding_model = None
gemini_model = None

# In-memory storage for conversation histories (consider a database for persistence)
conversation_histories = {}

# Context Management Configuration
MAX_HISTORY_TOKENS = 4096 # Example token limit (approximated by character count for simplicity)
LAST_TURNS_TO_KEEP = 4 # Number of last turns to keep after summarization

def count_tokens_simple(messages):
    """A simple heuristic for token counting (character count).
    For more accurate token counting, a specific tokenizer for the model would be needed.
    """
    total_chars = 0
    for message in messages:
        if 'parts' in message and message['parts']:
            for part in message['parts']:
                if 'text' in part:
                    total_chars += len(part['text'])
    return total_chars

def query_mistral_model(prompt):
    """Sends a prompt to the specified Mistral AI model."""
    try:
        payload = {
            "model": MISTRAL_GENERATIVE_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(MISTRAL_MODEL_API_URL, headers=MISTRAL_HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()

        if result and 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0]:
            return {"success": True, "content": result['choices'][0]['message']['content'].strip()}
        else:
            logging.warning(f"Unexpected response format from Mistral model: {result}")
            return {"success": False, "error": "Could not parse response from Mistral model."}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Mistral model at {MISTRAL_MODEL_API_URL}: {e}")
        return {"success": False, "error": "Could not connect to Mistral AI model. Please check API key or network."}
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing Mistral response: {e}")
        return {"success": False, "error": "Mistral model response format is invalid."}
    except Exception as e:
        logging.error(f"An unexpected error occurred with Mistral model: {e}")
        return {"success": False, "error": f"An unexpected error occurred with Mistral model: {e}"}


def load_preprocessed_data_and_configure_models():
    global text_chunks_db, faiss_index, embedding_model, gemini_model
    logging.info("Loading pre-processed data...")
    try:
        with open(CHUNKS_FILE_PATH, 'rb') as f:
            text_chunks_db = pickle.load(f)

        logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully.")

        logging.info("Loading FAISS index...")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE_PATH)
        logging.info("FAISS index loaded successfully.")

        logging.info("All data and embedding models loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"A required data file was not found: {e}. Please run the data processing script first.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading pre-processed data or models: {e}", exc_info=True)
        exit(1)


    logging.info("Configuring Gemini LLM...")
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Gemini 'gemini-1.5-flash' model configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Gemini LLM: {e}. Please check your GEMINI_API_KEY.")
        exit(1)

load_preprocessed_data_and_configure_models()

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_medprompt_ai():
    user_question = request.json.get('question')
    session_id = request.json.get('session_id')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
        logging.info(f"New session started: {session_id}")

    # Add the current user question to history
    conversation_histories[session_id].append({"role": "user", "parts": [{"text": user_question}]})
    logging.info(f"Session {session_id} - User: {user_question}")

    try:
        # Context Management: Summarize history if too long
        current_history_tokens = count_tokens_simple(conversation_histories[session_id])
        logging.info(f"Session {session_id} - Current history tokens (approx): {current_history_tokens}")

        if current_history_tokens > MAX_HISTORY_TOKENS:
            logging.info(f"Session {session_id} - History exceeding {MAX_HISTORY_TOKENS} tokens, summarizing...")

            # Extract full conversation for summarization
            full_conversation_text = ""
            for turn in conversation_histories[session_id]:
                role = "User" if turn["role"] == "user" else "Assistant"
                full_conversation_text += f"{role}: {turn['parts'][0]['text']}\n"

            summarization_prompt = f"Summarize the following medical conversation concisely, retaining all key medical details and questions. Focus on core topics discussed, symptoms, and advice given: \n\n{full_conversation_text}\n\nSummary:"

            try:
                # Use Gemini Flash for summarization
                summary_response_obj = gemini_model.generate_content(
                    summarization_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=512),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                )
                summary = summary_response_obj.text.strip()
                logging.info(f"Session {session_id} - Summary generated: {summary[:100]}...") # Log first 100 chars
            except Exception as e:
                logging.error(f"Error during summarization: {e}", exc_info=True)
                summary = "Previous conversation context could not be summarized due to an error."

            # Replace history with summary + last N turns
            new_history = [{"role": "system", "parts": [{"text": f"Summary of previous conversation: {summary}"}]}]
            # Ensure we don't try to slice more turns than exist if the history is short
            new_history.extend(conversation_histories[session_id][-LAST_TURNS_TO_KEEP:])
            conversation_histories[session_id] = new_history
            logging.info(f"Session {session_id} - History updated with summary and last {LAST_TURNS_TO_KEEP} turns.")


        # Retrieve context using the embedding model
        question_embedding = embedding_model.encode([user_question]).astype('float32')
        D, I = faiss_index.search(question_embedding, k=5)
        relevant_contexts = [text_chunks_db[idx]['text'] for idx in I[0] if idx < len(text_chunks_db)]
        context_str = "\n".join(relevant_contexts)

        # Create base prompt for both models, incorporating conversation history
        # Note: gemini_conversation_history is now potentially summarized
        gemini_conversation_history = conversation_histories[session_id]

        mistral_prompt_parts = []
        mistral_prompt_parts.append("The following is a medical conversation. Provide a detailed and comprehensive answer based on the context and conversation history.")
        mistral_prompt_parts.append("\n\nMedical Context:\n" + context_str + "\n\n")

        mistral_prompt_parts.append("Conversation History:")
        for turn in gemini_conversation_history: # Iterate through the (potentially summarized) history
            role = "User" if turn["role"] == "user" else "Assistant" if turn["role"] == "model" else "System" # Add system role
            mistral_prompt_parts.append(f"{role}: {turn['parts'][0]['text']}")
        mistral_prompt_parts.append(f"User: {user_question}\n\nAnswer:") # User question is already in history now.
        mistral_full_prompt = "\n".join(mistral_prompt_parts)

        # Safety settings for Gemini
        safer_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        gemini_response = ""
        mistral_model_response = ""
        
        # Query Gemini
        try:
            logging.info("Querying Gemini with conversation history...")
            gemini_response_obj = gemini_model.generate_content(
                gemini_conversation_history, # This history is now context-managed
                generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1024),
                safety_settings=safer_settings
            )
            gemini_response = gemini_response_obj.text.strip() if gemini_response_obj.parts else "Gemini: I couldn't get a clear answer for this part."
            logging.info(f"Session {session_id} - Gemini Raw Response: {gemini_response}")
        except Exception as e:
            logging.error(f"Error querying Gemini model: {e}", exc_info=True)
            gemini_response = "Gemini: Could not connect or generate content. Please check API key or network."
            # Continue to Mistral, but Gemini's part will be an error message

        # Query Mistral model
        logging.info("Querying Mistral model with constructed prompt...")
        mistral_result = query_mistral_model(mistral_full_prompt)
        if mistral_result["success"]:
            mistral_model_response = mistral_result["content"]
            logging.info(f"Session {session_id} - Mistral Raw Response: {mistral_model_response}")
        else:
            mistral_model_response = f"Mistral: {mistral_result['error']}"
            logging.error(f"Session {session_id} - Mistral Error: {mistral_result['error']}")


        # --- ENHANCED SYNTHESIS PROMPT ---
        synthesis_prompt = (
            "You are an expert medical AI assistant. Your primary goal is to synthesize information from two AI models into a single, exceptionally clear, empathetic, and well-structured response. The output will be rendered in a chat interface, so the formatting must be perfect.\n\n"
            "--- CRITICAL FORMATTING RULES ---\n"
            "1.  **Structure and Flow:** Start with a direct answer, then elaborate with structured sections. Use headings to organize the information logically (e.g., Overview, Symptoms, Treatment Options, Risks).\n\n"
            "2.  **Headings:** To create a main heading, start the line with `##` followed by a space. Example: `## Common Symptoms`.\n\n"
            "3.  **Emphasis (Bolding):** This is very important for user experience. Proactively identify and bold key medical terms, important concepts, warnings, or crucial advice using double asterisks (`**`). Do not just bold random words; emphasize what a patient or user absolutely needs to notice. For example: `This could be a sign of a **serious condition** and requires **immediate medical attention**.` or `Common side effects include **nausea** and **dizziness**.`\n\n"
            "4.  **Lists:** For bullet points, start each line with a single asterisk followed by a space. Example: `* **Medication:** Prescribed by a doctor.`\n\n"
            "5.  **Disclaimer:** ALWAYS conclude the entire response with the following disclaimer, formatted exactly as shown:\n"
            "   `**DISCLAIMER:** This AI tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider.`\n\n"
            "--- YOUR TASK ---\n"
            "Synthesize the two answers below. Follow all formatting rules strictly. Be comprehensive, elaborate, and prioritize the most cautious medical view if there is a conflict. Your final response should be a perfect example of clarity and helpfulness.\n\n"
            "--- Model 1 (Gemini) Answer ---\n"
            f"{gemini_response}\n\n"
            "--- Model 2 (Mistral) Answer ---\n"
            f"{mistral_model_response}\n\n"
            "--- Synthesized and Perfectly Formatted Final Answer ---\n"
        )

        logging.info("Synthesizing responses with detailed UX-focused prompt...")

        def generate():
            full_response_content = ""
            try:
                stream = gemini_model.generate_content(
                    synthesis_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048),
                    safety_settings=safer_settings,
                    stream=True
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                        full_response_content += chunk.text
            except Exception as e:
                logging.error(f"Error during streaming synthesis: {e}", exc_info=True)
                yield "An error occurred during response generation."
            finally:
                # Store the complete synthesized response in history after streaming is done
                conversation_histories[session_id].append({"role": "model", "parts": [{"text": full_response_content}]})
                logging.info(f"Session {session_id} - Assistant (Full Response): {full_response_content}")


        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        logging.error(f"Error in /ask endpoint: {e}", exc_info=True)
        # If an error occurs, remove the last user message from history to avoid
        # polluting it with unresponded queries.
        if session_id in conversation_histories and len(conversation_histories[session_id]) > 0:
            if conversation_histories[session_id][-1]["role"] == "user":
                conversation_histories[session_id].pop()
        # Return a more specific error message from the main endpoint
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)