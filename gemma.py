import ollama
import time

# --- Configuration ---
# Set the model name that you pulled earlier (e.g., llama2, gemma:2b)
MODEL_NAME = "gemma4:e4b"

def send_prompt_to_llm(prompt: str, model_name: str):
    """
    Connects to the local Ollama server and sends a prompt to the specified LLM model.

    Args:
        prompt (str): The text prompt to send to the LLM.
        model_name (str): The local model name (must be pulled via 'ollama pull').
    """
    print("\n" + "="*60)
    print(f"🚀 Attempting to connect to Ollama server using model: {model_name}")
    print("="*60)

    try:
        # 1. Initialize the Ollama client
        client = ollama.Client()

        print("✅ Connection successful. Sending prompt...")

        start_time = time.time()

        # 2. Call the API to generate content
        response = client.generate(
            model=model_name,
            prompt=prompt,
            # Setting temperature controls creativity (0.0 = deterministic, 1.0 = creative)
            options={"temperature": 0.7}
        )

        end_time = time.time()

        # The response object contains the generated text in the 'response' key
        generated_text = response['response'].strip()

        print("\n" + "="*60)
        print("🤖 AI Response Received:")
        print(generated_text)
        print("="*60)
        print(f"⏱️ Operation completed in {end_time - start_time:.2f} seconds.")


    except ollama.ResponseError as e:
        print("\n\n🛑 MODEL ERROR: Ollama returned an error.")
        print(f"Details: {e}")
        print("\n➡️ TROUBLESHOOTING: Check that the model name is correct and that the model is actually installed.")

    except ConnectionError:
        print("\n\n🛑 CONNECTION ERROR: Could not connect to the Ollama server.")
        print("🚨 ACTION REQUIRED: Please ensure the Ollama application is RUNNING and the server is active in the background.")

    except Exception as e:
        print("\n\n🛑 AN UNEXPECTED ERROR OCCURRED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Message: {e}")

# ------------------------------------------------
# This is the main execution block that gets run when the script executes
# ------------------------------------------------
if __name__ == "__main__":

    # --- USER INPUT AREA ---
    user_prompt = "Explain the concept of black holes to a group of ten-year-olds using only kitchen analogies."

    # --- FUNCTION CALL ---
    send_prompt_to_llm(user_prompt, MODEL_NAME)