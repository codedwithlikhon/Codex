import requests
import json
import os
import argparse

# --- Configuration ---
# API endpoint for the Gemini model
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
# Default prompt if none is provided
DEFAULT_PROMPT = "Explain how AI works in a few words"

def get_api_key():
    """
    Retrieves the API key from an environment variable.
    Exits if the key is not found.
    """
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Error: API_KEY environment variable not set.")
        exit(1)
    return api_key

def generate_explanation(prompt: str, api_key: str) -> str:
    """
    Generates an explanation from the Gemini API using the provided prompt.

    Args:
        prompt: The text prompt to send to the model.
        api_key: The API key for authentication.

    Returns:
        The generated text from the model.
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        # Make the POST request to the API
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Safely extract the generated text
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("API response did not contain 'candidates'.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise ValueError("API response did not contain 'parts'.")

        return parts[0].get("text", "No text found in response.")

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        exit(1)
    except (ValueError, KeyError) as e:
        print(f"Error parsing API response: {e}")
        exit(1)

def main():
    """
    Main function to parse arguments and run the explanation generation.
    """
    parser = argparse.ArgumentParser(description="Generate a short AI explanation using Google's Gemini API.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"The prompt to send to the AI model. Defaults to: '{DEFAULT_PROMPT}'"
    )
    args = parser.parse_args()

    api_key = get_api_key()
    explanation = generate_explanation(args.prompt, api_key)
    print(explanation)

if __name__ == "__main__":
    main()