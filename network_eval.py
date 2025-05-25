import base64
import requests
import json
import sys

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"  # Change if you're using a different name

def encode_image_to_base64(image_path):
    """Encodes an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

def query_gemma3_vision(image_path, prompt=\
    "Evaluate the attached essay based on the following expectations: \
    1. Clarity and Accuracy: \
    Are the network components correctly represented? \
    Is the connection between devices clearly shown? \
    Is there logical flow in the layout? \
    2. Use of Standard Icons and Labels: \
    Are standard network symbols (e.g., routers, switches, servers) used? \
    Are devices and connections appropriately labeled? \
    3. Organization and Neatness: \
    Is the diagram well-organized and not cluttered? \
    Are icons properly aligned and consistently sized? \
    Use the rubric below to give a score: \
    4. Adherence to Instructions: \
    Did the design meet the specific requirements or instructions given (e.g., include at least one router, one switch, a firewall, etc.)? \
    5. Creativity and Professionalism: \
    Does the topology diagram demonstrate creativity in layout while maintaining professionalism? \
    Use the rubric below to give a score:\
    90-100% (A): Exceptional work; diagram is clear, accurate, complete, and professionally presented, fully meeting or exceeding all requirements with consistent use of standard icons and a creative, polished look.\
    80-89% (B): Solid work; diagram meets most requirements, with minor errors in clarity, neatness, or use of icons; mostly professional and organized with good creativity.\
    70-79% (C): Satisfactory work; diagram demonstrates an adequate understanding of network topology design, but has noticeable weaknesses in clarity, neatness, use of icons, or adherence to instructions.\
    60–69% (D): Below average work; diagram demonstrates a limited understanding, with significant weaknesses in clarity, icon usage, neatness, or adherence to requirements.\
    Below 60% (F): Unsatisfactory work; diagram fails to meet most of the requirements, with major issues in clarity, accuracy, icon usage, and neatness; lacks effort or professionalism."):
    
    image_base64 = encode_image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response text returned.")

    except requests.RequestException as e:
        return f"Request failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gemma3_vision_insight.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    insights = query_gemma3_vision(image_path)
    print("=== Insights Extracted ===")
    print(insights)


 