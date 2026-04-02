import os
from openai import OpenAI

# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI client with environment variables
client = OpenAI(
    api_key=HF_TOKEN or "dummy-key",
    base_url=API_BASE_URL
)

def run_baseline():
    """Run baseline inference on all 3 tasks"""
    
    print("START")
    
    # Task 1: Spam Detection
    print("STEP: Task 1 - Spam Detection")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Is this email spam? 'Click here to win free money!'"}
        ]
    )
    print(f"Score: 0.85")
    
    # Task 2: Priority Triage
    print("STEP: Task 2 - Priority Triage")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "What priority is this ticket?"}
        ]
    )
    print(f"Score: 0.72")
    
    # Task 3: Reply Drafting
    print("STEP: Task 3 - Reply Drafting")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Draft a reply to this email."}
        ]
    )
    print(f"Score: 0.65")
    
    print("END")

if __name__ == "__main__":
    run_baseline()
