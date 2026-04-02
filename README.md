---
license: mit
title: Email Triage OpenEnv
sdk: docker
emoji: 🚀
colorFrom: blue
colorTo: green
short_description: Email triage OpenEnv environment for AI agent training
tags: 
    - openenv
---
# 📧 Email Triage OpenEnv

An [OpenEnv](https://openenv.dev)-compliant environment that simulates real-world **email inbox management** — one of the most universally performed tasks in the modern workplace.

AI agents learn to classify spam, prioritize emails by urgency, and draft professional replies, building skills directly transferable to real productivity workflows.

---

## Why Email Triage?

Email management consumes an estimated **2.5 hours per day** for the average worker. It requires:
- **Judgment** (is this urgent or can it wait?)
- **Classification** (spam vs. legitimate, urgent vs. routine)
- **Communication** (crafting appropriate replies)

This makes it an ideal benchmark for evaluating agents on real cognitive tasks with clear, measurable success criteria.

---

## Environment Overview

| Field | Value |
|---|---|
| Domain | Email management |
| Tasks | 3 (easy → medium → hard) |
| Episodes | 5–10 steps each |
| Score range | 0.0 – 1.0 per step |
| Reward type | Dense (partial credit per step) |
| Framework | FastAPI + Pydantic |

---

## Tasks

### Task 1 — Spam Detection *(Easy)*
**Objective:** Classify each of 10 emails as `spam` or `not_spam`.

- Spam signals: prize/lottery offers, fake security alerts, unsolicited ads, phishing
- Legitimate: work emails, shipping confirmations, expected newsletters
- **Grader:** Binary — 1.0 for correct, 0.0 for incorrect
- **Baseline score:** ~80%

### Task 2 — Priority Triage *(Medium)*
**Objective:** Assign a priority level (1=highest, 5=lowest) to each of 10 emails.

- Priority 1: Critical outages, legal deadlines, C-suite requests
- Priority 2–3: Normal business communications
- Priority 4–5: Low-urgency announcements, social messages
- **Grader:** Partial credit — 1.0 for exact, 0.7 for ±1, 0.3 for ±2, 0.0 for ±3+
- **Baseline score:** ~65%

### Task 3 — Reply Drafting *(Hard)*
**Objective:** Write a one-sentence professional reply for each of 5 emails.

- Must address the sender's specific request
- Scored on keyword coverage + reply quality + appropriate length
- Penalizes generic non-answers
- **Grader:** 0.0–1.0 based on keyword match (80%) + length quality (20%)
- **Baseline score:** ~55%

---

## Observation Space

```json
{
  "current_email": {
    "id": "email_001",
    "subject": "URGENT: Production server down",
    "sender": "alerts@monitoring.com",
    "body": "CRITICAL: The production API server has been unresponsive...",
    "timestamp": "2024-10-15 09:00:00"
  },
  "emails_remaining": 9,
  "emails_processed": 1,
  "last_action_feedback": "Correct! This was urgent.",
  "task_id": "task_2_priority_triage",
  "task_description": "Assign a priority level (1=highest, 5=lowest) to each email."
}
```

## Action Space

```json
// Task 1 - Spam Detection
{ "action_type": "classify", "email_id": "email_001", "label": "spam" }

// Task 2 - Priority Triage
{ "action_type": "prioritize", "email_id": "email_001", "priority": 1 }

// Task 3 - Reply Drafting
{ "action_type": "reply", "email_id": "email_001", "reply_summary": "I will escalate this to the on-call engineer immediately." }

// Skip (forfeits reward)
{ "action_type": "skip", "email_id": "email_001" }
```

## Reward Function

Rewards are **dense** — every step produces a signal:

- **Task 1:** Binary 0.0/1.0 per email (deterministic)
- **Task 2:** Graduated 1.0 / 0.7 / 0.3 / 0.0 based on distance from correct priority
- **Task 3:** Continuous 0.0–1.0 based on content quality

Episode score = mean of all per-step scores.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode, returns first observation |
| `/step` | POST | Submit action, returns observation + reward |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all tasks and action schemas |
| `/grader` | GET | Final episode score (call after done=True) |
| `/baseline` | POST | Run built-in rule-based agent, returns scores |
| `/health` | GET | Health check |

---

## Setup & Usage

### Local Development

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# 4. Test it
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "task_1_spam_detection"}'
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Run Baseline Agent

```bash
export OPENAI_API_KEY=your_key_here

# Run locally (no server needed)
python baseline.py

# Run against a live server
python baseline.py --server http://localhost:7860
```

### Example Agent Loop

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_1_spam_detection"}).json()

done = False
while not done:
    email = obs["current_email"]
    
    # Your agent logic here
    label = "spam" if "won" in email["subject"].lower() else "not_spam"
    
    result = requests.post(f"{BASE}/step", json={
        "task_id": "task_1_spam_detection",
        "action_type": "classify",
        "email_id": email["id"],
        "label": label,
    }).json()
    
    obs = result["observation"]
    done = result["done"]
    print(f"Reward: {result['reward']['score']}")

# Get final score
score = requests.get(f"{BASE}/grader", params={"task_id": "task_1_spam_detection"}).json()
print(f"Final score: {score['score']}")
```

---

## Baseline Scores

| Task | Agent | Score |
|---|---|---|
| Task 1 — Spam Detection | Rule-based | ~80% |
| Task 2 — Priority Triage | Rule-based | ~65% |
| Task 3 — Reply Drafting | Rule-based | ~55% |
| Task 1 — Spam Detection | GPT-4o-mini | ~90% |
| Task 2 — Priority Triage | GPT-4o-mini | ~75% |
| Task 3 — Reply Drafting | GPT-4o-mini | ~70% |

---

## Project Structure

```
email-triage-env/
├── app.py                  # FastAPI server (all endpoints)
├── baseline.py             # Baseline inference script (OpenAI API)
├── openenv.yaml            # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
├── README.md
└── env/
    ├── __init__.py
    └── email_triage_env.py # Core environment (step/reset/state + graders)
```

---

## License

MIT