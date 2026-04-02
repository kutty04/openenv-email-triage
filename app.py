"""
FastAPI server for Email Triage OpenEnv environment.
Exposes required endpoints: /reset, /step, /state, /tasks, /grader, /baseline
"""

import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.email_triage_env import (
    EmailTriageEnv,
    EmailAction,
    ActionType,
    TASKS,
)

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv environment simulating real-world email triage tasks for AI agent training.",
    version="1.0.0",
)

# ──────────────────────────────────────────────
# Session store (in-memory, one session per task)
# ──────────────────────────────────────────────
_sessions: dict[str, EmailTriageEnv] = {}


def get_or_create_env(task_id: str) -> EmailTriageEnv:
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
    if task_id not in _sessions:
        _sessions[task_id] = EmailTriageEnv(task_id=task_id)
    return _sessions[task_id]


# ──────────────────────────────────────────────
# Request/Response models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_spam_detection"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    task_id: str = "task_1_spam_detection"
    action_type: str
    email_id: str
    label: Optional[str] = None
    priority: Optional[int] = None
    reply_summary: Optional[str] = None


# ──────────────────────────────────────────────
# Core OpenEnv endpoints
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "description": "OpenEnv environment for email triage tasks",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset the environment and return initial observation."""
    env = EmailTriageEnv(task_id=request.task_id, seed=request.seed or 42)
    _sessions[request.task_id] = env
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    """Take one step: submit an action, receive observation + reward."""
    env = get_or_create_env(request.task_id)

    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action_type '{request.action_type}'. Valid: {[a.value for a in ActionType]}")

    action = EmailAction(
        action_type=action_type,
        email_id=request.email_id,
        label=request.label,
        priority=request.priority,
        reply_summary=request.reply_summary,
    )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "task_1_spam_detection"):
    """Return current environment state."""
    env = get_or_create_env(task_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    """Return list of available tasks and their action schemas."""
    tasks_info = []
    for task_id, config in TASKS.items():
        tasks_info.append({
            "task_id": task_id,
            "name": config["name"],
            "description": config["description"],
            "difficulty": config["difficulty"],
            "num_emails": config["num_emails"],
            "action_schema": {
                "action_type": config["action_type"].value,
                "email_id": "string — ID of the email to act on",
                "label": "string — required for classify: 'spam' or 'not_spam'" if task_id == "task_1_spam_detection" else None,
                "priority": "integer 1-5 — required for prioritize" if task_id == "task_2_priority_triage" else None,
                "reply_summary": "string — required for reply: one-sentence reply" if task_id == "task_3_reply_drafting" else None,
            }
        })
    return {"tasks": tasks_info}


@app.get("/grader")
def grader(task_id: str = "task_1_spam_detection"):
    """Return the final grader score for a completed episode."""
    env = get_or_create_env(task_id)
    state_data = env.state()
    if not state_data["done"]:
        return {
            "task_id": task_id,
            "done": False,
            "score": None,
            "message": "Episode not complete yet. Keep stepping until done=True.",
        }
    score = env.grade_episode()
    return {
        "task_id": task_id,
        "done": True,
        "score": score,
        "emails_graded": state_data["emails_graded"] if "emails_graded" in state_data else len(state_data["scores"]),
        "message": f"Episode complete. Final score: {score:.2%}",
    }


@app.post("/baseline")
def run_baseline():
    """
    Trigger the baseline inference script.
    Runs a simple rule-based agent across all 3 tasks and returns scores.
    """
    results = {}
    for task_id in TASKS:
        env = EmailTriageEnv(task_id=task_id, seed=42)
        env.reset()
        results[task_id] = _run_rule_based_agent(env, task_id)

    return {
        "baseline_agent": "rule_based",
        "results": results,
        "mean_score": round(sum(r["score"] for r in results.values()) / len(results), 4),
    }


def _run_rule_based_agent(env: EmailTriageEnv, task_id: str) -> dict:
    """Simple deterministic rule-based baseline agent."""
    obs = env.reset()
    done = False
    step_count = 0

    SPAM_KEYWORDS = ["won", "prize", "free", "click here", "no prescription",
                     "make money", "limited offer", "verify your identity"]

    while not done and step_count < 20:
        email = obs.current_email
        if email is None:
            break

        subject_lower = email.subject.lower()
        body_lower = email.body.lower()
        combined = subject_lower + " " + body_lower

        if task_id == "task_1_spam_detection":
            is_spam = any(kw in combined for kw in SPAM_KEYWORDS)
            action = EmailAction(
                action_type=ActionType.CLASSIFY,
                email_id=email.id,
                label="spam" if is_spam else "not_spam",
            )

        elif task_id == "task_2_priority_triage":
            if any(w in combined for w in ["urgent", "critical", "immediately", "asap", "today"]):
                priority = 1
            elif any(w in combined for w in ["review", "feedback", "meeting", "request"]):
                priority = 2
            elif any(w in combined for w in ["update", "weekly", "schedule", "next week"]):
                priority = 3
            elif any(w in combined for w in ["newsletter", "reminder", "fyi"]):
                priority = 4
            else:
                priority = 5
            action = EmailAction(
                action_type=ActionType.PRIORITIZE,
                email_id=email.id,
                priority=priority,
            )

        else:  # task_3_reply_drafting
            reply = f"Thank you for your email regarding '{email.subject}'. I will review your request and follow up with the relevant information shortly."
            action = EmailAction(
                action_type=ActionType.REPLY,
                email_id=email.id,
                reply_summary=reply,
            )

        obs, reward, done, info = env.step(action)
        step_count += 1

    final_score = env.grade_episode()
    return {
        "task_id": task_id,
        "score": final_score,
        "steps": step_count,
    }


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
