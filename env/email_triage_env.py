"""
Email Triage OpenEnv Environment
Simulates a real-world email inbox management task where an agent
must classify, prioritize, and respond to emails.
"""

import random
from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ──────────────────────────────────────────────
# Typed Models (OpenEnv spec)
# ──────────────────────────────────────────────

class ActionType(str, Enum):
    CLASSIFY = "classify"       # Label email as spam/not_spam/urgent/normal/low
    PRIORITIZE = "prioritize"   # Assign priority 1-5 to email
    REPLY = "reply"             # Write a short reply summary
    SKIP = "skip"               # Skip this email (costs reward)


class EmailAction(BaseModel):
    action_type: ActionType = Field(..., description="Type of action to take")
    email_id: str = Field(..., description="ID of the email to act on")
    label: Optional[str] = Field(None, description="Label for classify action: spam|not_spam|urgent|normal|low")
    priority: Optional[int] = Field(None, description="Priority for prioritize action: 1 (highest) to 5 (lowest)")
    reply_summary: Optional[str] = Field(None, description="One-sentence reply summary for reply action")


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class EmailObservation(BaseModel):
    current_email: Optional[Email] = Field(None, description="The current email to process")
    emails_remaining: int = Field(..., description="Number of emails left in inbox")
    emails_processed: int = Field(..., description="Number of emails processed so far")
    last_action_feedback: Optional[str] = Field(None, description="Feedback on the last action taken")
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Description of what the agent should do")


class EmailReward(BaseModel):
    score: float = Field(..., description="Reward for last action (0.0 to 1.0 per email)")
    cumulative_score: float = Field(..., description="Total score so far")
    reason: str = Field(..., description="Explanation of the reward")


# ──────────────────────────────────────────────
# Email Dataset
# ──────────────────────────────────────────────

SPAM_EMAILS = [
    {"subject": "YOU WON $1,000,000!!!", "sender": "prize@winlottery.xyz",
     "body": "Congratulations! You have been selected as our lucky winner. Click here to claim your prize NOW!", "label": "spam"},
    {"subject": "Cheap meds - no prescription needed", "sender": "pharmacy@nodrugs.ru",
     "body": "Buy Viagra, Xanax, and more without prescription. Discreet shipping. Order today!", "label": "spam"},
    {"subject": "Your account has been compromised", "sender": "security@paypa1.com",
     "body": "We noticed unusual activity. Click this link immediately to verify your identity and restore access.", "label": "spam"},
    {"subject": "Make $5000 a week from home!", "sender": "jobs@easymoney.biz",
     "body": "No experience needed. Work just 2 hours a day. Hundreds of people are already earning huge money!", "label": "spam"},
    {"subject": "FREE iPhone 15 Pro - Limited offer", "sender": "gifts@freephone.click",
     "body": "You are our 1 millionth visitor! Claim your free iPhone 15 Pro. Only 3 left. Hurry!", "label": "spam"},
]

NOT_SPAM_EMAILS = [
    {"subject": "Q3 Budget Review - Action Required", "sender": "cfo@company.com",
     "body": "Hi team, please review the attached Q3 budget report and submit your department's projections by Friday EOD.", "label": "not_spam"},
    {"subject": "Team lunch this Friday", "sender": "hr@company.com",
     "body": "We're doing a team lunch this Friday at 12:30pm at The Italian Place. Please RSVP by Thursday.", "label": "not_spam"},
    {"subject": "Your order has shipped", "sender": "noreply@amazon.com",
     "body": "Your order #112-3456789 has shipped and will arrive by Thursday. Track your package here.", "label": "not_spam"},
    {"subject": "Monthly newsletter - October 2024", "sender": "newsletter@techdigest.com",
     "body": "This month in tech: AI breakthroughs, new product launches, and our top 10 tools for developers.", "label": "not_spam"},
    {"subject": "Meeting notes from yesterday", "sender": "colleague@company.com",
     "body": "Hi, here are the notes from yesterday's product review meeting. Let me know if I missed anything.", "label": "not_spam"},
]

URGENT_EMAILS = [
    {"subject": "URGENT: Production server down", "sender": "alerts@monitoring.com",
     "body": "CRITICAL: The production API server has been unresponsive for 15 minutes. Error rate is 100%. Immediate action required.", "priority": 1, "label": "urgent"},
    {"subject": "Customer data breach detected", "sender": "security@company.com",
     "body": "Our intrusion detection system flagged a potential data breach at 3:42AM. Security team needs immediate response.", "priority": 1, "label": "urgent"},
    {"subject": "Board meeting in 30 minutes - missing slides", "sender": "ceo@company.com",
     "body": "The board meeting starts in 30 minutes and I don't have the Q4 slides. Can you send them IMMEDIATELY?", "priority": 1, "label": "urgent"},
    {"subject": "Legal deadline TODAY 5PM", "sender": "legal@lawfirm.com",
     "body": "The contract must be signed and returned by 5PM today or we lose the deal. Please review and sign ASAP.", "priority": 1, "label": "urgent"},
]

NORMAL_EMAILS = [
    {"subject": "Project update - Week 42", "sender": "pm@company.com",
     "body": "Weekly project status: Development on track. Testing phase starts Monday. No blockers at this time.", "priority": 3, "label": "normal"},
    {"subject": "Can we schedule a call next week?", "sender": "partner@clientcorp.com",
     "body": "Hi, I'd like to catch up on the partnership proposal. Are you available Tuesday or Wednesday afternoon?", "priority": 3, "label": "normal"},
    {"subject": "Feedback on your presentation", "sender": "manager@company.com",
     "body": "Good job on the presentation yesterday! A few minor suggestions: consider adding more data to slide 4 and simplifying the conclusion.", "priority": 2, "label": "normal"},
]

LOW_PRIORITY_EMAILS = [
    {"subject": "Office plants need watering", "sender": "facilities@company.com",
     "body": "Reminder: it's your turn to water the office plants this week. Plants are near the kitchen.", "priority": 5, "label": "low"},
    {"subject": "Parking lot resurfacing next month", "sender": "facilities@company.com",
     "body": "The parking lot will be resurfaced on November 15th. Please use the overflow lot on that day.", "priority": 5, "label": "low"},
    {"subject": "Happy Birthday to Mike!", "sender": "hr@company.com",
     "body": "Please join us in wishing Mike from accounting a very happy birthday! Stop by his desk with a treat.", "priority": 5, "label": "low"},
]

REPLY_EMAILS = [
    {"subject": "Question about your service pricing", "sender": "prospect@newclient.com",
     "body": "Hi, I'm interested in your enterprise plan. Could you send me pricing information and a demo?",
     "expected_reply_keywords": ["pricing", "demo", "enterprise", "contact", "send", "information", "schedule"],
     "label": "normal"},
    {"subject": "Invoice #4521 - Payment due", "sender": "billing@supplier.com",
     "body": "Please find attached invoice #4521 for $2,450 due on October 31st. Please confirm receipt.",
     "expected_reply_keywords": ["received", "invoice", "confirm", "payment", "process"],
     "label": "normal"},
    {"subject": "Request for project extension", "sender": "contractor@freelance.com",
     "body": "Due to unexpected complexity, I need 3 additional days to complete the design work. Please advise.",
     "expected_reply_keywords": ["extension", "days", "deadline", "approve", "understand", "confirm"],
     "label": "normal"},
    {"subject": "Can you review my code PR?", "sender": "dev@company.com",
     "body": "Hey, I submitted PR #234 for the auth module. It's blocking my next task. Can you review it today?",
     "expected_reply_keywords": ["review", "PR", "look", "today", "check"],
     "label": "normal"},
    {"subject": "Partnership opportunity inquiry", "sender": "biz@partnerco.com",
     "body": "We would like to explore a potential integration partnership between our companies. Who should I speak to?",
     "expected_reply_keywords": ["partnership", "contact", "team", "speak", "discuss", "forward"],
     "label": "normal"},
]


# ──────────────────────────────────────────────
# Task Definitions
# ──────────────────────────────────────────────

TASKS = {
    "task_1_spam_detection": {
        "name": "Spam Detection (Easy)",
        "description": "Classify each email as 'spam' or 'not_spam'. You will see 10 emails one at a time.",
        "difficulty": "easy",
        "action_type": ActionType.CLASSIFY,
        "valid_labels": ["spam", "not_spam"],
        "num_emails": 10,
    },
    "task_2_priority_triage": {
        "name": "Priority Triage (Medium)",
        "description": "Assign a priority level (1=highest, 5=lowest) to each email. Consider urgency, sender, and content.",
        "difficulty": "medium",
        "action_type": ActionType.PRIORITIZE,
        "num_emails": 10,
    },
    "task_3_reply_drafting": {
        "name": "Reply Drafting (Hard)",
        "description": "Write a one-sentence professional reply summary for each email that addresses the sender's request.",
        "difficulty": "hard",
        "action_type": ActionType.REPLY,
        "num_emails": 5,
    },
}


# ──────────────────────────────────────────────
# Environment Class
# ──────────────────────────────────────────────

class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.
    Agents practice real-world email management across 3 difficulty levels.
    """

    def __init__(self, task_id: str = "task_1_spam_detection", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASKS.keys())}")
        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self.seed = seed
        self._rng = random.Random(seed)
        self._emails: list[dict] = []
        self._current_index: int = 0
        self._scores: list[float] = []
        self._done: bool = False
        self._last_feedback: str = ""

    def _build_email_list(self) -> list[dict]:
        """Build a shuffled email list appropriate for the task."""
        rng = random.Random(self.seed)

        if self.task_id == "task_1_spam_detection":
            pool = SPAM_EMAILS * 2 + NOT_SPAM_EMAILS * 2
            emails = rng.sample(pool, min(self.task_config["num_emails"], len(pool)))

        elif self.task_id == "task_2_priority_triage":
            pool = URGENT_EMAILS + NORMAL_EMAILS + LOW_PRIORITY_EMAILS
            pool = pool * 2
            emails = rng.sample(pool, min(self.task_config["num_emails"], len(pool)))

        else:  # task_3_reply_drafting
            emails = rng.sample(REPLY_EMAILS, min(self.task_config["num_emails"], len(REPLY_EMAILS)))

        # Assign IDs
        result = []
        for i, e in enumerate(emails):
            entry = dict(e)
            entry["id"] = f"email_{i+1:03d}"
            entry["timestamp"] = f"2024-10-{15+i:02d} 09:{i*3:02d}:00"
            result.append(entry)
        return result

    def reset(self) -> EmailObservation:
        """Reset the environment and return the first observation."""
        self._emails = self._build_email_list()
        self._current_index = 0
        self._scores = []
        self._done = False
        self._last_feedback = "Starting new episode. Good luck!"
        return self._get_observation()

    def step(self, action: EmailAction) -> tuple[EmailObservation, EmailReward, bool, dict]:
        """
        Take one step in the environment.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        current_email = self._emails[self._current_index]

        # Validate email_id matches current
        if action.email_id != current_email["id"]:
            feedback = f"Wrong email_id. Expected '{current_email['id']}', got '{action.email_id}'."
            reward_score = 0.0
        elif action.action_type == ActionType.SKIP:
            feedback = "Email skipped. No points awarded."
            reward_score = 0.0
        else:
            reward_score, feedback = self._grade_action(action, current_email)

        self._scores.append(reward_score)
        self._last_feedback = feedback
        self._current_index += 1

        done = self._current_index >= len(self._emails)
        self._done = done

        cumulative = sum(self._scores) / len(self._scores) if self._scores else 0.0

        reward = EmailReward(
            score=reward_score,
            cumulative_score=round(cumulative, 4),
            reason=feedback,
        )

        obs = self._get_observation() if not done else self._get_final_observation()
        info = {"episode_done": done, "emails_graded": len(self._scores)}
        return obs, reward, done, info

    def state(self) -> dict:
        """Return current environment state (for OpenEnv spec compliance)."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_config["name"],
            "current_index": self._current_index,
            "total_emails": len(self._emails),
            "scores": self._scores,
            "cumulative_score": round(sum(self._scores) / len(self._scores), 4) if self._scores else 0.0,
            "done": self._done,
        }

    def _get_observation(self) -> EmailObservation:
        if self._current_index >= len(self._emails):
            return self._get_final_observation()
        email_data = self._emails[self._current_index]
        email = Email(
            id=email_data["id"],
            subject=email_data["subject"],
            sender=email_data["sender"],
            body=email_data["body"],
            timestamp=email_data["timestamp"],
        )
        return EmailObservation(
            current_email=email,
            emails_remaining=len(self._emails) - self._current_index,
            emails_processed=self._current_index,
            last_action_feedback=self._last_feedback,
            task_id=self.task_id,
            task_description=self.task_config["description"],
        )

    def _get_final_observation(self) -> EmailObservation:
        final_score = round(sum(self._scores) / len(self._scores), 4) if self._scores else 0.0
        return EmailObservation(
            current_email=None,
            emails_remaining=0,
            emails_processed=len(self._emails),
            last_action_feedback=f"Episode complete! Final score: {final_score:.2%}",
            task_id=self.task_id,
            task_description=self.task_config["description"],
        )

    def _grade_action(self, action: EmailAction, email: dict) -> tuple[float, str]:
        """Grade an action and return (score, feedback)."""

        # ── Task 1: Spam detection ──
        if self.task_id == "task_1_spam_detection":
            if action.action_type != ActionType.CLASSIFY:
                return 0.0, f"Wrong action type. Expected 'classify', got '{action.action_type}'."
            if action.label not in ["spam", "not_spam"]:
                return 0.0, f"Invalid label '{action.label}'. Must be 'spam' or 'not_spam'."
            correct = email["label"]
            if action.label == correct:
                return 1.0, f"Correct! This was {correct}."
            else:
                return 0.0, f"Incorrect. This was {correct}, you said {action.label}."

        # ── Task 2: Priority triage ──
        elif self.task_id == "task_2_priority_triage":
            if action.action_type != ActionType.PRIORITIZE:
                return 0.0, f"Wrong action type. Expected 'prioritize', got '{action.action_type}'."
            if action.priority is None or not (1 <= action.priority <= 5):
                return 0.0, "Invalid priority. Must be integer 1-5."

            correct_priority = email.get("priority", 3)
            diff = abs(action.priority - correct_priority)

            if diff == 0:
                score, msg = 1.0, f"Perfect! Correct priority {correct_priority}."
            elif diff == 1:
                score, msg = 0.7, f"Close. Correct was {correct_priority}, you said {action.priority}."
            elif diff == 2:
                score, msg = 0.3, f"Somewhat off. Correct was {correct_priority}, you said {action.priority}."
            else:
                score, msg = 0.0, f"Incorrect. Correct was {correct_priority}, you said {action.priority}."
            return score, msg

        # ── Task 3: Reply drafting ──
        elif self.task_id == "task_3_reply_drafting":
            if action.action_type != ActionType.REPLY:
                return 0.0, f"Wrong action type. Expected 'reply', got '{action.action_type}'."
            if not action.reply_summary or len(action.reply_summary.strip()) < 10:
                return 0.0, "Reply too short. Provide at least one meaningful sentence."

            reply = action.reply_summary.lower()
            keywords = email.get("expected_reply_keywords", [])
            matched = sum(1 for kw in keywords if kw.lower() in reply)
            keyword_score = matched / max(len(keywords), 1)

            # Length quality signal (10-200 chars is ideal)
            length = len(action.reply_summary.strip())
            if 20 <= length <= 200:
                length_bonus = 0.2
            else:
                length_bonus = 0.0

            # Penalize generic/empty responses
            generic_phrases = ["i will get back to you", "noted", "ok", "sure", "yes"]
            is_generic = any(p in reply for p in generic_phrases) and length < 50
            if is_generic:
                return 0.1, "Reply too generic. Please address the specific request."

            score = min(1.0, keyword_score * 0.8 + length_bonus)
            score = round(score, 4)
            msg = f"Reply scored {score:.0%}. Matched {matched}/{len(keywords)} key concepts."
            return score, msg

        return 0.0, "Unknown task."

    def grade_episode(self) -> float:
        """Return final episode score (0.0 - 1.0). Call after done=True."""
        if not self._scores:
            return 0.0
        return round(sum(self._scores) / len(self._scores), 4)
