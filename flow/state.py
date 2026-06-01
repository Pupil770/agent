from enum import Enum
from threading import Lock

_state_lock = Lock()
_conversations: dict[str, "ConversationState"] = {}


class ConversationStage(str, Enum):
    GREETING = "greeting"
    INQUIRY = "inquiry"
    ANSWERING = "answering"
    CONFIRMING = "confirming"
    TRANSFERRING = "transferring"
    ENDED = "ended"


class ConversationState:
    def __init__(self):
        self.stage = ConversationStage.GREETING
        self.turn_count = 0
        self.intent: str | None = None
        self.needs_human = False
        self.unresolved_count = 0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "turn_count": self.turn_count,
            "intent": self.intent,
            "needs_human": self.needs_human,
            "unresolved_count": self.unresolved_count,
        }


def get_state(thread_id: str) -> ConversationState:
    with _state_lock:
        if thread_id not in _conversations:
            _conversations[thread_id] = ConversationState()
        return _conversations[thread_id]


def reset_state(thread_id: str) -> ConversationState:
    with _state_lock:
        _conversations[thread_id] = ConversationState()
        return _conversations[thread_id]
