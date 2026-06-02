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
        self.last_stage: ConversationStage | None = None
        self.turn_count = 0
        self.intent: str | None = None
        self.needs_human = False
        self.unresolved_count = 0
        self.summary: str = ""

    def transition_to(self, new_stage: ConversationStage):
        self.last_stage = self.stage
        self.stage = new_stage

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "last_stage": self.last_stage.value if self.last_stage else None,
            "turn_count": self.turn_count,
            "intent": self.intent,
            "needs_human": self.needs_human,
            "unresolved_count": self.unresolved_count,
            "summary_length": len(self.summary),
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
