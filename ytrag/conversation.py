"""Conversation memory for context-aware queries.

Handles anaphora ("it", "this", "that") and conversational continuity.
Example:
    User: What is sliding window?
    AI: <explanation>
    User: When do we shrink it?  ← "it" refers to "sliding window"
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """A single message in conversation history."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class ConversationContext:
    """Manages conversation history and context-aware query rewriting."""
    messages: list[Message] = field(default_factory=list)
    max_history: int = 4  # Keep last N exchanges (2 user + 2 assistant = 1 full exchange)

    def add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content))

        # Trim history if needed
        if len(self.messages) > self.max_history * 2:  # Each exchange = 2 messages
            self.messages = self.messages[-self.max_history * 2:]

    def get_context(self, n: int = 2) -> str:
        """Get last N messages as formatted context."""
        recent = self.messages[-n*2:] if len(self.messages) >= n*2 else self.messages
        if not recent:
            return ""

        lines = []
        for msg in recent:
            prefix = "User: " if msg.role == "user" else "Assistant: "
            lines.append(f"{prefix}{msg.content}")
        return "\n".join(lines)

    def rewrite_query(self, query: str, use_llm: bool = False) -> str:
        """Rewrite query to be context-aware.

        Two approaches:
        1. Rule-based (fast, no API cost): Simple pattern matching for pronouns
        2. LLM-based (accurate, costs a call): Ask LLM to rewrite with context

        Args:
            query: Current user query
            use_llm: If True, use LLM for rewriting; else use rules

        Returns:
            Rewritten query with resolved references
        """
        if not self.messages:
            return query

        # Quick check: does query contain pronouns/references that need resolution?
        query_lower = query.lower()
        needs_rewrite = any(
            word in query_lower
            for word in ['it', 'this', 'that', 'these', 'those', 'ye', 'wo', 'uske', 'iske']
        )

        if not needs_rewrite:
            return query

        if use_llm:
            return self._llm_rewrite(query)
        else:
            return self._rule_rewrite(query)

    def _rule_rewrite(self, query: str) -> str:
        """Simple rule-based query rewriting.

        Strategy:
        - Find the last user message
        - Extract key technical terms from it
        - If current query has pronouns, prepend context
        """
        # Find last user question
        last_user_msg = None
        for msg in reversed(self.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return query

        # Simple heuristic: if current query is short and has pronouns,
        # prepend the topic from last query
        query_lower = query.lower()
        if len(query.split()) <= 10 and any(p in query_lower for p in ['it', 'this', 'that', 'ye', 'wo']):
            # Extract nouns/technical terms from last message (very crude)
            import re
            terms = re.findall(r'\b[a-z]{4,}\b', last_user_msg.lower())
            # Filter common words
            stops = {'what', 'when', 'where', 'which', 'kaise', 'kahan', 'kaise', 'explain', 'solve'}
            terms = [t for t in terms if t not in stops]

            if terms:
                # Use first significant term as context
                return f"{terms[0]}: {query}"

        return query

    def _llm_rewrite(self, query: str) -> str:
        """LLM-based query rewriting for better accuracy."""
        from ytrag.config import GROQ_API_KEY, GROQ_MODEL
        from groq import Groq

        if not GROQ_API_KEY:
            return self._rule_rewrite(query)

        context = self.get_context(n=2)
        if not context:
            return query

        prompt = f"""Given this conversation history:

{context}

The user now asks: "{query}"

If the query contains pronouns or references that need context (like "it", "this", "that"), rewrite it to be self-contained by replacing pronouns with what they refer to from the conversation.

If the query is already clear and doesn't need rewriting, return it unchanged.

Return ONLY the rewritten query, nothing else."""

        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            rewritten = (response.choices[0].message.content or "").strip()

            # Sanity check: if rewrite is way longer or empty, use original
            if not rewritten or len(rewritten) > len(query) * 3:
                return query

            return rewritten
        except Exception as e:
            print(f"⚠️  Query rewriting failed: {e}")
            return self._rule_rewrite(query)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


# Global conversation store (for stateful sessions)
_conversations: dict[str, ConversationContext] = {}


def get_conversation(session_id: str = "default") -> ConversationContext:
    """Get or create a conversation context for a session."""
    if session_id not in _conversations:
        _conversations[session_id] = ConversationContext()
    return _conversations[session_id]


def clear_conversation(session_id: str = "default") -> None:
    """Clear a conversation session."""
    if session_id in _conversations:
        _conversations[session_id].clear()
