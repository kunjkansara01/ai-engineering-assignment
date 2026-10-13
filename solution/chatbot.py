"""Q6: access-filtered, evidence-only QA across independent project stores."""
import copy
import re
import uuid
from dataclasses import dataclass, field

STOP = set("what is are the a an of for in on to how many much does do it that about and please tell me".split())
ALIASES = {"vacation": "leave", "holidays": "leave", "pto": "leave", "quota": "limit", "limits": "limit", "days": "day"}


def tokens(text):
    return {ALIASES.get(w, w) for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in STOP}


@dataclass(frozen=True)
class Principal:
    """Created by the trusted application after authentication, not by a chat prompt."""
    user_id: str


@dataclass
class Session:
    owner: str
    project: str
    history: list = field(default_factory=list)
    topic: str | None = None


def demo_documents():
    # Two independent sources per project; all records are synthetic.
    return [
        {"id": "atlas-hr", "project": "atlas", "topic": "leave", "key": "annual_leave", "value": "20 days",
         "text": "Atlas annual leave allowance is 20 days.", "source": "atlas/hr.md#annual-leave", "version": "1"},
        {"id": "atlas-api", "project": "atlas", "topic": "api", "key": "rate_limit", "value": "100 requests per minute",
         "text": "Atlas API rate limit is 100 requests per minute.", "source": "atlas/api.md#rate-limit", "version": "1"},
        {"id": "boreal-hr", "project": "boreal", "topic": "leave", "key": "annual_leave", "value": "30 days",
         "text": "Boreal annual leave allowance is 30 days.", "source": "boreal/hr.md#annual-leave", "version": "1"},
        {"id": "boreal-api", "project": "boreal", "topic": "api", "key": "rate_limit", "value": "50 requests per minute",
         "text": "Boreal API rate limit is 50 requests per minute.", "source": "boreal/api.md#rate-limit", "version": "1"},
        {"id": "atlas-retention-a", "project": "atlas", "topic": "retention", "key": "retention", "value": "30 days",
         "text": "Atlas log retention is 30 days.", "source": "atlas/security.md#retention", "version": "1"},
        {"id": "atlas-retention-b", "project": "atlas", "topic": "retention", "key": "retention", "value": "90 days",
         "text": "Atlas log retention is 90 days.", "source": "atlas/operations.md#retention", "version": "1"},
    ]


class KnowledgeBot:
    def __init__(self, documents=None, permissions=None):
        self.documents = copy.deepcopy(demo_documents() if documents is None else documents)
        self.permissions = copy.deepcopy(permissions if permissions is not None else
                                         {"alice": {"atlas"}, "bob": {"boreal"}, "manager": {"atlas", "boreal"}})
        self.sessions = {}

    def authorize(self, principal, project):
        if project not in self.permissions.get(principal.user_id, set()):
            raise PermissionError("Project access denied")

    def start_session(self, principal, project):
        self.authorize(principal, project)
        sid = uuid.uuid4().hex
        self.sessions[sid] = Session(principal.user_id, project)
        return sid

    def session(self, principal, sid):
        session = self.sessions.get(sid)
        if session is None or session.owner != principal.user_id:
            raise PermissionError("Session access denied")
        self.authorize(principal, session.project)
        return session

    def switch_project(self, principal, sid, project):
        session = self.session(principal, sid)
        self.authorize(principal, project)
        session.project = project
        session.topic = None
        session.history.clear()

    def history(self, principal, sid):
        return copy.deepcopy(self.session(principal, sid).history)

    def ask(self, principal, sid, question):
        s = self.session(principal, sid)  # recheck ACL on EVERY turn
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Question must contain 1–2000 characters")
        q = tokens(question)
        project_names = {d["project"].casefold() for d in self.documents}
        if (q & project_names) - {s.project.casefold()}:
            response = {"status": "PROJECT_SWITCH_REQUIRED", "answer": "Select the project explicitly; permissions will be checked.", "sources": []}
        elif re.search(r"ignore\b.*\b(?:instruction|rule)|system prompt|reveal.*secret", question, re.I):
            response = {"status": "REFUSED", "answer": "Only evidence-backed project questions are supported.", "sources": []}
        else:
            topic_terms = {"leave": {"leave", "allowance"}, "api": {"api", "rate", "limit", "requests"},
                           "retention": {"retention", "logs", "log"}}
            topics = [topic for topic, terms in topic_terms.items() if q & terms]
            referential = bool(re.search(r"\b(it|that|those|this)\b", question, re.I))
            topic = topics[0] if len(topics) == 1 else s.topic if not topics and referential else None
            if len(topics) > 1:
                response = {"status": "CLARIFY", "answer": "Ask about one topic at a time.", "sources": []}
            elif topic is None:
                response = {"status": "MISSING_EVIDENCE", "answer": "No supporting evidence found in the selected project.", "sources": []}
            else:
                # Physical metadata filtering before lexical retrieval; no global top-k.
                allowed = [d for d in self.documents if d["project"] == s.project and d["topic"] == topic]
                # A narrow fact prototype must abstain on unsupported aspects such as carryover.
                content_q = q - project_names - {"it", "that", "those", "this", "again", "explain", "per", "policy"}
                vocabulary = set().union(*(tokens(d["text"]) for d in allowed)) if allowed else set()
                unknown = content_q - vocabulary - set().union(*topic_terms.values()) - {"day", "annual", "allowance"}
                if unknown or not allowed:
                    response = {"status": "MISSING_EVIDENCE", "answer": "The sources do not establish that detail.", "sources": []}
                else:
                    allowed.sort(key=lambda d: len(q & tokens(d["text"])), reverse=True)
                    groups = {}
                    for d in allowed:
                        groups.setdefault(d["key"], set()).add(d["value"])
                    conflict = any(len(values) > 1 for values in groups.values())
                    # Evidence is never treated as executable instructions or an LLM prompt.
                    response = {"status": "CONTRADICTORY" if conflict else "ANSWERED",
                                "answer": ("Conflicting sources: " if conflict else "") + " ".join(d["text"] for d in allowed),
                                "sources": [{"id": d["id"], "source": d["source"], "version": d["version"],
                                             "project": d["project"], "quote": d["text"]} for d in allowed]}
                    assert all(d["project"] == s.project for d in response["sources"])
                    s.topic = topic
        s.history.append({"question": question, "response": copy.deepcopy(response)})
        s.history[:] = s.history[-20:]
        return response
