from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate_create"]

class Citation(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None   # filename
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    module_id: Optional[str] = None
    module_position: Optional[int] = None

class MaterialGenRequest(BaseModel):
    course_id: int
    topic: str
    target_duration_min: int = Field(default=50, ge=10, le=180)
    audience_level: Literal["intro", "intermediate", "advanced"] = "intro"
    constraints: Optional[str] = None

class LearningObjective(BaseModel):
    objective: str
    bloom_level: BloomLevel

class OutlineItem(BaseModel):
    minute_range: str
    section_title: str
    bullets: List[str] = Field(default_factory=list)

class WorkedExample(BaseModel):
    problem: str
    solution_steps: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

class PracticeProblem(BaseModel):
    problem: str
    solution: str

class ConfusionItem(BaseModel):
    confusion: str
    fix: str

class TeachingPacket(BaseModel):
    topic: str
    learning_objectives: List[LearningObjective] = Field(default_factory=list)
    lecture_outline: List[OutlineItem] = Field(default_factory=list)
    worked_examples: List[WorkedExample] = Field(default_factory=list)
    in_class_questions: Dict[str, List[str]] = Field(default_factory=dict)
    practice: List[PracticeProblem] = Field(default_factory=list)
    common_confusions: List[ConfusionItem] = Field(default_factory=list)
    sources: List[Citation] = Field(default_factory=list)

class TeachingPacketResponse(BaseModel):
    teaching_packet: TeachingPacket
    markdown: str
    trace_id: str

    @staticmethod
    def stub(req: MaterialGenRequest) -> "TeachingPacketResponse":
        pkt = TeachingPacket(
            topic=req.topic,
            learning_objectives=[
                LearningObjective(objective=f"Explain {req.topic} in your own words", bloom_level="understand"),
                LearningObjective(objective=f"Apply {req.topic} to a worked example", bloom_level="apply"),
            ],
            lecture_outline=[
                OutlineItem(minute_range="0–10", section_title="Motivation & intuition", bullets=["Why this matters", "Key idea"]),
                OutlineItem(minute_range="10–30", section_title="Core definitions", bullets=["Definitions", "Notation"]),
                OutlineItem(minute_range="30–45", section_title="Worked example", bullets=["Walk through step-by-step"]),
                OutlineItem(minute_range="45–50", section_title="Check understanding", bullets=["Quick questions"]),
            ],
            in_class_questions={
                "remember": [f"What is the definition of {req.topic}?"],
                "apply": [f"Use {req.topic} to solve a simple problem."],
            },
            common_confusions=[
                ConfusionItem(confusion="Mixing up similar terms", fix="Add a comparison table and a quick quiz."),
            ],
        )
        md = f"""# Teaching Packet: {req.topic}

## Learning objectives
- (Understand) Explain {req.topic} in your own words
- (Apply) Apply {req.topic} to a worked example

## Lecture outline ({req.target_duration_min} min)
- 0–10: Motivation & intuition
- 10–30: Core definitions
- 30–45: Worked example
- 45–50: Check understanding
"""
        return TeachingPacketResponse(teaching_packet=pkt, markdown=md, trace_id="stub-trace-0001")
