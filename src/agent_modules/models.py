"""Every data model that crosses a module boundary. Layer 0: imports nothing internal.

These shapes are contracts. They are serialised into the journey trace and into model prompts, so
their field names are part of the wire format and are not free to change casually.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CandidateProfile(BaseModel):
    """Candidate facts. Missing or null facts mean unknown, never no."""

    model_config = ConfigDict(extra="allow")

    identity: dict[str, Any] = Field(default_factory=dict)
    location: dict[str, Any] = Field(default_factory=dict)
    experience: dict[str, Any] = Field(default_factory=dict)
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    work_authorization: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    documents: dict[str, Any] = Field(default_factory=dict)


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str


class PageElement(BaseModel):
    """One interactive control as the reader found it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    label: str
    label_source: str | None = None
    group: str | None = None
    input_type: str | None
    required: bool
    value: str | None
    enabled: bool
    checked: bool | None
    options: list[Option]
    help_text: str | None
    visible: bool
    file_attached: bool = False
    invalid: bool = False
    readonly: bool = False


class PageSnapshot(BaseModel):
    """The whole readable state of one page at one moment."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    url: str
    title: str
    visible_text: list[str]
    elements: list[PageElement]
    validation_errors: list[str]
    blockers: list[str]


ActionType = Literal["fill", "select", "set_checked", "click", "upload", "scroll"]
PlanStatus = Literal["continue", "complete", "needs_input", "blocked", "failed"]


class Action(BaseModel):
    """One safe browser primitive. Nullable fields are intentional for strict JSON output."""

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    target: str | None
    value: str | None
    value_ref: str | None
    option_value: str | None
    checked: bool | None
    asset_id: str | None


class CompletionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None
    url: str | None
    reference: str | None


class ApplicationPlan(BaseModel):
    """A model's constrained intent for one page. Nothing here has run yet.

    No snapshot id. A plan is produced and executed against the same observation inside one step, so
    there is no window in which it could go stale - and asking the model to echo an id back was a way
    for a run to die on a copied string rather than on anything real.
    """

    model_config = ConfigDict(extra="forbid")

    status: PlanStatus
    actions: list[Action]
    reason: str
    completion_evidence: CompletionEvidence | None


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    ok: bool
    message: str


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "needs_input", "blocked", "failed", "step_limit", "timeout"]
    message: str
    url: str
    steps: int
    evidence: CompletionEvidence | None
    action_results: list[ActionResult]
    journey_log: str | None = None
    journey_jsonl: str | None = None
    journey_timing: str | None = None
