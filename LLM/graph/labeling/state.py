from pydantic import BaseModel

class LabelState(BaseModel):
    images_uri: list[str]
    target_classes: list[str]
    synonyms: dict[str, list[str]] | None = None
    neg_prompts: list[str] | None = None
    current_model: str | None = None   # model@vA.B
    thresholds: dict = {...}
    dataset_version: str | None = None
    stats: dict | None = None