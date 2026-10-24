from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class CrawledPage(BaseModel):
    url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    page_type: Optional[str] = None
    text: str
    content_hash: Optional[str] = None
    status_code: Optional[int] = None


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 10
    allowed_domains: list[str] = Field(default_factory=list)
    idempotency_key: str


class CrawlerClient(ABC):
    @abstractmethod
    def crawl_pages(self, request: CrawlRequest) -> list[CrawledPage]:
        raise NotImplementedError("crawl_pages method not implemented")
