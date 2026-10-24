from src.adapters.interfaces.crawler import CrawledPage, CrawlerClient, CrawlRequest


class CrawlerAdapter:
    def __init__(self):
        self.crawler = None

    def add_client(self, client: CrawlerClient) -> None:
        self.crawler = client

    def crawl_pages(self, request: CrawlRequest) -> list[CrawledPage]:
        return self.crawler.crawl_pages(request)
