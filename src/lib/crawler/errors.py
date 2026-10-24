class CrawlerError(Exception):
    pass


class CrawlerConfigError(CrawlerError):
    pass


class CrawlerQuotaError(CrawlerError):
    pass


class CrawlerFailedError(CrawlerError):
    pass


class CrawlerTimeoutError(CrawlerError):
    pass
