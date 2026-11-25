from source_sync_api import create_admin_router, create_webhook_router


def register(context) -> None:
    context.include_router(create_admin_router())
    context.include_router(
        create_webhook_router(),
        skip_pat=True,
        skip_brain=True,
    )
