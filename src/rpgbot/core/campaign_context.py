from contextvars import ContextVar


_current_campaign: ContextVar[str] = ContextVar(
    "current_campaign",
    default="default"
)


class CampaignContext:
    """
    Contexto de campanha isolado por coroutine.
    Seguro para execução concorrente.
    """

    def set_campaign(self, campaign_id: str):

        _current_campaign.set(str(campaign_id))

    def get_campaign(self) -> str:

        return _current_campaign.get()

    def reset(self):

        _current_campaign.set("default")

    # ---------------------------------------------------------
    # context manager útil para escopo automático
    # ---------------------------------------------------------

    def scope(self, campaign_id):

        token = _current_campaign.set(str(campaign_id))

        class _Scope:

            def __enter__(self_inner):
                return campaign_id

            def __exit__(self_inner, exc_type, exc, tb):
                _current_campaign.reset(token)

        return _Scope()