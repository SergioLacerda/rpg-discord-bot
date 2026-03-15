try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

from rpgbot.usecases.embeddings.embedding_provider import EmbeddingProvider
from rpgbot.core.providers import embedding_registry


def _detect_device(device):

    if device is not None:

        valid = {"cpu", "cuda", "mps"}

        if device not in valid:
            raise ValueError(
                f"Invalid device '{device}'. Expected one of {valid}"
            )

        return device

    if torch and torch.cuda.is_available():
        return "cuda"

    if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _auto_batch_size(device):

    if device == "cuda":
        return 128

    if device == "mps":
        return 64

    return 16


if SentenceTransformer:

    class SentenceEmbeddingProvider(EmbeddingProvider):

        def __init__(self, model: str, device: str | None = None, batch_size: int | None = None):

            self.model_name = model

            device = _detect_device(device)
            self.device = device

            self.batch_size = batch_size or _auto_batch_size(device)

            self.model = SentenceTransformer(
                model,
                device=device
            )

            # detectar modelos E5
            self.is_e5 = "e5" in model.lower()

        # ---------------------------------------------------------
        # helpers
        # ---------------------------------------------------------

        def _prepare_doc(self, text):

            if self.is_e5:
                return "passage: " + text

            return text

        # ---------------------------------------------------------
        # single embedding
        # ---------------------------------------------------------

        async def embed(self, text):

            text = self._prepare_doc(text)

            with torch.no_grad():

                vec = self.model.encode(
                    text,
                    normalize_embeddings=True
                )

            return vec.tolist()

        # ---------------------------------------------------------
        # batch embedding
        # ---------------------------------------------------------

        async def embed_batch(self, texts):

            if self.is_e5:
                texts = ["passage: " + t for t in texts]

            with torch.no_grad():

                vectors = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )

            return [v.tolist() for v in vectors]


    @embedding_registry.register("sentence", aliases=["local", "hf"])
    def create_sentence_embedding(**kwargs):

        model = kwargs.get("model", "intfloat/multilingual-e5-base")
        device = kwargs.get("device")
        batch_size = kwargs.get("batch_size")

        return SentenceEmbeddingProvider(
            model=model,
            device=device,
            batch_size=batch_size
        )
