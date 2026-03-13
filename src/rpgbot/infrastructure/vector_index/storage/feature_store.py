import mmap
import struct
from pathlib import Path


class FeatureStore:

    RECORD_STRUCT = struct.Struct("fii")
    # context_score, entity_offset, entity_len

    def __init__(self, path):

        self.path = Path(path)

        self.doc_to_row = {}

        self.features_file = None
        self.entities = []

    # ---------------------------------------------------------
    # load memory mapped features
    # ---------------------------------------------------------

    def load(self):

        f = open(self.path / "features.mmap", "rb")

        self.features_file = mmap.mmap(
            f.fileno(),
            0,
            access=mmap.ACCESS_READ
        )

    # ---------------------------------------------------------
    # feature access
    # ---------------------------------------------------------

    def get(self, doc_id):

        row = self.doc_to_row.get(doc_id)

        if row is None:
            return None

        offset = row * self.RECORD_STRUCT.size

        context, ent_offset, ent_len = self.RECORD_STRUCT.unpack_from(
            self.features_file,
            offset
        )

        entities = set(
            self.entities[ent_offset: ent_offset + ent_len]
        )

        return {
            "context": context,
            "entities": entities
        }