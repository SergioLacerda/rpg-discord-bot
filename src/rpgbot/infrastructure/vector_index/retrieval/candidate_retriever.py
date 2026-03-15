import bisect

from rpgbot.utils.vector.vector_math import cosine_similarity
from rpgbot.utils.vector.vector_utils import (
    lsh_hash,
    project
)


class CandidateRetriever:

    def __init__(
        self,
        docs,
        projections,
        vector_store,
        ivf_router=None,
        lsh_buckets=None,
        hnsw_index=None,
        hierarchical_retriever=None,
        super_centroids=None,
        cluster_to_super=None,
        window_size=50,
        hnsw_k=200
    ):

        self.docs = docs
        self.vector_store = vector_store
        self.projections = projections

        self.ivf_router = ivf_router
        self.lsh_buckets = lsh_buckets or {}
        self.hnsw_index = hnsw_index
        self.hierarchical_retriever = hierarchical_retriever

        # new: two-level routing
        self.super_centroids = super_centroids
        self.cluster_to_super = cluster_to_super or {}

        self.window_size = window_size
        self.hnsw_k = hnsw_k

    # ---------------------------------------------------------
    # coarse routing
    # ---------------------------------------------------------

    def _route_super_cluster(self, q_vec):

        if not self.super_centroids:
            return None

        best = None
        best_score = -1

        for sid, centroid in enumerate(self.super_centroids):

            s = cosine_similarity(q_vec, centroid)

            if s > best_score:
                best_score = s
                best = sid

        return best

    # ---------------------------------------------------------
    # candidate retrieval
    # ---------------------------------------------------------

    def run(self, ctx, candidates):

        q_vec = ctx.q_vec

        # ---------------------------------------------
        # two-level ANN routing
        # ---------------------------------------------

        if self.ivf_router:

            subset = getattr(ctx, "prefilter_ids", None)

            if self.super_centroids:

                super_id = self._route_super_cluster(q_vec)

                if super_id is not None:

                    # limitar clusters ao super cluster
                    subset_clusters = {
                        cid for cid, sid in self.cluster_to_super.items()
                        if sid == super_id
                    }

            cluster_ids = self.ivf_router.route(
                q_vec,
                subset=subset,
                cluster_subset=subset_clusters
            )

            if cluster_ids:

                # -----------------------------------------
                # cluster-level HNSW search
                # -----------------------------------------

                results = []

                if hasattr(ctx.index.cluster_manager, "cluster_indexes"):

                    for cid in cluster_ids:

                        index = ctx.index.cluster_manager.cluster_indexes.get(cid)

                        if index:

                            docs = index.search(q_vec, k=self.hnsw_k)

                            results.extend(docs)

                if results:
                    return results

                else:
                    ids = self.ivf_router.search(q_vec, subset=subset)

            else:
                ids = self.ivf_router.search(q_vec, subset=subset)

            if ids:
                return ids

        # ---------------------------------------------
        # hierarchical retrieval
        # ---------------------------------------------

        if self.hierarchical_retriever:

            ids = self.hierarchical_retriever(q_vec)

            if ids:
                return ids

        # ---------------------------------------------
        # LSH fallback
        # ---------------------------------------------

        bucket = self.lsh_buckets.get(lsh_hash(q_vec))

        if bucket:
            return list(bucket)

        # ---------------------------------------------
        # HNSW fallback
        # ---------------------------------------------

        if self.hnsw_index:

            ids = self.hnsw_index.search(q_vec, k=self.hnsw_k)

            if ids:
                return ids

        # ---------------------------------------------
        # projection window fallback
        # ---------------------------------------------

        if not self.projections:
            return []

        q_proj = project(q_vec)

        pos = bisect.bisect_left(self.projections, q_proj)

        window = max(self.window_size, int(len(self.docs) ** 0.5))

        start = max(0, pos - window)
        end = min(len(self.docs), pos + window)

        return self.docs[start:end]