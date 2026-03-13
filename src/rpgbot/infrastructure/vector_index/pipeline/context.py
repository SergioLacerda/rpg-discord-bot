class SearchContext:

    __slots__ = (
        "query",
        "q_vec",
        "query_tokens",
        "query_type",

        "vector_store",
        "token_store",
        "metadata_store",

        "cluster_manager",
        "ivf_router",
        "temporal_index",

        "prefilter_ids",

        "_vec_get",
        "_tok_get",
        "_meta_get",

        "_token_cache",
        "_meta_cache"
    )


    def __init__(
        self,
        query,
        q_vec,
        query_tokens,
        query_type,
        vector_store,
        token_store,
        metadata_store,
        cluster_manager,
        ivf_router,
        temporal_index,
        prefilter_ids=None
    ):

        self.query = query
        self.q_vec = q_vec
        self.query_tokens = query_tokens
        self.query_type = query_type

        self.vector_store = vector_store
        self.token_store = token_store
        self.metadata_store = metadata_store

        self.cluster_manager = cluster_manager
        self.ivf_router = ivf_router
        self.temporal_index = temporal_index

        self.prefilter_ids = prefilter_ids

        # -----------------------------------------
        # fast bindings (performance)
        # -----------------------------------------

        self._vec_get = vector_store.get
        self._tok_get = token_store.get
        self._meta_get = metadata_store.get

        # -----------------------------------------
        # local caches
        # -----------------------------------------

        self._token_cache = {}
        self._meta_cache = {}


    # -------------------------------------------------
    # vector access
    # -------------------------------------------------

    def get_vector(self, doc_id):
        return self._vec_get(doc_id)


    # -------------------------------------------------
    # token access
    # -------------------------------------------------

    def get_tokens(self, doc_id):

        cached = self._token_cache.get(doc_id)

        if cached:
            return cached

        tokens = self._tok_get(doc_id)

        self._token_cache[doc_id] = tokens

        return tokens


    # -------------------------------------------------
    # metadata access
    # -------------------------------------------------

    def get_metadata(self, doc_id):

        cached = self._meta_cache.get(doc_id)

        if cached:
            return cached

        meta = self._meta_get(doc_id)

        self._meta_cache[doc_id] = meta

        return meta