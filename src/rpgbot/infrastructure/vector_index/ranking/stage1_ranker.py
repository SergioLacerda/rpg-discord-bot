from rpgbot.utils.vector.vector_utils import cosine_similarity, keyword_score


class Stage1Ranker:

    def __init__(
        self,
        small_candidate_threshold=200,
        low_similarity=0.03,
        high_similarity=0.05,
        vector_weight=0.7,
        keyword_weight=0.3,
        min_limit=40,
        early_stop_ratio=0.6,
    ):

        self.small_candidate_threshold = small_candidate_threshold
        self.low_similarity = low_similarity
        self.high_similarity = high_similarity

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        self.min_limit = min_limit
        self.early_stop_ratio = early_stop_ratio

    # ---------------------------------------------------------
    # ranking
    # ---------------------------------------------------------

    def rank(
        self,
        q_vec,
        query_tokens,
        candidate_ids,
        vector_store,
        token_store,
        k
    ):

        stage1 = []

        # -----------------------------------------
        # bindings locais (performance)
        # -----------------------------------------

        vec_get = vector_store.get
        tok_get = token_store.get

        cos_sim = cosine_similarity
        kw_score_fn = keyword_score

        # -----------------------------------------
        # thresholds
        # -----------------------------------------

        threshold = (
            self.low_similarity
            if len(candidate_ids) < self.small_candidate_threshold
            else self.high_similarity
        )

        limit = max(self.min_limit, k * 10)

        best_score = None

        # -----------------------------------------
        # lazy vector similarity
        # -----------------------------------------

        lazy_sim = LazyVectorSimilarity(vector_store)

        candidate_ids = lazy_sim.top_k(q_vec, candidate_ids, k * 20)

        # -----------------------------------------
        # ranking
        # -----------------------------------------

        for doc_id in candidate_ids:

            vec = vec_get(doc_id)

            vec_score = cos_sim(q_vec, vec)

            if vec_score < threshold:
                continue

            # keyword scoring apenas quando necessário
            if self.keyword_weight > 0:

                tokens, _ = tok_get(doc_id)

                kw_score = kw_score_fn(query_tokens, tokens)

            else:
                kw_score = 0.0

            quick_score = (
                self.vector_weight * vec_score
                + self.keyword_weight * kw_score
            )

            stage1.append((quick_score, vec_score, doc_id))

            # -----------------------------------------
            # registrar melhor score
            # -----------------------------------------

            if best_score is None:
                best_score = quick_score

            # -----------------------------------------
            # early stop
            # -----------------------------------------

            elif (
                len(stage1) >= limit
                and quick_score < best_score * self.early_stop_ratio
            ):
                break

        # -----------------------------------------
        # ordenar
        # -----------------------------------------

        stage1.sort(key=lambda x: x[0], reverse=True)

        return stage1[:limit]