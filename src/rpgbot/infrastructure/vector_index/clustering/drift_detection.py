class ClusterDriftDetector:

    def __init__(self, rebuild_threshold=0.20):
        self.rebuild_threshold = rebuild_threshold

    def should_rebuild(
        self,
        current_size: int,
        previous_size: int,
        has_centroids: bool
    ) -> bool:

        # no clusters yet
        if not has_centroids:
            return True

        # first build
        if previous_size == 0:
            return True

        growth = abs(current_size - previous_size)

        return (growth / previous_size) > self.rebuild_threshold