import time
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock
import config
from utils.helpers import cosine_similarity

class ClusterCache:
    """Per‑cluster cache with LRU eviction and optional TTL."""
    def __init__(self, cluster_id: int, max_size: int = None, ttl: int = None):
        self.cluster_id = cluster_id
        self.max_size = max_size or config.PER_CLUSTER_CACHE_SIZE
        self.ttl = ttl or config.CACHE_TTL
        self.entries: OrderedDict = OrderedDict()   # key = query text (or we can use index)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = Lock()

    def _is_expired(self, entry: Dict) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - entry['timestamp']) > self.ttl

    def lookup(self, query_emb: np.ndarray, threshold: float) -> Optional[Tuple[str, float, Any]]:
        """Return (matched_query, similarity, result) if hit, else None."""
        with self.lock:
            best_sim = -1.0
            best_entry = None
            best_key = None
            # Check all entries in this cluster
            for key, entry in list(self.entries.items()):
                if self._is_expired(entry):
                    # remove expired
                    del self.entries[key]
                    continue
                sim = cosine_similarity(query_emb, entry['query_embedding'])
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_key = key
            if best_sim >= threshold:
                # Hit
                self.hit_count += 1
                # Update LRU: move to end (most recent)
                self.entries.move_to_end(best_key)
                self.entries[best_key]['lru_time'] = time.time()
                return best_entry['query_text'], best_sim, best_entry['result']
            else:
                self.miss_count += 1
                return None

    def insert(self, query: str, query_emb: np.ndarray, result: Any):
        """Insert new entry; evict if needed (LRU)."""
        with self.lock:
            # If already present, remove old
            if query in self.entries:
                del self.entries[query]
            # Evict if over max_size
            if len(self.entries) >= self.max_size:
                # pop first (least recently used) – OrderedDict keeps insertion order,
                # but we update order on lookup. So we use popitem(last=False) for LRU.
                self.entries.popitem(last=False)
            self.entries[query] = {
                'query_text': query,
                'query_embedding': query_emb,
                'result': result,
                'timestamp': time.time(),
                'lru_time': time.time()
            }

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.hit_count = 0
            self.miss_count = 0

    def get_stats(self):
        with self.lock:
            return {
                'size': len(self.entries),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
            }


class ClusterCacheManager:
    """Manages per‑cluster caches."""
    def __init__(self, n_clusters: int, global_max_size: int = None):
        self.n_clusters = n_clusters
        self.caches: Dict[int, ClusterCache] = {}
        self.global_max_size = global_max_size or config.GLOBAL_CACHE_SIZE
        # Configure per cluster (simple uniform for now)
        for cid in range(n_clusters):
            self.caches[cid] = ClusterCache(cid, max_size=self.global_max_size // n_clusters)

    def get_cache(self, cluster_id: int) -> ClusterCache:
        return self.caches[cluster_id]

    def lookup(self, query_emb: np.ndarray, cluster_id: int, threshold: float = config.CACHE_THRESHOLD):
        cache = self.get_cache(cluster_id)
        return cache.lookup(query_emb, threshold)

    def insert(self, query: str, query_emb: np.ndarray, cluster_id: int, result: Any):
        cache = self.get_cache(cluster_id)
        cache.insert(query, query_emb, result)

    def get_stats(self):
        total_entries = 0
        total_hits = 0
        total_misses = 0
        for cache in self.caches.values():
            stats = cache.get_stats()
            total_entries += stats['size']
            total_hits += stats['hit_count']
            total_misses += stats['miss_count']
        total_queries = total_hits + total_misses
        hit_rate = total_hits / total_queries if total_queries > 0 else 0
        return {
            'total_entries': total_entries,
            'hit_count': total_hits,
            'miss_count': total_misses,
            'hit_rate': hit_rate
        }

    def flush(self):
        for cache in self.caches.values():
            cache.clear()