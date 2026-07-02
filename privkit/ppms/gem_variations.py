import math
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from typing import Dict

from privkit.ppms import PPM
from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.utils import geo_utils as gu, constants


class GEMVariations(PPM):
    """
    Geo-Graph Indistinguishability with adaptive epsilon class to apply the mechanism

    Variant of GEM that adapts the privacy parameter epsilon per point: rows flagged as `predicted`
    use the per-row epsilon, rows flagged as `close_points` are obfuscated around their predicted
    centroid, and all other rows use the base epsilon. The obfuscated location is sampled from the
    road graph with a graph-exponential mechanism over a local, ε-tied shortest-path neighbourhood,
    avoiding the construction of a full shortest-path-distance matrix.

    References
    ----------
    Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November).
    Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 ACM SIGSAC
    conference on Computer & communications security (pp. 901-914).
    """
    PPM_ID = "gem_variations"
    PPM_NAME = "GEM_VAR"
    PPM_INFO = "Adaptive geo-graph indistinguishability restricts the obfuscated location to the nodes of a road " \
               "graph and dynamically adapts the privacy parameter epsilon per point. Predicted points may use a " \
               "per-row epsilon and close points share an obfuscation around their predicted centroid. A " \
               "graph-exponential mechanism samples the obfuscated node with probability proportional to " \
               "exp(-(ε/2) * d_sp) over a local, ε-tied shortest-path neighbourhood."
    PPM_REF = "Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November). " \
              "Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 " \
              "ACM SIGSAC conference on Computer & communications security (pp. 901-914)."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    # Tunables (safe defaults)
    _ETA_TAIL = 1e-6          # max tail mass we are okay to ignore
    _R_MAX = 2000.0           # hard cap radius (meters) for very small epsilon
    _MIN_CAND = 25            # minimal number of candidates before we grow the cutoff
    _CUTOFF_GROW_FACTORS = (1.0, 1.5, 2.0, 3.0)  # try wider cutoffs if too few nodes
    _CACHE_MAX = 20000        # at most this many cached SSSPs

    def __init__(self, G: nx.MultiDiGraph, epsilon: float):
        """
        Initializes the adaptive GEM mechanism with the road network and base privacy parameter epsilon

        :param networkx.MultiDiGraph G: road network graph
        :param float epsilon: base privacy parameter
        """
        super().__init__()

        try:
            G = ox.utils_graph.get_largest_component(G, strongly=False)
        except Exception:
            pass

        self.G = G
        self.epsilon = float(epsilon)
        self.spanner_graph = self._maybe_make_spanner(G)  # optional light spanner; may just return G
        self.epsilon_values = []
        # Small manual cache: (src_node, round(cutoff, 1)) -> {node: dist}
        self._sssp_cache: Dict[tuple, Dict[int, float]] = {}

    def execute(self, location_data: LocationData):
        """
        Executes the adaptive GEM mechanism row-by-row. Uses the per-row epsilon for `predicted` rows,
        a shared obfuscation around the predicted centroid for `close_points` rows, and the base epsilon
        otherwise.

        :param privkit.LocationData location_data: location data where the mechanism should be executed
        :return: location data with obfuscated latitude and longitude and the quality loss metric
        """
        trajectories = location_data.get_trajectories()

        total, count_close = 0, 0

        for _, trajectory in trajectories:
            i = 0
            while i < len(trajectory):
                total += 1
                row = trajectory.iloc[i]
                row_type = row.get("type", "input")

                if row_type == "close_points":
                    # Obfuscate around the predicted centroid position
                    epsilon_i = float(row.get("epsilon", self.epsilon))
                    self.epsilon_values.append(epsilon_i)

                    lat_c = float(row["pred_lat"])
                    lon_c = float(row["pred_lon"])
                    d = self.compute_shortest_path_distance_dict(self.spanner_graph, lat_c, lon_c, eps=epsilon_i)
                    obf_node = self.graph_exponential_mechanism(d, epsilon_i)
                    obf_lat, obf_lon = self._node_latlon(obf_node)

                    # Quality loss vs. the true position for this row
                    qloss = gu.great_circle_distance(float(row[constants.LATITUDE]), float(row[constants.LONGITUDE]),
                                                     obf_lat, obf_lon)

                    # Write the same obfuscation for all consecutive 'close_points'
                    while i < len(trajectory) and trajectory.iloc[i]["type"] == "close_points":
                        idx = trajectory.iloc[i].name
                        location_data.data.loc[idx, constants.OBF_LATITUDE] = obf_lat
                        location_data.data.loc[idx, constants.OBF_LONGITUDE] = obf_lon
                        location_data.data.loc[idx, QualityLoss.METRIC_ID] = qloss
                        count_close += 1
                        i += 1
                else:
                    # Normal or 'predicted' point
                    lat = float(row[constants.LATITUDE])
                    lon = float(row[constants.LONGITUDE])

                    # Per-row epsilon for predicted points, else base epsilon
                    epsilon_i = float(row.get("epsilon", self.epsilon)) if row_type == "predicted" else self.epsilon
                    epsilon_i = max(epsilon_i, 1e-9)  # guard
                    self.epsilon_values.append(epsilon_i)

                    d = self.compute_shortest_path_distance_dict(self.spanner_graph, lat, lon, eps=epsilon_i)
                    obf_node = self.graph_exponential_mechanism(d, epsilon_i)
                    obf_lat, obf_lon = self._node_latlon(obf_node)

                    qloss = gu.great_circle_distance(lat, lon, obf_lat, obf_lon)

                    idx = row.name
                    location_data.data.loc[idx, constants.OBF_LATITUDE] = obf_lat
                    location_data.data.loc[idx, constants.OBF_LONGITUDE] = obf_lon
                    location_data.data.loc[idx, QualityLoss.METRIC_ID] = qloss

                    i += 1

        return location_data

    def _maybe_make_spanner(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Optional edge sparsification of the graph. If anything fails, just returns G unchanged.

        :param networkx.MultiDiGraph G: road network graph
        :return: a (possibly sparsified) MultiDiGraph
        """
        try:
            from networkx.algorithms.spanner import greedy_spanner
            H = greedy_spanner(G.to_undirected(as_view=True), stretch=3, weight="length")
            # Put back into MultiDiGraph form with the same node attributes
            H = nx.MultiDiGraph(H)
            for n, data in G.nodes(data=True):
                if n in H:
                    H.nodes[n].update(data)
            return H
        except Exception:
            return G

    @staticmethod
    def _cutoff_from_eps(eps: float, eta: float = _ETA_TAIL, R_max: float = _R_MAX) -> float:
        # Tail mass outside radius R is ~exp(-eps * R / 2); choose R so that it is <= eta, capped at R_max
        R = (2.0 / max(eps, 1e-9)) * math.log(1.0 / max(eta, 1e-15))
        return min(R, R_max)

    def _nearest_node(self, G: nx.MultiDiGraph, lat: float, lon: float) -> int:
        # osmnx expects X=lon, Y=lat for nearest_nodes
        return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))

    def _node_latlon(self, n: int):
        return float(self.G.nodes[n]['y']), float(self.G.nodes[n]['x'])

    def compute_shortest_path_distance_dict(self, graph, lat: float, lon: float,
                                            weight: str = "length", eps: float = None):
        """
        Computes a dict {node -> shortest-path distance (m)} for a local neighborhood around (lat, lon).
        The neighborhood radius is tied to epsilon via _cutoff_from_eps; the cutoff grows slightly if there
        are too few candidates, and results are cached per (source, cutoff).

        :param graph: graph to search (falls back to self.G if None)
        :param float lat: latitude of the query point
        :param float lon: longitude of the query point
        :param str weight: edge attribute used as distance
        :param float eps: epsilon used to derive the cutoff (defaults to self.epsilon)
        :return: dict mapping node -> shortest-path distance (m)
        """
        Gq = graph if graph is not None else self.G
        src = self._nearest_node(Gq, lat, lon)
        cutoff_base = self._cutoff_from_eps(self.epsilon if eps is None else float(eps))

        for mult in self._CUTOFF_GROW_FACTORS:
            cutoff = round(cutoff_base * mult, 1)
            key = (src, cutoff)
            if key in self._sssp_cache:
                d = self._sssp_cache[key]
            else:
                d = nx.single_source_dijkstra_path_length(Gq, src, weight=weight, cutoff=cutoff)
                # Cache maintenance: evict the oldest entry when full
                if len(self._sssp_cache) >= self._CACHE_MAX:
                    self._sssp_cache.pop(next(iter(self._sssp_cache)))
                self._sssp_cache[key] = d

            # Ensure we have at least a modest number of candidates
            if len(d) >= self._MIN_CAND:
                return d

        # Last attempt result (may be small; at least includes src: 0)
        return d

    def graph_exponential_mechanism(self, distances, epsilon: float):
        """
        Samples a node with probability proportional to exp(-(ε/2) * d_sp), using log-space for stability.

        :param distances: mapping node -> shortest-path distance
        :param float epsilon: privacy parameter
        :return: ID of the sampled obfuscated node
        """
        if not distances:
            raise ValueError("Empty distance dict in graph_exponential_mechanism.")

        nodes = np.fromiter(distances.keys(), dtype=object)
        ds = np.fromiter(distances.values(), dtype=float)

        # Log-weights: -(ε/2) * d
        logw = -0.5 * float(epsilon) * ds
        m = np.max(logw)
        w = np.exp(logw - m)
        s = w.sum()

        if not np.isfinite(s) or s <= 0:
            # Fallback: nearest node
            return int(nodes[np.argmin(ds)])

        p = w / s
        return int(np.random.choice(nodes, p=p))
