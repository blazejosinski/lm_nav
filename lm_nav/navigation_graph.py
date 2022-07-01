import pickle
from tkinter import W
from PIL import Image  # type: ignore
import numpy as np
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import base64
from io import BytesIO
from lm_nav.utils import latlong_to_utm

import logging


logger = logging.getLogger(__name__)


class NavigationGraph(object):
    EPS = 3e-5

    def __init__(self, path=None):
        if path is None:
            self._pos = []
            self._images = []
            self._graph = nx.Graph()
        else:
            self.load_from_file(path)

    @property
    def vert_count(self):
        return self._graph.number_of_nodes()

    def load_from_file(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pos = data["pos"]
        self._images = data["images"]
        self._graph = nx.readwrite.json_graph.node_link_graph(data["json_graph"])

    def add_edge(self, node, adj_node, weight=None):
        if weight is None:
            weight = np.linalg.norm(self._pos[node] - self._pos[adj_node])
        self._graph.add_edge(node, adj_node, weight=weight)

    def add_vertix(self, obs):
        assert (
           (b"gps/latlong" in obs) or (b"gps/utm" in obs)
        ), "Observation should contain latitude and longitude"
        inx = self.vert_count
        if b"gps/utm" in obs:
            pos = obs[b"gps/utm"]
        else:
            pos = latlong_to_utm(obs[b"gps/latlong"])
        self._graph.add_node(inx)
        self._pos.append(pos)
        self._images.append([obs[b"images/rgb_left"]])

    def add_image(self, obs):
        assert (
            (b"gps/latlong" in obs) or (b"gps/utm" in obs)
        ), "Observation should contain latitude and longitude"
        if b"gps/utm" in obs:
            pos = obs[b"gps/utm"]
        else:
            pos = latlong_to_utm(obs[b"gps/latlong"])
        for inx, pos1 in enumerate(self._pos):
            if np.linalg.norm(pos - pos1) < self.EPS:
                self._images[inx].append(obs[b"images/rgb_left"])
                break
        else:
            self.add_vertix(obs)

    def json_repr_for_visualization(self, image_size=300):
        positions = np.vstack(self._pos)
        positions[:,1] = -positions[:,1]
        min_pos = np.min(positions, 0)
        positions = positions - min_pos
        max_pos = np.max(positions)
        positions = positions / max_pos * image_size * 0.9
        positions += 0.05 * image_size
        verticies = {}
        for inx in range(self.vert_count):
            images_str = []
            for image in self._images[inx]:
                # buffered = BytesIO()
                # image = Image.fromarray(image)
                # image.save(buffered, format="JPEG")
                # img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
                img_str = str(base64.b64encode(image))[2:-1]
                images_str.append(img_str)
            verticies[str(inx)] = {
                "position": [float(z) for z in positions[inx]],
                "images": images_str,
            }

        edges = list(self._graph.edges)
        return verticies, edges
