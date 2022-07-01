import cv2  # type: ignore
import heapq
import io
from mimetypes import init
import numpy as np
import torch
from typing import List, Tuple
import clip  # type: ignore
import PIL  # type: ignore


from lm_nav.navigation_graph import NavigationGraph
from lm_nav.utils import rectify_and_crop


def dijskra_transform(
    initial: np.ndarray, graph: NavigationGraph, alpha: float
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    next = np.copy(initial)
    prev = [(i, -1) for i in range(graph.vert_count)]
    priority_queue = []
    for i in range(len(initial)):
        priority_queue.append((-initial[i], i))
    heapq.heapify(priority_queue)
    while priority_queue:
        (value, node) = heapq.heappop(priority_queue)
        value = -value
        if next[node] != value:
            continue
        for neighbor in graph._graph.neighbors(node):
            weight = alpha * graph._graph.get_edge_data(node, neighbor)["weight"]
            if next[neighbor] < value - weight:
                next[neighbor] = value - weight
                heapq.heappush(priority_queue, (-next[neighbor], neighbor))
                prev[neighbor] = (node, 0)
    return next, prev


def nodes_landmarks_similarity(
    graph: NavigationGraph, landmarks: List[str]
) -> np.ndarray:
    result = np.zeros((graph.vert_count, len(landmarks)))
    model, preprocess = clip.load("ViT-L/14")
    model.cuda().eval()

    text_labels = ["A photo of " + desc for desc in landmarks]
    text_tokens = clip.tokenize(text_labels).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for i in range(graph.vert_count):
        rectified_croped_images = [
            rectify_and_crop(np.array(PIL.Image.open(io.BytesIO(img))))
            for img in graph._images[i]
        ]
        processed_images = [preprocess(image) for image in rectified_croped_images]
        image_input = torch.tensor(np.stack(processed_images)).cuda()
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        result[i, :] = np.max(similarity, axis=1)
    return result


def find_optimal_route(
    graph: NavigationGraph, landmarks: List[str], start: int, alpha: float = 0.2
) -> List[int]:
    score = np.full(graph.vert_count, -1e9, dtype=np.float32)
    score[start] = 0.0
    score, prev1 = dijskra_transform(score, graph, alpha)
    prev_tables = [prev1]
    similarity_matrix = nodes_landmarks_similarity(graph, landmarks)
    for i in range(len(landmarks)):
        score += similarity_matrix[:, i]
        score, prev = dijskra_transform(score, graph, alpha)
        prev_tables.append(prev)
    node: int = int(np.argmax(score))
    table_index = len(prev_tables) - 1
    traversal = [(node, 0)]
    while node != start or table_index > 0:
        node, table_index_change = prev_tables[table_index][node]
        table_index += table_index_change
        traversal.append((node, table_index_change))
    return {
        "walk": list(reversed(traversal)),
        "score": score,
        "prev_table": prev_tables,
        "similarity_matrix": similarity_matrix,
    }
