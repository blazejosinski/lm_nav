import numpy as np

from lm_nav import optimal_route
from lm_nav import landmark_extraction


def full_pipeline(graph, start_node, landmarks = None, instructions = None, alpha=0.0002, debug=True):
    if landmarks is None:
        assert instructions is not None, "If landmarks is not provided, instructions must be provided"
        landmarks = landmark_extraction.text_to_landmarks_gpt3(instructions)

    walk_and_metadata = optimal_route.find_optimal_route(
        graph, landmarks, start_node, alpha=alpha
    )
    supplementary_data = {
        str(i): [
            "score: " + str(walk_and_metadata["score"][i]),
            list(graph._graph.neighbors(i)),
        ]
        for i in range(graph.vert_count)
    }

    if debug:
        distances = np.full(graph.vert_count, -1e9)
        distances[start_node] = 0
        nd = optimal_route.dijskra_transform(distances, graph, alpha=1)
        for k, l in supplementary_data.items():
            l.append("dist: " + str(nd[0][int(k)]))
    walk_and_metadata["supplementary_data"] = supplementary_data
    walk_and_metadata["landmarks"] = landmarks

    return walk_and_metadata
