"""Known Domain Agent.

Produces embeddings based on the info provided.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch

from models.base_embedding_agent import BaseEmbeddingAgent


class KnownDomainAgent(BaseEmbeddingAgent):
    def __init__(self):
        super().__init__()
        self.device = None

    def _produce_action(self, observation: torch.Tensor,
                        info: list[dict[str, any]]) -> torch.Tensor:
        """Produce an action based on the info dictionary.

        The info dictionary looks like:
            {'user': '2', 'room_num': 1, 'date': '20181109',
             'torso_location': '1', 'face_orientation': '2',
             'gesture': '4'}.
        We use everything except the date and gesture as part of our known
        domain embedding. The values are all transformed to a one-hot encoding.
        The one-hot encoding is hardcoded, as we know what the info dictionary
        always looks like.
        """
        # 17 users, 3 rooms, 8 torso locations, 5 face orientations
        # Total length of 33
        embedding = torch.zeros((len(info), 33), device=self.device)
        for i, sample_info in enumerate(info):
            # -1 since the info dictionary is 1-indexed instead of 0-indexed.
            user_val = int(sample_info["user"]) - 1
            room_val = 17 + int(sample_info["room_num"]) - 1
            torso_val = 17 + 3 + int(sample_info["torso_location"]) - 1
            face_val = 17 + 3 + 8 + int(sample_info["face_orientation"]) - 1
            embedding[i, [user_val, room_val, torso_val, face_val]] = 1.

        return embedding

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, sd: dict[any]):
        pass

    def to(self, device: int | torch.device | None):
        self.device = device
