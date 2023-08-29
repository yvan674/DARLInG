"""Known Domain Agent.

Produces embeddings based on the info provided.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch


class KnownDomainAgent:
    def __init__(self, domain_embedding_size: int):
        self.domain_embedding_size = domain_embedding_size
        self.device = None

    def __repr__(self):
        return "KnownDomainAgent()"

    def predict(self, info: dict[str, list[any]], **kwargs) -> torch.Tensor:
        """Produce an action based on the info dictionary.

        The info dictionary looks like:
            {
                'user': [2,...],
                'room_num': [1,...],
                'date': ['20181109',...],
                'torso_location': ['1',...],
                'face_orientation': ['2',...],
                'gesture': ['4',...]
            }.
        We use everything except the date and gesture as part of our known
        domain embedding. The values are all transformed to a one-hot encoding.
        The one-hot encoding is hardcoded, as we know what the info dictionary
        always looks like.
        """
        # 17 users, 3 rooms, 8 torso locations, 5 face orientations
        # Total length of 33
        batch_size = len(info["user"])
        embedding = torch.zeros((batch_size, 33), device=self.device)
        # Extract required keys
        info = {k: info[k]
                for k in ("user", "room_num", "torso_location",
                          "face_orientation")}
        # Not the usual one-hot way since this is kinda weirdly formed
        for i in range(len(info["user"])):
            user_val = info["user"][i]
            room_val = 17 + info["room_num"][i]
            torso_val = 17 + 3 + info["torso_location"][i]
            face_val = 17 + 3 + 8 + info["face_orientation"][i]
            embedding[i, [user_val, room_val, torso_val, face_val]] = 1.

        return embedding

    def save(self, fp):
        pass

    def learn(self, *args):
        pass

    @staticmethod
    def load_state_dict(sd: dict[any]):
        return KnownDomainAgent(sd["domain_embedding_size"]).to(sd["device"])

    def to(self, device: int | torch.device | None):
        self.device = device
