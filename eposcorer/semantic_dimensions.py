from typing import List, Optional
import numpy as np
import torch
import collections


class SemanticDimension:
    """Computes a semantic transformation vector that can be used for scoring
    text documents using an instance of the SemanticScorer class.

    Use find_pole_parameters() first if you are unsure about the proper
    arguments to use.

    Attributes:
        self.sent_pos: A sentence or expression representing the "positive"
            pole of the semantic dimension.
        self.sent_neg: A sentence or expression representing the "negative"
            pole of the semantic dimension.
        self.token_1_pos: The position of the first token of the term defining
            the positive pole of the semantic dimension.
        self.token_n_pos: The position of the last token of the term defining
            the positive pole of the semantic dimension.
        self.token_1_neg: See token_1_pos but for the negative pole.
        self.token_n_pos: See token_n_pos but for the negative pole.
        self.dimension_name: A short string to identify the semantic
            dimension by.
        self.semantic_dimension: A namedtuple with two fields:
            a short string to identify the semantic dimension by and the
            corresponding transformation vector.
    """

    def __init__(
        self,
        sent_pos: str,
        sent_neg: str,
        tokens_pos: List[tuple[str,int,int]],
        tokens_neg: List[tuple[str,int,int]],
        dimension_name: str,
        language_model: TransformersModel,
        layers: Optional[List[int]] = [-1],
        concatenate_blocks: Optional[bool] = True,

    ):
        self.sent_pos = sent_pos
        self.sent_neg = sent_neg
        self.tokens_pos = tokens_pos
        self.tokens_neg = tokens_neg
        self.dimension_name = dimension_name
        self.language_model = language_model
        self.semantic_dimension = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.concatenate_blocks = concatenate_blocks
        

    def __call__(self):
        """Computes the semantic transformation vector.

        Args:
            None, only uses attributes.
        """

        self.positive_pole = self.get_pole_embedding(
            self.sent_pos,
            self.tokens_pos,
        )

        self.negative_pole = self.get_pole_embedding(
            self.sent_neg,
            self.tokens_neg,
        )

        dimension_vector = self.compute_dimension_vector()
        
        self.get_max_min(dimension_vector)

        semantic_dimension_transform = collections.namedtuple(
            "semantic_dimension", ["dimension",
                                   "transform_vector",
                                   "max_value",
                                   "min_value"])
        
        self.semantic_dimension = semantic_dimension_transform(
            self.dimension_name,
            dimension_vector,
            self.pos_score,
            self.neg_score
            )


    def get_pole_embedding(
        self,
        sent: str,
        tokens_list: List[tuple[str,int,int]]
        ):

        """Returns the mean_embedding of a semantic pole.
        """

        self.language_model.model.to(self.device)

        tensor_list = []

        for tokens_tuple in tokens_list:
            pole_start_token = tokens_tuple[1]
            pole_n_tokens = tokens_tuple[2]
            pole_tokens = sent + tokens_tuple[0]
            sent_encoded = self.language_model.tokenizer(pole_tokens, return_tensors="pt",
                                                    padding=True).to(self.device)
            print(sent_encoded.tokens())
            # I retrieve the contextual embeddings for the encoded tokens
            # and average or concatenate them over the requested layers of the model.

            with torch.no_grad():
                output = self.language_model.model(**sent_encoded)
                states = output.hidden_states

            if self.concatenate_blocks:
                # Concatenate embeddings from specified layers:
                output = torch.cat([states[i] for i in self.layers], dim=-1).squeeze()
            else:
                # Average embeddings from specified layers:
                output = torch.stack([states[i] for i in self.layers]).sum(0).squeeze()

            if(pole_n_tokens > 1): # True if dimension consists of only one token.
                tokens_for_pole = output[pole_start_token : pole_start_token + pole_n_tokens] 
                mean_embedding = torch.mean(tokens_for_pole, dim = 0, keepdim = True)
                
            else:
                mean_embedding = output[pole_start_token : pole_start_token + pole_n_tokens]

            tensor_list.append(torch.transpose(mean_embedding, 0, 1))

        pole_embedding = torch.mean(torch.stack(tensor_list), dim=0)

        return pole_embedding


    def compute_dimension_vector(self):
        
        midpoint = (self.positive_pole + self.negative_pole) / 2
        self.positive_pole = self.positive_pole - midpoint
        self.negative_pole = self.negative_pole - midpoint
            
        difference_vec = (self.positive_pole - self.negative_pole).cpu()
        norm = np.linalg.norm(difference_vec)
        difference_unit_vec = (difference_vec / norm)
        dimension_vector = np.transpose(difference_unit_vec)

        return dimension_vector


    def get_max_min(
        self,
        dimension_vector
        ):

        """Returns the semantic scores of the poles themselves for scaling
           scores between 0 and 1.
        """

        self.pos_score = np.tensordot(dimension_vector, self.positive_pole.cpu(), axes = 1)
        self.neg_score = np.tensordot(dimension_vector, self.negative_pole.cpu(), axes = 1)

        print(self.pos_score)
        print(self.neg_score)

        return None