from typing import List, Optional
import numpy as np
import torch
import collections


class EPODimension(SemanticDimension):
    '''Computes an EPODimension from two sets of statements that form an 
    antonymical relationship (ie. semantic differentials).
    The EPODimension's transformation vector will be orthogonal to the subspace 
    spanned by the set of SemanticDimensions specified by make_orthogonal_to'''
    
    def __init__(
        self,
        sent_pos: str,
        sent_neg: str,
        tokens_pos: List[tuple[str,int,int]],
        tokens_neg: List[tuple[str,int,int]],
        dimension_name: str,
        language_model: TransformersModel,
        make_orthogonal_to: List[SemanticDimension],
        layers: Optional[List[int]] = [-1],
        concatenate_blocks: Optional[bool] = True,
    ):
        
        super().__init__(
            sent_pos,
            sent_neg,
            tokens_pos,
            tokens_neg,
            dimension_name,
            language_model,
            layers,
            concatenate_blocks,
        )

        self.make_orthogonal_to = make_orthogonal_to


    def __call__(self):
        """Computes the epo transformation vector.

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

        epo_vector = self.compute_dimension_vector()

        self.get_max_min(epo_vector)

        semantic_dimension_transform = collections.namedtuple(
            "semantic_dimension", ["dimension",
                                   "epo_vector",
                                   "max_value",
                                   "min_value"])
        
        self.semantic_dimension = semantic_dimension_transform(
            self.dimension_name,
            epo_vector,
            self.pos_score,
            self.neg_score
            )


    def compute_dimension_vector(
            self,
            ):
        
        midpoint = (self.positive_pole + self.negative_pole) / 2
        self.positive_pole = self.positive_pole - midpoint
        self.negative_pole = self.negative_pole - midpoint
            
        difference_vec = (self.positive_pole - self.negative_pole).cpu()
        norm = np.linalg.norm(difference_vec)
        difference_unit_vec = (difference_vec / norm)
        dimension_vector = np.transpose(difference_unit_vec)

        epo_vector_T = self.orthogonalize(dimension_vector)

        epo_vector = np.transpose(epo_vector_T)

        return epo_vector


    def orthogonalize(
        self,
        dimension_vector
        ):
                    
        U = self.gram_schmidt_orthogonalize(self.make_orthogonal_to)
                        
        self.transformation_matrix = np.matmul(U, U.T)
        
        orth_complement = np.dot(self.transformation_matrix, dimension_vector.T)
        epo_vector_T = dimension_vector.T - orth_complement
        
        return epo_vector_T


    def gram_schmidt_orthogonalize(V):

        '''
        Constructs an orthonormal basis from any number of 
        linearly independent vectors contained in the rows of matrix V
        '''

        # extract the number of rows in V which is equal to 
        # the dimensionality of the subspace V is a basis for
        
        basis_vectors = []

        for semantic_dimension in self.make_orthogonal_to:

            vector = semantic_dimension.semantic_dimension.dimension_vector.T
            basis_vectors.append(vector)

        V = np.hstack(basis_vectors)

        vec_len = V.shape[0]
        dim = V.shape[1]

        # normalize the first row of V to get the first orthonormal basis vector:
        v1 = V[:,0]
        U = v1 / np.linalg.norm(v1)

        for i in range(1, dim):
            v = V[:,i]

            # If the subspace is two-dimensional, the computation simplifies:
            if i == 1:
                proj_v = np.dot(v.T, U) * U

            else:
                # Initialize the projection vector for v onto the current orthonormal basis U
                # Numpy arrays are not dynamically resizable (unlike pd.DataFrames), 
                # so I allocate memory for the final array size beforehand
                proj_v = np.zeros([vec_len,])

                # Compute iteratively the projection proj_j of v onto every 
                # orthonormal basis vector U[j] that was attained so far
                # and add them up to attain the projection proj_v of v onto the entire 
                # current orthonormal basis

                for j in range(i):
                    proj_j = np.dot(v.T, U[:,j]) * U[:,j]
                    proj_v += proj_j

            # Subtract the projection proj_v of v onto the current orthonormal basis and normalize
            # to attain another orthonormal basis vector
                    
            y_new = v - proj_v
            u_dim = y_new / np.linalg.norm(y_new)
            U = np.column_stack([U, u_dim])

        return U