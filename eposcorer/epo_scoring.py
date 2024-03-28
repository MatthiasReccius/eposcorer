from typing import NamedTuple, List, Optional
import transformers
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
import torch
import collections
import os
import h5py
import gc


class EPOScorer():

    """Performs a semantic scoring algorithm on a collection of documents.
    Semantic scoring involves computing document-level embeddings
    and then comparing them to a pre-defined semantic dimension.
    Attributes:
        self.semantic_dimension: An instance of the SemanticDimension
            class specifying the pre-defined semantic dimension
        self.docs: A list of namedtuples containing two fields:
            A string denoting a document id and a list of strings,
            each item containing a sequence from the corresponding document.
        self.path: The path to the .csv-file the semantic scores
            will be written to.
        self.store_embeddings: A boolean indicating if the
            sequence-level embeddings should get written to disc.
        self.embeddings_to_file_name: The name of the .hdf5-file the
            aggregated sequence-level embedding  will be written to.
            The same folder as self.path will be used. If not specified,
            embeddings will not be stored.
        _self.pooling: The pooling method used to aggregate the token-level
            embeddings. Can either be "max_pooling" or "mean_pooling".
            "mean_pooling": embeddings are aggregated to the document-level
            before scoring.
            "max_pooling": token-level embeddings are scored individually, the
            token with the maximum absolute value is chosen to represent its
            sentence. The mean of these sentence-level values is the
            document-level score.
        self.threshold: A value between 0 and 1 that represents the max share of
            GPU memory that may be occupied before the cache is cleaned.
    """

    def __init__(
        self,
        semantic_dimension,
        path: str,
        concatenate_blocks: Optional[bool] = True,
        docs: Optional[List[NamedTuple]] = None,
        embeddings_path: Optional[str] = None,
        embeddings_to_file_name: Optional[str] = False,
        append_embeddings: Optional[bool] = False,
        storage_interval: Optional[int] = 100,
        threshold: Optional[float] = 0.9,
        weight_method: Optional[str] = "sentence_weight",
        sentence_level_scoring: Optional[bool] = False,
    ):
        """
        Raises:
            TypeError: If semantic_dimension is not of type SemanticDimension.
            ValueError: If neither docs nor embeddings are provided.
        """

        if not docs and not embeddings_path:
          raise ValueError("Either docs or embeddings must be provided.")

        self.semantic_dimension = semantic_dimension
        self.docs = docs
        self.path = path
        self.embeddings_path = embeddings_path
        self.embeddings_to_file_name = embeddings_to_file_name
        self.append_embeddings = append_embeddings
        self.storage_interval = storage_interval
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.semantic_scores_docs = []
        self.threshold = threshold
        self.weight_method = weight_method
        self.sentence_level_scoring = sentence_level_scoring
        self.concatenate_blocks = concatenate_blocks
        self.seq_n_sentences = []

        self.embeddings_storage = collections.namedtuple("doc",
         ["doc_id", "embeddings_list", "seq_length_list"])

        print(sentence_level_scoring)
        
        self.encoding_test = []

        self.check_file_path()

        if not self.embeddings_path and self.embeddings_to_file_name:
            self.embeddings_buffer = []
            self.embeddings_file_path = os.path.join(os.path.dirname(self.path),
                                                self.embeddings_to_file_name)

        if not isinstance(self.semantic_dimension, SemanticDimension):
            raise TypeError("The semantic_dimension must be an instance " +
                            "of the SemanticDimension class. Create " +
                            "an instance of SemanticDimension " +
                            "with the desired parameters to continue.")


    def __call__(self):
        """A wrapper function that executes the semantic scoring algorithm
        Args:
            None, only uses attributes:
        """

        language_model = self.semantic_dimension.language_model

        # In the standard usecase, ie. if no embeddings are provided,
        # semantic scoring is preceeded by tokenization of the documents,
        # computation of token-level embeddings and aggregation to the
        # sequence-level:

        if self.embeddings_file_path and not self.append_embeddings:
        
            with h5py.File(self.embeddings_file_path, 'w') as hdf5_file:
                pass

        if not self.embeddings_path:
            
            print("Encoding and scoring documents...")
            docs = tqdm(self.docs, leave=True)

            model = language_model.model.to(self.device)

            for doc_i, doc in enumerate(docs):
                # Copy the list of sentences to a local variable
                # to not change the original data during processing:
                doc_sents = doc.sents

                # If sentence_level_scoring is active, set length_maximized
                # parameter to True to avoid sentences getting merged to
                # maximize the length of scored text sequences:
                if self.sentence_level_scoring:
                    self.length_maximized = True

                else:
                    self.length_maximized = False

                try:
                    doc_encoded, doc_sents = self.tokenize_text(doc_sents,
                                                             language_model)

                except Exception as e:
                    print(f"Exception occurred during tokenization: {e}")
                    continue

                doc_embeddings = self.compute_embeddings(
                    doc_encoded, doc.id, doc_sents, model, language_model)

                doc_embedding = self.aggregate_embeddings(doc_embeddings)

                if self.embeddings_to_file_name:
                    self.embeddings_buffer.append(doc_embedding)

                semantic_score, score_storage = self.score_doc(doc_embedding)

                self.semantic_scores_docs.append(semantic_score)

                if self.embeddings_to_file_name and doc_i % self.storage_interval == (self.storage_interval - 1):
                    self.store_embeddings()

                if self.device == "cuda:0":
                    self.check_and_clean_gpu_cache()

        else: # If embeddings are provided, the pipeline reduces to scoring:

            print("Scoring documents...")
            with h5py.File(self.embeddings_path, 'r') as f:

                for key in f.keys():
                    doc = f[key]
                    doc_id = doc['doc_id'][()].decode('utf-8')
                    doc_level_embedding = torch.tensor(doc['embeddings'][()]).to(self.device)

            # for doc in embeddings:
                    doc_embedding = self.embeddings_storage(doc_id,
                                                doc_level_embedding,
                                                None)
                    semantic_score, score_storage = self.score_doc(doc_embedding)
                    self.semantic_scores_docs.append(semantic_score)

                    if self.device == "cuda:0":
                      self.check_and_clean_gpu_cache()

        self.save_scores(score_storage)
        self.store_embeddings()


    def check_file_path(self):
        """Performs checks on the write path for scores and embeddings.

        Args:
            None, only uses attributes:
        Raises:
            OSError: If self.path either does not exist, does not grant write
                permissions or is not of the required .csv extension.
        """

        if not os.path.exists(os.path.dirname(self.path)):
            path_error = (f"Your 'path' '{os.path.dirname(self.path)}' " +
                          "does not exist.")
            raise OSError(path_error)

        elif not os.access(os.path.dirname(self.path), os.W_OK):
            write_error = ("You do not have write permissions for the 'path' " +
                          f"'{os.path.dirname(self.path)}'.")
            raise OSError(write_error)

        elif not self.path.endswith('.csv'):
            csv_error = ("Your 'path' should be of csv format.")
            raise OSError(csv_error)

        return None


    def tokenize_text(
        self,
        doc_sents,
        language_model): # -> AutoTokenizer.from_pretrained:
        """Splits sentences into tokens using the tokenizer in language_model.

        Args:
            doc_sents: The field sents from the self.doc attribute containing
                a list of all sequences in doc.
            language_model: A namedtuple that contains model and tokenizer.

        Returns:
            doc_encoded: An instance of the AutoTokenizer.from_pretrained class
                from the transformer library. Contains, most prominently,
                the tokens and token_ids from doc_sents.
        Raises:
            TypeError: If doc_sents is not a list.
        """

        max_seq_length = language_model.model.config.max_position_embeddings - 2

        if not isinstance(doc_sents, list):
            type_error = ("'doc' must be a NamedTuple that contains the " +
                          "field sents which holds a list of all sequences " +
                          "in the document.")
            raise TypeError(type_error)

        doc_encoded = language_model.tokenizer(doc_sents, return_tensors="pt",
                                                  padding=True).to(self.device)
        attention = doc_encoded.get("attention_mask")
        seq_length = len(attention[0])

        if seq_length > max_seq_length:
            doc_sents = self.exclude_long_texts(
                attention, doc_sents, max_seq_length,
                language_model)
            return self.tokenize_text(doc_sents, language_model)

        if self.length_maximized == False:
            doc_sents = self.maximize_sequence_length(doc_sents, attention,
                                                      language_model)
            return self.tokenize_text(doc_sents, language_model)

        else:
            return (doc_encoded, doc_sents)


    def exclude_long_texts(
        self,
        attention,
        doc_sents,
        max_seq_length,
        language_model):
        """Deletes sequences that exceed the max sequence length of the model.
        Passes a cleaned list of sequences back to self.compute_embeddings()
        Args:
            attention: The tokenizers attention_mask attribute, used to identify
                the true length of each sequence (minus padding at the end)
            doc: A namedtuple containing two fields: A string of a document id
                and a list of strings with each item containing a sequence
                from the corresponding document.
            language_model: A namedtuple that contains model and tokenizer.
        """

        for i in range(0, len(doc_sents)):
            n_tokens = int(torch.count_nonzero(attention[i]))

            if n_tokens > max_seq_length:
              doc_sents[i] = ""

        return (doc_sents)


    def maximize_sequence_length(
        self,
        doc_sents,
        attention,
        language_model):
        """Merges consecutive sequences until each merged sequence exhausts
        the max sequence length of the model.
        Passes the merged sequences back to self.compute_embeddings()
        Args:
            doc_encoded: An instance of AutoTokenizer.from_pretrained from the
                transformer library. Contains, most prominently, the tokens from
                the document.
            attention: The tokenizers attention_mask attribute, used to identify
                the true length of each sequence (minus padding at the end)
            doc: A namedtuple containing two fields: A string of a document id
                and a list of strings with each item containing a sequence
                from the corresponding document.
            language_model: A namedtuple that contains model and tokenizer.
        """
        self.seq_n_sentences = []

        max_seq_length = language_model.model.config.max_position_embeddings - 2

        doc_len = len(doc_sents)
        n_sentences = 0
        sequence_n_tokens = 0
        merged_docs_list = []
        i = 0

        # loop over each sentence in doc and check its length:
        while i < doc_len:
            n_tokens = int(torch.count_nonzero(attention[i]))

            # Check if sentence i is smaller than the models max length:
            if n_tokens < max_seq_length:
                sequence_n_tokens += n_tokens

                # Check if the sequence is still smaller or equal to
                # the models max length. If so, include sentence i:
                if sequence_n_tokens <= max_seq_length:
                    n_sentences += 1
                    i += 1
                    merged_doc = " ".join(doc_sents[(i - n_sentences):i])
                    doc_encoded = language_model.tokenizer(merged_doc, return_tensors="pt",
                                              padding=True).to(self.device)
                    test_attention = doc_encoded.get("attention_mask")
                    n_tokens = int(torch.count_nonzero(test_attention[0]))


                # If the sequence would be larger than the models max length,
                # exclude sentence i, merge the sequence and store it in
                # merged_docs_list:
                else:
                    sequence_n_tokens -= n_tokens
                    merged_doc = " ".join(doc_sents[(i - n_sentences):n_sentences])
                    merged_docs_list.append(merged_doc)
                    self.seq_n_sentences.append(n_sentences)
                    n_sentences = 0
                    sequence_n_tokens = 0

            # If sentence i by itself is larger than or equal
            # to the models max length, check if there is a previous sequence
            # that must be stored. If so, merge that sequence and store it in
            # merged_docs_list. Also, store sentence i in merged_docs_list:
            elif n_sentences > 0:
                merged_doc = " ".join(doc_sents[(i - n_sentences):n_sentences])
                merged_docs_list.append(merged_doc)
                self.seq_n_sentences.append(n_sentences)
                sequence_n_tokens = n_tokens
                n_sentences = 1
                merged_docs_list.append(doc_sents[i])
                self.seq_n_sentences.append(n_sentences)
                i += 1

            # If there is no previous sequence in memory,
            # simply store sentence i in merged_docs_list:
            else:
                n_sentences = 1
                merged_docs_list.append(doc_sents[i])
                self.seq_n_sentences.append(n_sentences)
                i += 1

        # Check whether iteration i - 1 is the last sentence in doc.
        # If so merge the sequence and store it in merged_docs_list.
        if i == (doc_len):
            merged_doc = " ".join(doc_sents[(i - n_sentences):i])
            self.seq_n_sentences.append(n_sentences)
            merged_docs_list.append(merged_doc)

            doc_encoded = language_model.tokenizer(merged_docs_list, return_tensors="pt",
                                              padding=True).to(self.device)
            attention = doc_encoded.get("attention_mask")

        self.length_maximized = True

        return (merged_docs_list)


    def compute_embeddings(
        self,
        doc_encoded,
        doc_id,
        doc_sents,
        model,
        language_model):
        """Computes the embedding vectors for the documents.

        Uses attention_mask to exclude excess tokens that were added to the
        encoded as padding in the tokenize_text method.

        Args:
            doc_encoded: An instance of the AutoTokenizer.from_pretrained class
                from the transformer library. Contains, most prominently,
                the tokens and token_ids from doc.
            doc: A namedtuple containing two fields: A string of a document id
                and a list of strings with each item containing a sequence
                from the corresponding document.
            language_model: A namedtuple that contains model and tokenizer.
        Returns:
            doc_embeddings: a namedtuple with three fields: a document ID,
                a nested list of token-level embeddings per sequence and
                a list containing the length (ie. weight) of each sequence.
            embeddings_storage: A namedtuple subclass designed to store the
                computed document embeddings.
        """

        embeddings_list = []
        seq_length_list = []

        layers = self.semantic_dimension.layers
        max_seq_length = language_model.model.config.max_position_embeddings - 2

        attention = doc_encoded.get("attention_mask")
        seq_length = len(attention[0])

        if seq_length > 510:
            for i in range(0, len(doc_sents)):
                n_tokens = int(torch.count_nonzero(attention[i]))

        try:
            with torch.no_grad():

                output = model(**doc_encoded)
                # states is a tuple with as many elements as layers were specified:
                states = output.hidden_states

                # if not self.include_cls_token:
                # Process each layer to exclude the CLS token
                states = tuple(layer[:, 1:, :] for layer in states)

                if self.concatenate_blocks:
                    # Concatenate embeddings from specified layers:
                    output = torch.cat(
                        [states[l] for l in layers], dim=-1).squeeze()
                else:
                    # Average embeddings from specified layers:
                    output = torch.stack(
                        [states[l] for l in layers]).sum(0).squeeze()

        except Exception as e:
            doclength_error = (f"\n\nA sequence in document {doc_id}" +
                              " was too long for the model. The sequence was" +
                              f" omitted.\n Full error message:\n{e}\n\n")
            print(doclength_error)

            merged_doc_list = self.exclude_long_texts(attention, doc_sents,
                                                      max_seq_length,
                                                      language_model)

            doc_encoded = self.tokenize_text(merged_doc_list,
                                              language_model)


        else:
            try:
                # if the encoded list only contained one document, we must
                # enhance the shape of the output to feature the
                # number of documents in dimension 0:
                if output.dim() == 2:
                    output = output.unsqueeze(0)  # adds the document dimension

                # Iterate over the list of aggregated sentences:
                for j in range((output.shape[0])):
                    n_tokens = int(torch.count_nonzero(attention[j]))
                    embeddings = output[j][:n_tokens, ]
                    embeddings_list.append(embeddings)
                    seq_length_list.append(n_tokens)

            except Exception as e:
                doc_error = (f"\n\nThere was an issue with document {doc_id}." +
                            f" Full error message:\n{e}\n\n")

                print(doc_error)

        doc_embeddings = self.embeddings_storage(doc_id, embeddings_list,
                                            seq_length_list)

        return doc_embeddings


    def aggregate_embeddings(
        self,
        doc_embeddings):
        """Computes sequence-level embeddings from token-level embeddings

        Args:
          doc_embeddings: a namedtuple with three fields: a document ID,
              a nested list of token-level embeddings per sequence and
              a list containing the length (ie. weight) of each sequence.
          embeddings_storage: A namedtuple subclass that can store
              document embeddings and the corresponding document id.
        Returns:
          A namedtuple with two fields: the document ID and a list of average
          sequence-level word embeddings.
        """

        # Store the length of the word embeddings in an aux variable
        emb_len = len(self.semantic_dimension.semantic_dimension.transform_vector)

        sequence_level_embeddings = []

        token_level_embeddings = doc_embeddings.embeddings_list

        for sequence in token_level_embeddings:
            mean_embedding = torch.mean(sequence, 0)
            sequence_level_embeddings.append(mean_embedding)

        if self.weight_method == "token_weight":
            weights = doc_embeddings.seq_length_list

        elif self.weight_method == "sentence_weight":
            weights = self.seq_n_sentences

        try:
            # Normalize weights and convert from a list to a tensor:
            seq_length_tensor = torch.Tensor(weights)
            weights_tensor = seq_length_tensor / seq_length_tensor.sum()
            weights_tensor = weights_tensor.to(self.device)
            # Convert embeddings to tensor and compute weighted average:
            embeddings_tensor = torch.stack(sequence_level_embeddings)

        except RuntimeError as e:
            embeddings_tensor = torch.zeros(emb_len).to(self.device)
            doc_embeddings = self.embeddings_storage(doc_embeddings.doc_id,
                                        embeddings_tensor,
                                        doc_embeddings.seq_length_list
                                          )

            return doc_embeddings


        # Compute the doc embedding as a weighted sum by adding ip
        # sequence-level tensor using sequence lengths as weights:
        if not self.sentence_level_scoring:

            doc_level_embedding = torch.einsum('ij,i->j',
                                              embeddings_tensor,
                                              weights_tensor)
            doc_embeddings = self.embeddings_storage(doc_embeddings.doc_id,
                                                    doc_level_embedding,
                                                    doc_embeddings.seq_length_list)

        else:
            doc_embeddings = self.embeddings_storage(doc_embeddings.doc_id,
                                                    embeddings_tensor,
                                                    doc_embeddings.seq_length_list
                                                     )

        return doc_embeddings


    def score_doc(
        self,
        doc_embeddings):
        """Executes the semantic scoring algorithm on (aggregated) embeddings

        ATTENTION: max_pooling scores are not scaled to between 0 and 1 yet!

        Args:
            doc_embedding: A namedtuple with two fields: the document ID and
                a list of average sequence-level word embeddings.
        Returns:
            semantic_score: A namedtuple with two fields:
                the document ID and the score.
            score_storage: A namedtuple subclass
        """

        transform_vector = self.semantic_dimension.semantic_dimension.transform_vector.to(
            self.device)
        dimension_name = str(
            self.semantic_dimension.semantic_dimension.dimension)
        max_val = self.semantic_dimension.semantic_dimension.max_value
        min_val = self.semantic_dimension.semantic_dimension.min_value
        score_storage = collections.namedtuple(
            "semantic_score", ["doc_id", dimension_name])

        doc_score = []

        if doc_embeddings.embeddings_list is not None:

            # Normalize every embedding in the embeddings_list
            doc_score = torch.nn.functional.cosine_similarity(
                transform_vector, doc_embeddings.embeddings_list, dim = -1)
            doc_score = doc_score.cpu().numpy()
            
            if len(doc_score == 1):
                doc_score = doc_score[0]

        else:
            doc_score = None

        semantic_score = score_storage(doc_embeddings.doc_id, doc_score)

        return(semantic_score, score_storage)


    def save_scores(
        self,
        score_storage):
        """Writes the document IDs and the corresponding scores to a csv-file.
        Args:
            score_storage: A namedtuple subclass
        """

        # Convert layers list to a string with elements separated by a semicolon
        # to allow for proper representation in the csv file
        layers_str = ' '.join(map(str, self.semantic_dimension.layers))

        metadata = ("# Model parameters:\n " +
                    "# model_path: " +
                    self.semantic_dimension.language_model.model_path + "\n" +
                    "# SemanticDimension parameters:\n " +
                    "# sent_pos: " +
                    self.semantic_dimension.sent_pos + "\n" +
                    "# sent_neg: " +
                    self.semantic_dimension.sent_neg + "\n" +
                    "# tokens_pos: " +
                    str(self.semantic_dimension.tokens_pos) + "\n" +
                    "# tokens_neg: " +
                    str(self.semantic_dimension.tokens_neg) + "\n" +
                    "# sentence_level_scoring: " +
                    str(self.sentence_level_scoring) + "\n" +
                    "# concatenate_blocks: " +
                    str(self.concatenate_blocks) + "\n" +
                    "# tokens_pos: " +
                    str(layers_str) + "\n\n")

        with open(self.path, "w", newline = "") as f:
            f.write(metadata)
            writer = csv.writer(f)
            writer.writerow(score_storage._fields)
            writer.writerows(self.semantic_scores_docs)

        return None


    def store_embeddings(self):
        """Writes the document IDs and the doc-level embeddings to an hdf5-file.
        Args:
            None, only uses attributes:
        """

        language_model = self.semantic_dimension.language_model.model_path
        layers = self.semantic_dimension.layers

        with h5py.File(self.embeddings_file_path, "a") as f:
            f.attrs["model"] = language_model
            f.attrs["layers"] = layers

            for i, doc in enumerate(self.embeddings_buffer):
                doc_id = str(doc.doc_id) # Convert id to str to write to file
                grp = f.create_group(doc_id)
                grp.create_dataset('doc_id', data = doc_id.encode('utf-8'))
                grp.create_dataset('embeddings',
                                   data = doc.embeddings_list.cpu().numpy())
            
            # Clear embeddings buffer:
            self.embeddings_buffer = []

        return None


    def check_and_clean_gpu_cache(self):
        """Helper function that checks available GPU memory.
        Also cleans CUDA cache if necessary.
        """

        gpu = torch.cuda
        mem_allocated = gpu.memory_allocated() / (1024**2)
        mem_cached = gpu.memory_reserved() / (1024**2)
        total_mem = gpu.get_device_properties(0).total_memory / (1024 ** 2)
        available_mem = total_mem - mem_allocated - mem_cached

        if (mem_allocated + mem_cached) > (self.threshold * available_mem):
            gc.collect()
            gpu.empty_cache()

        return None