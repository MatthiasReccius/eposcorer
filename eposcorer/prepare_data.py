import re
from typing import Tuple, NamedTuple, List, Optional
from tqdm import tqdm
import pandas as pd
import collections
from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")


def prepare_documents(
		path_to_csv: str,
		doc_column_name,
		id_column_name,
		sentence_sep
) -> Tuple[List[int], List[List[str]]]:

		"""Converts csv-file to a namedtuple of documents and document-ids

		Splits up documents into sentences using a custom function.

		Args:
				path_to_csv: A path to a csv file containing at least two columns:
						documents and unique ids for the documents.
				doc_column_name: The name of the column containing the documents
						in the csv-file
				id_column_name: The name of the column containing the unique ids
						in the csv-file

		Returns:
				list of namedtuples containing two fields:
				the document id and a list of sentences contained in the document
		"""

		docs = pd.read_csv(path_to_csv, on_bad_lines = "skip")

		# Remove newlines (\n), carriage returns (\r), and strip extra whitespace:
		docs[doc_column_name] = (docs[doc_column_name]
								 .str.replace('\n', ' ')
								 .str.replace('\r', ' ')
								 .str.replace(' +', ' ', regex=True)
								 .str.strip())

		docs = sentencize(docs, doc_column_name, id_column_name, sentence_sep)

		return docs



def sentencize(
    docs: List[NamedTuple],
    doc_column_name,
    id_column_name,
    sentence_sep,
) -> List[NamedTuple]:
    """Performs separation of documents into sentences.

    Args:
        docs: A list of namedtuples containing two field: the document id and
            a string of the document.
        doc_column_name: The name of the column containing the documents
						in the csv-file
				id_column_name: The name of the column containing the unique ids
						in the csv-file

    Returns:
        list of namedtuples containing two fields:
        the document id and a list of sentences contained in the document
    """

    doc_column_type = type(docs.iloc[0].loc[doc_column_name])

    if doc_column_type != str:

      doc_column_sample = ast.literal_eval(docs.iloc[0].loc[doc_column_name])
      doc_column_type = type(doc_column_sample)

    sents_single_doc: List[str] = []
    sents_from_docs: List[List[str]] = []
    sents_doc = collections.namedtuple("document", ["id", "sents"])

    docs = list(docs.itertuples(index=False, name="document"))

    # Wrap docs iterable with tqdm for progress bar
    docs = tqdm(docs, desc='Sentencizing')

    if sentence_sep:
        for doc in docs:
            sents_single_doc = []
            id = getattr(doc, id_column_name)
            text = getattr(doc, doc_column_name)
            text = str(text)

            for sent in nlp(text).sents:
                sents_single_doc.append(sent.text)

            single_doc = sents_doc(id, sents_single_doc)
            sents_from_docs.append(single_doc)

    elif doc_column_type == list:
        for doc in docs:
            single_doc = getattr(doc, doc_column_name)
            sents_single_doc = ast.literal_eval(single_doc)
            id = getattr(doc, id_column_name)

            single_doc = sents_doc(id, sents_single_doc)
            sents_from_docs.append(single_doc)

    return sents_from_docs


def exclude_boilerplate(doc_chunks: List[NamedTuple]) -> List[NamedTuple]:
    # Compile the regex pattern for case-insensitive matching
    pattern = re.compile(r'forward[- ]looking', re.IGNORECASE)

    # List to hold results
    results = []

    for doc_chunk in doc_chunks:
        sentences = doc_chunk.sents
        result = []

        for sentence in sentences:
            # If the sentence contains the term, stop capturing and break the loop
            if pattern.search(sentence):
                break
            result.append(sentence)

        # If any relevant sentences were found, create a new DocumentChunk with the same id
        if result:
            results.append(sents_doc(id=doc_chunk.id, sents=result))

    return results