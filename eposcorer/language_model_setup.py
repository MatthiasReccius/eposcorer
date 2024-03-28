from typing import List, Optional
import transformers


class TransformersModel():
    """Provides a wrapper for a pre-trained language model from transformer.co.

    Currently, the program expects the attribute attention_mask, so it
    requires a model that uses padding during tokenization. Defaults to the
    DistilRoBERTa model the function has been tested with extensively.

    Attributes:
        self.model: A checkpoint of a pre-trained model that contains the
            model weights needed to compute word embeddings.
        self.tokenizer: The tokenizer that corresponds to self.model.
        self.model_path: The path to the model checkpoint hosted on
            huggingface.co. Defaults to a distilroberta model fine-tuned on a
            sentiment classification task using financial news.
        self.layers: Specifies the transformer blocks whose output embeddings
            are to be stacked and used for the analyses.
    """

    def __init__(
        self,
        model_path: Optional[str] = ("mrm8488/distilroberta-finetuned" +
                                    "-financial-news-sentiment-analysis"),
    ):
        """Initializes the instance based on the language model and layers used.

        Args:
            None, only uses attributes.
        """

        self.model_path = model_path

        self.model = transformers.AutoModel.from_pretrained(model_path,
                                                            output_hidden_states
                                                            = True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        return None
