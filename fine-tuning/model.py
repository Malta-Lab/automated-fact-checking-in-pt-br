from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_name, cache_dir="/mnt/E-SSD/model_cache/hf", **kwargs):
    """
    Load the model and tokenizer from the specified model name.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                cache_dir="/mnt/E-SSD/model_cache/hf",
                                                use_fast=True,
                                                )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise