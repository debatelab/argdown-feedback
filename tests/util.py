import warnings

from openai import OpenAI
import pytest



MODEL_KWARGS = {
    "inference_base_url": "http://localhost:8000/v1",
    "model_id": "llama-3.2-3b-instruct"
#    "model_id": "debatelabkit/llama-3.1-argunaut-1-8b-spin-gguf/llama-3.1-argunaut-1-8b-spin-q4_k_m.gguf",
}


def llm_available() -> bool:
    base_url = MODEL_KWARGS["inference_base_url"]
    model_id = MODEL_KWARGS["model_id"]
    try:
        models = OpenAI(api_key="EMPTY", base_url=base_url).models.list()
        avail = model_id in [model.id for model in models.data]
        if not avail:
            warnings.warn(
                UserWarning(
                    f"Model {model_id} not available at local inference server {base_url} (available models are: {[model.id for model in models.data]})"
                )
            )
        return avail
    except Exception as e:
        warnings.warn(
            UserWarning(
                f"Could not connect to local inference server {base_url} (Error: {e})"
            )
        )
        return False
