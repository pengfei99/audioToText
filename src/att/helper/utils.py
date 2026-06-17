import os

from att.conf.creds import hf_token


def get_hf_token() -> str | None:
    """
    Retrieve the Hugging Face token from environment variables or fallback.

    :return: The token string if found, otherwise None.
    """
    token = os.environ.get("HF_TOKEN") or hf_token
    return token if token else None
