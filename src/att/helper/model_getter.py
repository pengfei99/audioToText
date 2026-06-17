import logging
from pathlib import Path

from huggingface_hub import snapshot_download, model_info
from huggingface_hub.errors import RepositoryNotFoundError, HFValidationError, HfHubHTTPError

from att.helper.log_manager import setup_logger
from src.att.conf.constants import MODEL_ROOT_DIR


logger = logging.getLogger(__name__)

def is_model_id_valid(model_id: str) -> bool:
    """
    This function is used to check if the model id is valid and exists in Hugging Face model repository.
    :param model_id:
    :return:
    """
    if not model_id or not isinstance(model_id, str):
        err_msg = "Invalid model ID (empty or not a string)"
        logger.error(err_msg)
        return False

    try:
        # This will raise an exception if the model doesn't exist or is not accessible
        info = model_info(model_id)
        model_details = f"Model exists: {info.modelId} ({info.pipeline_tag or 'unknown task'})"
        logger.info(model_details)
        return True

    except RepositoryNotFoundError:
        err_msg = f"Model not found: {model_id}"
        logger.error(err_msg)
        return False
    except HFValidationError as e:
        err_msg = f"Invalid model ID format: {e}"
        logger.error(err_msg)
        return False
    except (HfHubHTTPError, Exception) as e:
        # Could be gated model without token, network issue, private repo, etc.
        err_msg = f"Error checking model: {type(e).__name__} - {e}"
        logger.error(err_msg)
        return False


def get_model_name(model_id: str) -> str:
    """
    This function returns the name of the model based on the model id.
    It splits the model name by '/' and returns the last part. If the model id is invalid, return empty string.
    :param model_id: the model repo id in Hugging Face model repository. e.g. Systran/faster-whisper-large-v3
    :return:
    """
    if is_model_id_valid(model_id):
        parts = [part for part in model_id.strip().split('/') if part]
        return parts[-1] if parts else ""
    else:
        return ""


def download_model(model_id: str, local_path: str = None):
    model_name = get_model_name(model_id)
    # if no path is provided, use the default one
    if local_path is None:
        local_model_path = MODEL_ROOT_DIR / f"{model_name}"
    else:
        # use provided path
        local_model_root_dir = Path(local_path)
        # if the provided path is not valid, abort
        if not local_model_root_dir.is_dir():
            err_msg = f"Invalid local model path: {local_model_root_dir}. The given path must be a directory. The downloading is skipped."
            logger.error(err_msg)
            return
        # if the path is valid,
        local_model_path = local_model_root_dir / f"{model_name}"

    logger.info(f"Start the downloading model: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_path.as_posix(),
    )
    logger.info(f"The download is complete. The model is saved at: {local_model_path.as_posix()}")


if __name__ == "__main__":
    setup_logger()
    model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    download_model(model_id)
