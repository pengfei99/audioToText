import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, model_info
from huggingface_hub.errors import RepositoryNotFoundError, HFValidationError, HfHubHTTPError

from att.helper.log_manager import setup_logger
from att.helper.utils import get_hf_token
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
        info = model_info(model_id, token=get_hf_token())
        task = info.pipeline_tag or "unknown task"
        logger.info(f"Model exists: {info.modelId} ({task})")
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
    if not is_model_id_valid(model_id):
        return ""
    parts = [part.strip() for part in model_id.split("/") if part.strip()]
    return parts[-1] if parts else ""


def download_model(
        model_id: str,
        local_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        force_download: bool = False,
) -> Optional[Path]:
    """
    Download a model from Hugging Face using snapshot_download.

    Args:
        model_id: Hugging Face model repository ID
        local_path: Custom directory to save the model (default: MODEL_ROOT_DIR/model_name)
        hf_token: Hugging Face token (falls back to HF_TOKEN env var)
        force_download: Force re-download even if files exist

    Returns:
        Path to the downloaded model directory or None if failed
    """
    model_name = get_model_name(model_id)
    if not model_name:
        logger.error("Failed to get valid model name. Aborting download.")
        return None

    # Determine target directory
    if local_path:
        base_dir = Path(local_path)
        if not base_dir.is_dir():
            logger.error(f"Invalid local model path: {base_dir}. Must be an existing directory.")
            return None
        local_model_path = base_dir / model_name
    else:
        local_model_path = MODEL_ROOT_DIR / model_name

    local_model_path.mkdir(parents=True, exist_ok=True)

    # Use token from parameter, env, or huggingface_hub default
    token = get_hf_token()

    logger.info(f"Starting download of {model_id} → {local_model_path}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_model_path.as_posix(),
            token=token,
            force_download=force_download,
            # resume_download is now default behavior, no need to set
            allow_patterns=None,  # Add e.g. ["*.safetensors", "*.json"] if you want to filter
            ignore_patterns=None,  # e.g. ["*.pt", "*.bin"] to skip old formats
        )

        logger.info(f"✅ Download completed successfully: {local_model_path}")
        return local_model_path

    except Exception as e:
        logger.error(f"❌ Download failed for {model_id}: {type(e).__name__} - {e}")
        return None


if __name__ == "__main__":
    log_dir = Path(__file__).parent.parent.parent / "logs"
    print(log_dir.as_posix())
    setup_logger(log_dir=log_dir)
    model_name = "bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mistralai_Voxtral-Mini-3B-2507-Q5_K_S.gguf"
    download_model(model_name)
