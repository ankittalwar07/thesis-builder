"""thesis-builder: multi-agent investment thesis generator."""

import logging
import os

__version__ = "0.1.0"

# LiteLLM emits noisy WARNINGs about optional providers (Bedrock, SageMaker)
# whose botocore SDK isn't installed. We're not using them, so quiet the logger
# before anything else imports litellm. Users can re-enable with LITELLM_LOG.
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
