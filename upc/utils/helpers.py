import json
from services.ai_client_service import make_classification_call
from utils.prompts import PRODUCT_CONVERSATION_CLASSIFIER_PROMPT
from upc.logger import setup_logger
import time
logger = setup_logger(__name__)


def is_product_related(message: str) -> bool:
    """
    Check if the given message is related to product creation using LLM.
    Also measures and logs the time taken for classification.
    """
    start_time = time.time()
    logger.info("Starting product classification check")

    prompt = PRODUCT_CONVERSATION_CLASSIFIER_PROMPT.format(message=message)
    response_data = None
    response = None

    try:
        logger.debug(f"Sending classification prompt to AI: {prompt[:100]}...")
        classification_start = time.time()
        response = make_classification_call(prompt)
        classification_end = time.time()
        logger.info(f"Classification API call took {classification_end - classification_start:.2f} seconds")

        logger.debug(f"Received classification response: {response}")

        parsing_start = time.time()
        response_data = json.loads(response)
        classification = response_data['classification']
        parsing_end = time.time()
        logger.info(f"JSON parsing took {parsing_end - parsing_start:.2f} seconds")

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total classification process took {total_time:.2f} seconds")
        logger.info(f"Message classified as: {classification}")

        return classification == 'product_related'

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {str(e)}")
        logger.debug(f"Raw response that caused the error: {response}")
        end_time = time.time()
        logger.error(f"Process failed after {end_time - start_time:.2f} seconds")
        return True

    except KeyError as e:
        logger.error(f"KeyError in classification response: {str(e)}")
        logger.debug(f"Response data that caused the error: {response_data}")
        end_time = time.time()
        logger.error(f"Process failed after {end_time - start_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Unexpected error in classification: {str(e)}", exc_info=True)
        end_time = time.time()
        logger.error(f"Process failed after {end_time - start_time:.2f} seconds")
        return True