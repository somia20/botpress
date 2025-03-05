from upc.config import Config, ModelType
from upc.exceptions import APICallError
import groq
import anthropic
import instructor
from typing import Literal
from pydantic import BaseModel
from upc.logger import setup_logger
import os
import openai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from upc.metric_logger import track_execution_time
logger = setup_logger(__name__)
dimension = 768  # Dimension of embeddings
faiss_index = faiss.IndexFlatIP(dimension)  # FAISS index for similarity search
embeddings = []  # List to store embeddings
icd_codes = []  # List to store ICD codes
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_client  = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
open_ai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
image_client = None
model = SentenceTransformer("paraphrase-distilroberta-base-v2")
def initialize_clients():
    global client, image_client
    logger.info("Initializing AI clients")
    if Config.SELECTED_MODEL == ModelType.GROQ:
        client = groq.Groq(api_key=Config.get_api_key())
        logger.info("Initialized Groq client")
    elif Config.SELECTED_MODEL == ModelType.OPENAI:
        client = openai.OpenAI(api_key=Config.get_api_key())
        logger.info("Initialized OpenAI client")
    elif Config.SELECTED_MODEL == ModelType.CLAUDE:
        client = anthropic.Anthropic(api_key=Config.get_api_key())
        logger.info("Initialized Claude client")
    else:
        logger.error(f"Invalid model selected: {Config.SELECTED_MODEL.value}")
        raise APICallError(Config.SELECTED_MODEL.value, "Invalid model selected")

    # Always use OpenAI for image processing
    image_client = instructor.patch(openai.OpenAI(api_key=Config.get_api_key(ModelType.OPENAI)))
    logger.info("Initialized OpenAI image client")


def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def get_embedding(text):
    return model.encode(text)


def query_faiss(text,threshold):
    if threshold is None:
        threshold = 0.7
    embedding = get_embedding(text)
    normalized_embedding = normalize_embedding(embedding)
    distances, indices = faiss_index.search(np.array([normalized_embedding]), k=1)


    return icd_codes[indices[0][0]] if distances[0][0] > threshold else None

def add_to_faiss(chief_complaint, icd_code):
    """
    Add a new chief complaint and corresponding ICD code to the FAISS index.

    Args:
        chief_complaint (str): The medical chief complaint text
        icd_code (str): The corresponding ICD code to be stored
    """
    try:
        # Get embedding for the chief complaint
        embedding = get_embedding(chief_complaint)

        # Normalize the embedding
        normalized_embedding = normalize_embedding(embedding)

        # Add to FAISS index
        faiss_index.add(np.array([normalized_embedding]))

        # Store the ICD code
        icd_codes.append(icd_code)

        print(f"Successfully added to FAISS index. Current index size: {len(icd_codes)}")

    except Exception as e:
        print(f"Error adding to FAISS index: {e}")
@track_execution_time()
def make_api_call_general(prompt: str) -> str:
    llm_response = Config.DEFAULT_GENERAL_RESPONSE
    try:
        resp = query_faiss(prompt,Config.GENERAL_CONVERSATION_CACHE_THRESHOLD)
        if resp:
            logger.info("From FAISS Cache")
            return resp
        logger.info(f"Making API call to {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT}")
        if Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == "groq":
            response = client.chat.completions.create(
                model=Config.GENERAL_CONVERSATION_SELECTED_MODEL_GROQ,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            logger.debug(f"Groq API response: {response.choices[0].message.content}")
            llm_response = response.choices[0].message.content
            # return response.choices[0].message.content
        elif Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == 'openai':
            response = open_ai_client.chat.completions.create(
                model=Config.GENERAL_CONVERSATION_SELECTED_MODEL_OPENAI,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            logger.debug(f"OpenAI API response: {response.choices[0].message.content}")
            llm_response = response.choices[0].message.content
            # return response.choices[0].message.content
        elif Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == 'claude':
            response = client.completions.create(
                model=Config.get_model_name(),
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1000
            )
            logger.debug(f"Claude API response: {response.completion}")
            return response.completion

        # adding to cache
        add_to_faiss(prompt, llm_response)
        return llm_response
    except Exception as e:
        logger.error(f"Error in {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT} API call: {str(e)}", exc_info=True)
        raise APICallError(Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT, str(e))
def make_api_call_final_message(prompt: str) -> str:
    llm_response = Config.DEFAULT_FINAL_RESPONSE
    try:
        resp = query_faiss(prompt, Config.GENERAL_CONVERSATION_CACHE_THRESHOLD)
        if resp:
            logger.info("From FAISS Cache")
            return resp
        logger.info(f"Making API call to {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT}")
        if Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT == "groq":
            response = client.chat.completions.create(
                model=Config.MODEL_FINAL_MSG_SELECTED_MODEL_GROQ,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            logger.debug(f"Groq API response: {response.choices[0].message.content}")
            llm_response = response.choices[0].message.content
            # return response.choices[0].message.content
        elif Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT == 'openai':
            response = open_ai_client.chat.completions.create(
                model=Config.MODEL_FINAL_MSG_MODEL_OPENAI,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            logger.debug(f"OpenAI API response: {response.choices[0].message.content}")
            llm_response = response.choices[0].message.content
            # return response.choices[0].message.content
        elif Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == 'claude':
            response = client.completions.create(
                model=Config.get_model_name(),
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1000
            )
            logger.debug(f"Claude API response: {response.completion}")
            return response.completion

        # adding to cache
        add_to_faiss(prompt, llm_response)
        return llm_response
    except Exception as e:
        logger.error(f"Error in {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT} API call: {str(e)}", exc_info=True)
        raise APICallError(Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT, str(e))
def make_api_call_greeting(prompt: str) -> str:
    try:
        logger.info(f"Making API call to {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT.value}")
        if Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == "groq":
            response = client.chat.completions.create(
                model=Config.get_model_name(),
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200
            )
            logger.debug(f"Groq API response: {response.choices[0].message.content}")
            return response.choices[0].message.content
        elif Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == "openai":
            response = client.chat.completions.create(
                model=Config.get_model_name(),
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            logger.debug(f"OpenAI API response: {response.choices[0].message.content}")
            return response.choices[0].message.content
        elif Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT == "claude":
            response = client.completions.create(
                model=Config.get_model_name(),
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1000
            )
            logger.debug(f"Claude API response: {response.completion}")
            return response.completion
    except Exception as e:
        logger.error(f"Error in {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT.value} API call: {str(e)}", exc_info=True)
        raise APICallError(Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT.value, str(e))


def make_api_call_missing(prompt: str) -> str:
    try:
        logger.info(f"Making API call to {Config.GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT}")
        if Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT == "groq":
            response = client.chat.completions.create(
                model=Config.MODEL_FINAL_MSG_SELECTED_MODEL_GROQ,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200
            )
            logger.debug(f"Groq API response: {response.choices[0].message.content}")
            return response.choices[0].message.content
        elif Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT == "openai":
            response = open_ai_client.chat.completions.create(
                model=Config.MODEL_FINAL_MSG_MODEL_OPENAI,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that extracts information and formats it as JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            logger.debug(f"OpenAI API response: {response.choices[0].message.content}")
            return response.choices[0].message.content
        elif Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT == "claude":
            response = client.completions.create(
                model=Config.MODEL_FINAL_MSG_SELECTED_MODEL_GROQ,
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1000
            )
            logger.debug(f"Claude API response: {response.completion}")
            return response.completion
    except Exception as e:
        logger.error(f"Error in {Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT} API call: {str(e)}", exc_info=True)
        raise APICallError(Config.MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT, str(e))

def make_image_api_call(prompt: str, image_data: str) -> str:
    try:
        logger.info("Making image API call to OpenAI")
        response = open_ai_client.chat.completions.create(
            model=Config.MODEL_IMAGE_OPENAI,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        logger.debug(f"OpenAI image API response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI image API call: {str(e)}", exc_info=True)
        raise APICallError(Config.MODEL_IMAGE_OPENAI, str(e))
        
        
def make_classification_call(prompt: str) -> str:
    """
    Makes an API call specifically for classification tasks using instructor for response parsing.
    
    Args:
        prompt (str): The formatted classification prompt
        
    Returns:
        str: JSON string containing the classification result
    """
    result = None
    try:

        logger.info("Making classification API call")
        
        # Define the Pydantic model for classification
        class ConversationClassification(BaseModel):
            classification: Literal["product_related", "general_conversation"]
            
        if Config.CLASSIFICATION_SELECTED_MODEL_CLIENT == "groq":
            # Use instructor-patched client for Groq
            patched_client = instructor.patch(groq_client)
            response = patched_client.chat.completions.create(
                model=Config.CLASSIFICATION_SELECTED_MODEL_GROQ,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_model=ConversationClassification
            )
            result = json.dumps({"classification": response.classification})
            
        elif Config.CLASSIFICATION_SELECTED_MODEL_CLIENT == "openai":
            # Use instructor-patched client for OpenAI
            patched_client = instructor.patch(open_ai_client)
            response = patched_client.chat.completions.create(
                model=Config.CLASSIFICATION_SELECTED_MODEL_OPENAI,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_model=ConversationClassification
            )
            result = json.dumps({"classification": response.classification})
            
        elif Config.CLASSIFICATION_SELECTED_MODEL_CLIENT == ModelType.CLAUDE:
            # For Claude, use regular API call since it doesn't support instructor
            response = None
            result = response  # Claude should return properly formatted JSON

        logger.debug(f"Classification API response: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in classification API call: {str(e)}", exc_info=True)
        raise APICallError(Config.SELECTED_MODEL.value, str(e))

# Initialize clients when this module is imported
# initialize_clients()