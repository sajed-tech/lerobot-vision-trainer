"""
Object Detection Module for Robot Vision Trainer
Uses OpenAI or OpenRouter API for vision models to detect objects and actions in images.
"""

import os
import json
import base64
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import re

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class to handle object detection using vision language models.
    """
    
    def __init__(self, 
                api_key: str,
                model_name: str = "gemini-2.5-pro-exp-03-25:free",
                use_openrouter: bool = True,
                default_prompt: Optional[str] = None):
        """
        Initialize the object detector.
        
        Args:
            api_key: API key for OpenAI or OpenRouter
            model_name: Name of the vision model to use
            use_openrouter: Whether to use OpenRouter (True) or OpenAI (False)
            default_prompt: Default prompt for object detection
        """
        self.api_key = api_key
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        
        # Set default prompt if not provided
        if default_prompt is None:
            self.default_prompt = (
                "Analyze this image and identify objects that a robot could interact with. "
                "List every visible object and suggest possible actions for each object. "
                "Format your response in JSON with objects as keys and actions as values. "
                "Example: {\"cup\": [\"pick up\", \"pour liquid\"], \"door\": [\"open\", \"close\"]}"
            )
        else:
            self.default_prompt = default_prompt
            
        logger.info(f"ObjectDetector initialized with model: {model_name}")
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode image as base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (width, height)
        """
        with Image.open(image_path) as img:
            return img.size
            
    def detect_objects(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect objects in an image using a vision model.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt to use for detection (optional)
            
        Returns:
            Dictionary with detected objects and actions
        """
        # Use default prompt if none provided
        if prompt is None:
            prompt = self.default_prompt
            
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"Image file not found: {image_path}"}
            
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            if self.use_openrouter:
                return self._detect_with_openrouter(base64_image, prompt)
            else:
                return self._detect_with_openai(base64_image, prompt)
                
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return {"error": str(e)}
            
    def _detect_with_openrouter(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects using OpenRouter API.
        
        Args:
            base64_image: Base64 encoded image
            prompt: Prompt for the vision model
            
        Returns:
            Dictionary with detected objects and actions
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        logger.info(f"Sending request to OpenRouter with model: {self.model_name}")
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # Add timeout to prevent hanging requests
            )
            
            if response.status_code != 200:
                logger.error(f"Error from OpenRouter API: {response.status_code} - {response.text}")
                return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}
                
            result = response.json()
            logger.info("Successfully received response from OpenRouter")
            
            # Dump full response for debugging
            logger.debug(f"Full API response: {json.dumps(result, indent=2)}")
            
            # Extract text from API response
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    text_response = result["choices"][0]["message"]["content"]
                else:
                    # Fallback for different response formats
                    text_response = str(result["choices"][0])
            else:
                # If 'choices' is missing, try to extract from the full response
                logger.warning("Response doesn't contain expected 'choices' field, using full response text")
                text_response = str(result)
                
            # Log the extracted text
            logger.info(f"Extracted text response: {text_response[:100]}...")
            
            # Extract JSON from response
            objects_actions = self._extract_json_from_text(text_response)
            return self._format_detection_result(objects_actions)
                
        except Exception as e:
            logger.error(f"Error in OpenRouter API call: {str(e)}")
            return {"success": False, "error": f"API call failed: {str(e)}"}
            
    def _detect_with_openai(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects using OpenAI API.
        
        Args:
            base64_image: Base64 encoded image
            prompt: Prompt for the vision model
            
        Returns:
            Dictionary with detected objects and actions
        """
        import openai
        
        openai.api_key = self.api_key
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        logger.info(f"Sending request to OpenAI with model: {self.model_name}")
        try:
            response = openai.chat.completions.create(**payload)
            logger.info("Successfully received response from OpenAI")
            
            text_response = response.choices[0].message.content
            # Extract JSON from response
            objects_actions = self._extract_json_from_text(text_response)
            return self._format_detection_result(objects_actions)
            
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return {"error": f"API Error: {e}"}
            
    def _extract_json_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text response from the API
            
        Returns:
            Dictionary of objects and actions
        """
        logger.info("Attempting to extract JSON from API response")
        
        if not text:
            logger.error("Empty text response from API")
            raise ValueError("Empty response from API")
            
        try:
            # First attempt: Try to parse the entire text as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.info("Full text is not valid JSON, trying to extract JSON portion")
            
            # Second attempt: Find JSON within curly braces
            json_start = text.find('{')
            json_end = text.rfind('}')
            
            if json_start >= 0 and json_end >= 0 and json_end > json_start:
                json_str = text[json_start:json_end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    logger.info(f"Found text between braces is not valid JSON: {json_str[:100]}...")
            
            # Third attempt: Use regex to find all JSON-like structures
            import re
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            try:
                import regex  # More powerful regex library that supports recursion
                json_matches = regex.findall(r'\{(?:[^{}]|(?R))*\}', text)
                
                for json_candidate in json_matches:
                    try:
                        obj = json.loads(json_candidate)
                        if isinstance(obj, dict) and obj:  # Non-empty dict
                            return obj
                    except json.JSONDecodeError:
                        continue
            except ImportError:
                logger.info("regex library not available, falling back to manual parsing")
            
            # Fourth attempt: Manual parsing
            logger.info("Attempting manual parsing of text into object-action format")
            objects_actions = {}
            lines = text.split('\n')
            current_object = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line contains an object name (with colon)
                if ':' in line and not line.startswith('-') and not line.startswith('*'):
                    parts = line.split(':', 1)
                    object_name = parts[0].strip().strip('"\'').lower()
                    # Clean up object name
                    object_name = re.sub(r'[^a-z0-9 ]+', '', object_name)
                    if object_name:
                        current_object = object_name
                        objects_actions[current_object] = []
                        
                        # Check if actions are on the same line
                        if len(parts) > 1 and parts[1].strip():
                            actions_text = parts[1].strip()
                            # Try to extract actions list
                            if '[' in actions_text and ']' in actions_text:
                                try:
                                    actions_json = actions_text[actions_text.find('['):actions_text.rfind(']')+1]
                                    actions = json.loads(actions_json)
                                    objects_actions[current_object].extend([a.lower() for a in actions if a])
                                except json.JSONDecodeError:
                                    pass
                            
                            # If JSON parsing failed, try comma-separated list
                            if not objects_actions[current_object]:
                                actions = [a.strip().strip('"-,\'[]').lower() for a in actions_text.split(',')]
                                objects_actions[current_object].extend([a for a in actions if a])
                
                # Check if line looks like an action item (with bullet or dash)
                elif (line.startswith('-') or line.startswith('*')) and current_object:
                    action = line.strip('- *"\'').lower()
                    if action:
                        objects_actions[current_object].append(action)
            
            # If we found any objects and actions, return them
            if objects_actions:
                logger.info(f"Successfully parsed {len(objects_actions)} objects with manual parsing")
                return objects_actions
                
            # If all else fails, create a simple dictionary from any text content
            if text.strip():
                logger.warning("Could not parse structured data, creating minimal object-action representation")
                # Just use the text as a single object with a generic action
                return {"detected_item": ["interact with"]}
                
            # If even that fails, raise an error
            raise ValueError("Could not extract object-action data from the response")
            
        except Exception as e:
            logger.error(f"Error extracting JSON from text: {e}")
            logger.error(f"Text that failed parsing: {text[:500]}")
            # Return a minimal valid structure
            return {"error_processing": ["review content"]}
    
    def _format_detection_result(self, objects_actions: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Format the detection result.
        
        Args:
            objects_actions: Dictionary of objects and their actions
            
        Returns:
            Formatted detection result
        """
        objects = list(objects_actions.keys())
        
        # Format actions as a dictionary where keys are objects
        actions = {obj: actions for obj, actions in objects_actions.items()}
        
        return {
            "success": True,
            "objects": objects,
            "actions": actions,
            "objects_actions": objects_actions
        } 