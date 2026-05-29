"""vision_engine.py

Analyze a local image with the Google Gemini 2.5 Flash model using the modern
google-genai SDK and Pydantic schema validation for structured JSON output.
"""

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from google import genai as google_genai
from database_manager import insert_memory


class ImageAnalysisSchema(BaseModel):
    """Structured schema for image analysis output."""
    
    item_detected: str = Field(
        description="Name of the object or item detected in the image"
    )
    spatial_location: str = Field(
        description="Description of where the item is located relative to its surroundings"
    )
    confidence_score: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the detection: high, medium, or low"
    )


def _configure_genai() -> google_genai.Client:
    """Configure and return the GenAI client with GEMINI_API_KEY from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment. "
            "Please set it before running this script."
        )
    return google_genai.Client(api_key=api_key)


def analyze_image_strict_json(image_path: str) -> ImageAnalysisSchema:
    """
    Analyze a local image and return structured data matching the schema.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        ImageAnalysisSchema: Structured data with item_detected, spatial_location, and confidence_score
        
    Raises:
        FileNotFoundError: If the image does not exist
        ValueError: If the model output fails schema validation
    """
    client = _configure_genai()
    
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image file
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    prompt = (
        "You are an image analysis assistant. Analyze the attached image and provide "
        "a detailed assessment of the main object or item in the image, its location "
        "relative to the surroundings, and your confidence in the detection."
    )
    
    # Use the modern gemini-2.5-flash model with strict schema enforcement
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=google_genai.types.Content(
            parts=[
                google_genai.types.Part(text=prompt),
                google_genai.types.Part(
                    inline_data=google_genai.types.Blob(
                        mime_type="image/png",
                        data=image_bytes,
                    )
                ),
            ]
        ),
        config=google_genai.types.GenerateContentConfig(
            response_schema=ImageAnalysisSchema,
            response_mime_type="application/json",
            temperature=1,  # Required for schema-constrained generation
        ),
    )
    
    # Parse the response text as JSON and validate against schema
    try:
        response_data = json.loads(response.text)
        result = ImageAnalysisSchema(**response_data)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {response.text}") from e
    except ValueError as e:
        raise ValueError(f"Model response failed schema validation: {e}") from e


def analyze_test_image() -> ImageAnalysisSchema:
    """Analyze root-level test.png for verification."""
    root_dir = Path(__file__).parent.parent
    test_image = root_dir / "test.png"
    return analyze_image_strict_json(str(test_image))


if __name__ == "__main__":
    try:
        # 1. Run the vision engine on our test image
        result = analyze_test_image()
        print("✅ Analysis successful!")
        print(json.dumps(result.model_dump(), indent=2))
        
        # 2. Automatically save the live result directly into our SQLite memory
        insert_memory(
            item=result.item_detected,
            location=result.spatial_location,
            confidence=result.confidence_score
        )
        print("💾 Live analysis successfully logged to SQLite spatial memory!")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("   Make sure test.png exists in the repository root.")
    except ValueError as e:
        print(f"❌ Schema Validation Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
