#!/usr/bin/env python3
"""
API client for st-api.geostyx.com image classifier endpoint.

This client provides a simple interface for sending PIL Images to the
Shop Titans text classification API and receiving structured responses.
"""

import io
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from PIL import Image


@dataclass
class ClassificationResult:
    """Result from image classification API"""
    extracted_text: str
    confidence: float
    num_characters: int
    processing_time_ms: float
    timestamp: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create ClassificationResult from API response dictionary"""
        return cls(
            extracted_text=data['extracted_text'],
            confidence=data['confidence'],
            num_characters=data['num_characters'],
            processing_time_ms=data['processing_time_ms'],
            timestamp=data['timestamp']
        )


@dataclass
class HealthStatus:
    """Health status from API"""
    status: str
    timestamp: str
    version: str
    model_loaded: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthStatus':
        """Create HealthStatus from API response dictionary"""
        return cls(
            status=data['status'],
            timestamp=data['timestamp'],
            version=data['version'],
            model_loaded=data['model_loaded']
        )


class STAPIError(Exception):
    """Custom exception for ST API errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class STAPIClient:
    """
    Client for interacting with the Shop Titans text classification API.

    Example usage:
        from PIL import Image

        client = STAPIClient()
        image = Image.open("path/to/image.png")
        result = client.classify_image(image)
        print(f"Extracted text: {result.extracted_text}")
        print(f"Confidence: {result.confidence}")
    """

    def __init__(self, base_url: str = "https://st-api.geostyx.com", timeout: int = 30):
        """
        Initialize the ST API client.

        Args:
            base_url: Base URL for the API (default: https://st-api.geostyx.com)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'User-Agent': 'ST-API-Client/1.0.0'
        })

    def classify_image(self, image: Image.Image, image_format: str = "PNG") -> ClassificationResult:
        """
        Classify text/digits in a PIL Image.

        Args:
            image: PIL Image object to classify
            image_format: Format to use when sending image (PNG, JPEG, etc.)

        Returns:
            ClassificationResult with extracted text and metadata

        Raises:
            STAPIError: If the API request fails or returns an error
        """
        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_format)
        image_bytes.seek(0)

        # Determine content type
        content_type = f"image/{image_format.lower()}"
        if image_format.upper() == "JPEG":
            content_type = "image/jpeg"

        # Prepare multipart form data
        files = {
            'file': ('image.' + image_format.lower(), image_bytes, content_type)
        }

        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                files=files,
                timeout=self.timeout
            )

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        error_msg += f": {error_data['detail']}"
                except:
                    error_msg += f": {response.text}"

                raise STAPIError(error_msg, response.status_code,
                                 response.json() if response.content else None)

            # Parse response
            try:
                data = response.json()
                return ClassificationResult.from_dict(data)
            except (ValueError, KeyError) as e:
                raise STAPIError(
                    f"Failed to parse API response: {e}", response.status_code, None)

        except requests.exceptions.Timeout:
            raise STAPIError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise STAPIError(f"Failed to connect to {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise STAPIError(f"Request failed: {e}")

    def health_check(self) -> HealthStatus:
        """
        Check the health status of the API.

        Returns:
            HealthStatus with API health information

        Raises:
            STAPIError: If the health check fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise STAPIError(
                    f"Health check failed with status {response.status_code}", response.status_code)

            data = response.json()
            return HealthStatus.from_dict(data)

        except requests.exceptions.RequestException as e:
            raise STAPIError(f"Health check request failed: {e}")

    def is_healthy(self) -> bool:
        """
        Quick check if the API is healthy and model is loaded.

        Returns:
            True if API is healthy and model is loaded, False otherwise
        """
        try:
            health = self.health_check()
            return health.status == "healthy" and health.model_loaded
        except STAPIError:
            return False

    def close(self):
        """Close the underlying session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function for quick usage
def classify_image(image: Image.Image, base_url: str = "https://st-api.geostyx.com") -> ClassificationResult:
    """
    Convenience function to classify an image without creating a client instance.

    Args:
        image: PIL Image object to classify
        base_url: Base URL for the API

    Returns:
        ClassificationResult with extracted text and metadata
    """
    with STAPIClient(base_url=base_url) as client:
        return client.classify_image(image)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python client.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Load image
        image = Image.open(image_path)
        print(f"Loaded image: {image_path} ({image.size})")

        # Classify image
        print("Classifying image...")
        result = classify_image(image)

        # Print results
        print(f"\nResults:")
        print(f"  Extracted text: '{result.extracted_text}'")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Characters: {result.num_characters}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")
        print(f"  Timestamp: {result.timestamp}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
