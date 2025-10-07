from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AIEngine(ABC):
    """
    Abstract base class for AI engines.
    """

    @abstractmethod
    def generate_section(self, section_name: str, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a specific report section.
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the AI engine.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        """
        pass

    @abstractmethod
    def call_ai(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        A generic method to call the AI model with a prompt.
        """
        pass