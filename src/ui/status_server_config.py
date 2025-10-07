from dataclasses import dataclass
import os

@dataclass
class FeatureConfig:
    advanced_features: bool = True
    ollama_integration: bool = True
    openai_integration: bool = False
    remote_llm_integration: bool = False
    cache_ttl_seconds: int = 300  # 5 minutes
    max_ticker_length: int = 12

@dataclass
class StatusServerConfig:
    project_root: str
    features: FeatureConfig
    max_queue_size: int = 100

    @classmethod
    def from_environment(cls):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return cls(
            project_root=project_root,
            features=FeatureConfig(
                advanced_features=os.getenv("ADVANCED_FEATURES", "true").lower() == "true",
                ollama_integration=os.getenv("OLLAMA_INTEGRATION", "true").lower() == "true",
                openai_integration=os.getenv("OPENAI_INTEGRATION", "false").lower() == "true",
                remote_llm_integration=os.getenv("REMOTE_LLM_INTEGRATION", "false").lower() == "true",
                cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
                max_ticker_length=int(os.getenv("MAX_TICKER_LENGTH", "12")),
            ),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "100"))
        )

    def validate(self) -> bool:
        # Count enabled integrations
        enabled_integrations = sum([
            self.features.ollama_integration,
            self.features.openai_integration,
            self.features.remote_llm_integration
        ])

        if enabled_integrations > 1:
            print("ERROR: Cannot enable multiple LLM integrations simultaneously.")
            return False
        return True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'project_root': self.project_root,
            'max_queue_size': self.max_queue_size,
            'features': {
                'advanced_features': self.features.advanced_features,
                'ollama_integration': self.features.ollama_integration,
                'openai_integration': self.features.openai_integration,
                'remote_llm_integration': self.features.remote_llm_integration,
                'cache_ttl_seconds': self.features.cache_ttl_seconds,
                'max_ticker_length': self.features.max_ticker_length,
            }
        }
