"""
Configuración centralizada del proyecto AEP CertMaster.

Carga variables de entorno y define constantes globales
para todo el sistema multi-agente.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Rutas del proyecto ---
AEP_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AEP_SRC_DIR = AEP_PROJECT_ROOT / "src"
AEP_DATA_DIR = AEP_PROJECT_ROOT / "data"
AEP_TESTS_DIR = AEP_PROJECT_ROOT / "tests"

# --- Constantes globales ---
AEP_APP_NAME = "AEP CertMaster"
AEP_MAX_ATTEMPTS = 3
AEP_DEFAULT_LANGUAGE = "es"
AEP_MAX_TOKENS_PER_AGENT = 4096
AEP_ASSESSMENT_PASS_THRESHOLD = 0.7
AEP_MIN_QUESTIONS_PER_ASSESSMENT = 10
AEP_MAX_QUESTIONS_PER_ASSESSMENT = 30


class AEPSettings(BaseSettings):
    """
    Configuración del proyecto cargada desde variables de entorno.

    Attributes:
        azure_ai_project_connection_string: Connection string
            del proyecto en Azure AI Foundry.
        azure_ai_model_deployment: Nombre del deployment del
            modelo (por defecto gpt-4o).
        aep_log_level: Nivel de logging (DEBUG, INFO, WARNING,
            ERROR).
        aep_max_retries: Reintentos máximos en llamadas a la API.
        aep_default_language: Idioma por defecto del sistema.
    """

    model_config = SettingsConfigDict(
        env_file=str(AEP_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Azure AI Foundry ---
    azure_ai_project_connection_string: str = ""
    azure_ai_model_deployment: str = "gpt-4o"

    # --- Configuración opcional individual ---
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_ai_project_name: str = ""

    # --- Azure OpenAI (para compatibilidad directa) ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"

    # --- Proyecto ---
    aep_log_level: str = "INFO"
    aep_max_retries: int = 3
    aep_default_language: str = "es"

    # --- Email (Engagement Agent) ---
    aep_smtp_host: str = ""
    aep_smtp_port: int = 587
    aep_smtp_user: str = ""
    aep_smtp_password: str = ""

    # --- Búsqueda web ---
    # True  → escribe .log/.jsonl y archivos traces/*.json y muestra INFO en consola
    # False → sólo Application Insights (sin escritura local, sin ruido en consola)
    aep_trace_log_enabled: bool = False

    def is_azure_configured(self) -> bool:
        """Verifica si las credenciales de Azure están configuradas."""
        return bool(self.azure_ai_project_connection_string) or (
            bool(self.azure_subscription_id)
            and bool(self.azure_resource_group)
            and bool(self.azure_ai_project_name)
        )

    def is_email_configured(self) -> bool:
        """Verifica si el servicio de email está configurado."""
        return bool(self.aep_smtp_host) and bool(
            self.aep_smtp_user
        )


def load_settings() -> AEPSettings:
    """
    Carga y retorna la configuración del proyecto.

    Returns:
        Instancia de AEPSettings con valores cargados
        desde .env y variables de entorno.
    """
    return AEPSettings()
