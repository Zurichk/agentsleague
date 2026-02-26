"""
Script de prueba para verificar conexión con Azure AI Foundry.

Hace una llamada simple a Azure OpenAI para confirmar que
las credenciales y configuración funcionan correctamente.

Uso:
    python -m src.config.test_azure_connection
"""

from __future__ import annotations

import sys

from openai import AzureOpenAI

from src.config.settings import load_settings
from src.utils.logger import setup_logger

logger = setup_logger("aep.test_azure")


def test_azure_connection() -> bool:
    """
    Prueba la conexión con Azure AI Foundry.

    Returns:
        True si la conexión funciona correctamente.
    """
    logger.info("=" * 60)
    logger.info("AEP CertMaster — Prueba de Conexión Azure AI")
    logger.info("=" * 60)

    try:
        settings = load_settings()

        # Usar Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

        # Hacer una llamada simple
        logger.info("Haciendo llamada de prueba a Azure OpenAI...")
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hola, ¿puedes confirmar que esta conexión funciona? Responde con una sola palabra: 'Sí'",
                }
            ],
            max_tokens=10,
            temperature=0.1,
        )

        # Verificar respuesta
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            logger.info(f"Respuesta recibida: '{content}'")
            if "Sí" in content or "si" in content.lower():
                logger.info("[OK] Conexión con Azure AI Foundry exitosa!")
                return True
            else:
                logger.warning(f"[WARN] Respuesta inesperada: {content}")
                return False
        else:
            logger.error("[FAIL] No se recibió respuesta válida")
            return False

    except Exception as exc:
        logger.error(f"[FAIL] Error conectando con Azure: {exc}")
        return False


if __name__ == "__main__":
    success = test_azure_connection()
    if success:
        logger.info("✅ ¡Listo para desarrollar agentes!")
    else:
        logger.error("❌ Revisa tu configuración de Azure")
    sys.exit(0 if success else 1)
