"""
Script de verificación de conexión con Azure AI Foundry.

Ejecutar para comprobar que las credenciales están
correctamente configuradas antes de iniciar el desarrollo.

Uso:
    python -m src.config.verify_setup
"""

from __future__ import annotations

import sys

from src.config.settings import load_settings
from src.utils.logger import setup_logger

logger = setup_logger("aep.verify")


def verify_environment() -> bool:
    """
    Verifica que el entorno esté correctamente configurado.

    Returns:
        True si todas las verificaciones pasan.
    """
    all_ok = True

    logger.info("=" * 60)
    logger.info("AEP CertMaster — Verificación de Entorno")
    logger.info("=" * 60)

    # 1. Cargar configuración
    try:
        settings = load_settings()
        logger.info("[OK] Configuración cargada desde .env")
    except Exception as exc:
        logger.error(f"[FAIL] Error cargando .env: {exc}")
        return False

    # 2. Verificar credenciales Azure
    if settings.is_azure_configured():
        logger.info("[OK] Credenciales de Azure configuradas")
    else:
        logger.warning(
            "[WARN] Credenciales de Azure NO configuradas. "
            "Configura AZURE_AI_PROJECT_CONNECTION_STRING "
            "en tu archivo .env"
        )
        all_ok = False

    # 3. Verificar modelo
    logger.info(
        f"[INFO] Modelo configurado: "
        f"{settings.azure_ai_model_deployment}"
    )

    # 4. Verificar email (opcional)
    if settings.is_email_configured():
        logger.info("[OK] Servicio de email configurado")
    else:
        logger.info(
            "[INFO] Email no configurado (opcional)"
        )

    # 5. Intentar conexión con Azure
    if settings.is_azure_configured():
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            logger.info(
                "[OK] Azure Identity SDK disponible"
            )
        except ImportError:
            logger.warning(
                "[WARN] azure-identity no instalado. "
                "Ejecuta: pip install -r requirements.txt"
            )
            all_ok = False
        except Exception as exc:
            logger.warning(
                f"[WARN] No se pudo verificar credencial: "
                f"{exc}"
            )

    logger.info("=" * 60)
    if all_ok:
        logger.info(
            "✓ Entorno listo. Puedes comenzar a desarrollar."
        )
    else:
        logger.warning(
            "⚠ Hay elementos pendientes de configurar. "
            "Revisa los warnings anteriores."
        )
    logger.info("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
