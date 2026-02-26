#!/usr/bin/env python3
"""
Script para listar agentes disponibles en Azure AI Foundry
Ejecuta este script para obtener los IDs de los agentes desplegados.
"""

import os
import logging
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_foundry_agents():
    """Listar agentes disponibles en Azure AI Foundry."""
    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
        if not connection_string:
            logger.error("AZURE_AI_PROJECT_CONNECTION_STRING no configurada")
            return

        # Extraer endpoint del connection string
        # Formato: https://<project>.services.ai.azure.com/api/projects/<project-name>
        import re
        match = re.match(r"https://([^.]+)\.services\.ai\.azure\.com.*", connection_string)
        if not match:
            logger.error("Formato de AZURE_AI_PROJECT_CONNECTION_STRING inválido")
            return

        project_name = match.group(1)
        endpoint = f"https://{project_name}.services.ai.azure.com"

        client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential()
        )

        logger.info("🔍 Buscando agentes en Azure AI Foundry...")

        # Listar agentes
        agents = client.agents.list_assistants()
        agent_list = list(agents)

        if not agent_list:
            logger.warning("No se encontraron agentes en el proyecto")
            return

        print("\n🤖 Agentes disponibles en Azure AI Foundry:")
        print("=" * 60)

        for agent in agent_list:
            print(f"📋 Nombre: {agent.name}")
            print(f"🆔 ID: {agent.id}")
            print(f"📝 Descripción: {agent.description or 'Sin descripción'}")
            print(f"📅 Creado: {agent.created_at}")
            print("-" * 40)

        print(f"\n✅ Total de agentes encontrados: {len(agent_list)}")
        print("\n💡 Copia los IDs correspondientes a tus variables de entorno en .env:")
        print("   AEP_AGENT_CURATOR_ID=...")
        print("   AEP_AGENT_STUDY_PLAN_ID=...")
        print("   etc.")

    except ImportError:
        logger.error("Azure AI Projects SDK no instalado. Ejecuta: pip install azure-ai-projects azure-identity")
    except Exception as e:
        logger.error(f"Error listando agentes: {e}")
        print("\n🔧 Verifica:")
        print("   - AZURE_AI_PROJECT_CONNECTION_STRING está configurada correctamente")
        print("   - Tienes permisos para acceder al proyecto de AI Foundry")
        print("   - Los agentes están desplegados en el proyecto")

if __name__ == "__main__":
    list_foundry_agents()