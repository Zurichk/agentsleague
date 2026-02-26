"""
Herramienta Azure OpenAI - AEP CertMaster

Esta herramienta proporciona integración con Azure OpenAI para generación de texto
y análisis de contenido.
"""

import logging
from typing import Any, Dict, Optional
from openai import AsyncAzureOpenAI


class AzureOpenAITool:
    """
    Herramienta para integración con Azure OpenAI.

    Proporciona métodos para generar completions y análisis de texto
    utilizando modelos de Azure OpenAI.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa la herramienta Azure OpenAI.

        Args:
            config: Configuración de Azure OpenAI.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuración de Azure OpenAI
        self.endpoint = config.get("endpoint", "")
        self.api_key = config.get("api_key", "")
        self.deployment_name = config.get("deployment_name", "")
        self.api_version = config.get("api_version", "2024-02-15-preview")

        # Cliente Azure OpenAI
        self.client = None
        if self.endpoint and self.api_key and self.deployment_name:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            self.logger.info("Cliente Azure OpenAI inicializado")
        else:
            raise ValueError(
                "Configuración incompleta de Azure OpenAI. "
                "Se requieren endpoint, api_key y deployment_name."
            )

    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Genera una completion usando Azure OpenAI.

        Args:
            prompt: Texto del prompt.
            max_tokens: Máximo número de tokens.
            temperature: Temperatura para la generación.
            **kwargs: Parámetros adicionales.

        Returns:
            Texto generado.
        """
        if not self.client:
            raise RuntimeError("Cliente Azure OpenAI no inicializado")

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error en Azure OpenAI: {str(e)}")
            raise

    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analiza texto usando Azure OpenAI.

        Args:
            text: Texto a analizar.
            analysis_type: Tipo de análisis.
            **kwargs: Parámetros adicionales.

        Returns:
            Resultado del análisis.
        """
        if not self.client:
            raise RuntimeError("Cliente Azure OpenAI no inicializado")

        try:
            prompt = f"Analiza el siguiente texto como {analysis_type}:\n\n{text}"

            response = await self.generate_completion(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )

            return {
                "analysis_type": analysis_type,
                "confidence": 0.9,
                "summary": response,
                "key_points": response.split("\n")[:3] if "\n" in response else [response]
            }

        except Exception as e:
            self.logger.error(f"Error analizando texto: {str(e)}")
            return {
                "analysis_type": analysis_type,
                "error": str(e),
                "confidence": 0.0
            }
