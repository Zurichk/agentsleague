"""
Clase base para todos los agentes de AEP CertMaster.

Define la interfaz común y funcionalidades compartidas
para todos los agentes especializados del sistema.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from src.config.settings import load_settings
from src.models.schemas import AEPWorkflowContext
from src.utils.logger import get_agent_logger
from .metrics import get_metrics_collector, calculate_quality_score


class AEPAgent(ABC):
    """
    Clase base abstracta para todos los agentes.

    Attributes:
        name: Nombre único del agente.
        description: Descripción de las responsabilidades del agente.
        capabilities: Lista de capacidades del agente.
        max_tokens: Máximo de tokens por llamada.
        temperature: Temperatura para generación de respuestas.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        """
        Inicializa el agente base.

        Args:
            name: Nombre único del agente.
            description: Descripción de responsabilidades.
            capabilities: Lista de capacidades.
            max_tokens: Máximo tokens por llamada.
            temperature: Temperatura para respuestas.
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Configuración
        self.settings = load_settings()
        self.logger = get_agent_logger(name)

        # Cliente Azure OpenAI
        self.client = AzureOpenAI(
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
        )

        self.logger.info(
            f"Agente {name} inicializado: {description}"
        )

    @abstractmethod
    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ejecuta la lógica principal del agente.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales específicos del agente.

        Returns:
            Diccionario con resultados de la ejecución.
        """
        pass

    async def execute_with_metrics(
        self,
        context: AEPWorkflowContext,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ejecuta el agente con tracking de métricas de calidad.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales específicos del agente.

        Returns:
            Diccionario con resultados de la ejecución.
        """
        start_time = time.time()

        try:
            result = await self.execute(context, **kwargs)
            response_time = time.time() - start_time

            # Calcular calidad de respuesta si hay una respuesta de texto
            quality_score = None
            if 'response' in result and isinstance(result['response'], str):
                quality_score = calculate_quality_score(
                    result['response'],
                    context={'expected_keywords': kwargs.get(
                        'expected_keywords', [])}
                )

            # Registrar métricas adicionales de calidad
            metrics_collector = get_metrics_collector()
            agent_metrics = metrics_collector.get_or_create_metrics(self.name)

            # Actualizar métricas de calidad si se calculó
            if quality_score is not None:
                agent_metrics.quality_scores.append(quality_score)
                if len(agent_metrics.quality_scores) > 1000:
                    agent_metrics.quality_scores = agent_metrics.quality_scores[-1000:]

            self.logger.info(
                f"Ejecución completada en {response_time:.2f}s con calidad {quality_score:.2f}"
            )

            return result

        except Exception as exc:
            response_time = time.time() - start_time
            self.logger.error(f"Error en ejecución: {exc}")

            # Registrar fallo en métricas
            metrics_collector = get_metrics_collector()
            metrics_collector.record_agent_call(
                agent_name=self.name,
                response_time=response_time,
                success=False
            )

            raise

    async def _call_azure_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Llama a Azure OpenAI para generar respuestas.

        Args:
            messages: Lista de mensajes para el chat.
            temperature: Temperatura (usa self.temperature si None).
            max_tokens: Máximo tokens (usa self.max_tokens si None).

        Returns:
            Respuesta generada por el modelo.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        start_time = time.time()
        success = False
        tokens_used = 0
        prompt_tokens = 0
        completion_tokens = 0

        try:
            self.logger.debug(
                f"Llamando a Azure OpenAI con {len(messages)} mensajes"
            )

            response = self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            content = response.choices[0].message.content
            if content:
                self.logger.debug(
                    f"Respuesta recibida: {len(content)} caracteres"
                )

                # Extraer información de tokens del uso
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens or 0
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0

                success = True
                return content
            else:
                raise ValueError("Respuesta vacía de Azure OpenAI")

        except Exception as exc:
            self.logger.error(f"Error llamando a Azure OpenAI: {exc}")
            raise
        finally:
            # Registrar métricas
            response_time = time.time() - start_time
            metrics_collector = get_metrics_collector()
            metrics_collector.record_agent_call(
                agent_name=self.name,
                response_time=response_time,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                success=success
            )

    def _build_system_prompt(self, base_prompt: str) -> str:
        """
        Construye el prompt del sistema con información contextual.

        Args:
            base_prompt: Prompt base específico del agente.

        Returns:
            Prompt completo del sistema.
        """
        return f"""Eres {self.name}, un agente especializado en AEP CertMaster.

Descripción: {self.description}

Capacidades: {', '.join(self.capabilities)}

Instrucciones generales:
- Sé preciso y útil en tus respuestas
- Usa toda tu capacidad analítica y todo el contexto disponible antes de responder
- Mantén un tono profesional pero amigable
- Si no tienes suficiente información, pide aclaraciones
- Registra tus razonamientos de manera clara
- Prioriza profundidad y calidad accionable sobre cantidad
- Evita respuestas genéricas y no inventes datos no soportados por el contexto

{base_prompt}
"""

    def log_reasoning(
        self,
        action: str,
        reasoning: str,
        result: str,
    ) -> None:
        """
        Registra el razonamiento del agente en el logger.

        Args:
            action: Acción realizada.
            reasoning: Razonamiento detrás de la acción.
            result: Resultado obtenido.
        """
        self.logger.info(
            f"[{action}] {reasoning} → {result}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas actuales del agente.

        Returns:
            Diccionario con métricas del agente.
        """
        metrics_collector = get_metrics_collector()
        agent_metrics = metrics_collector.get_agent_metrics(self.name)
        if agent_metrics:
            return agent_metrics.to_dict()
        return {
            'agent_name': self.name,
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'success_rate': 0.0,
            'total_tokens_used': 0,
            'average_response_time': 0.0,
            'average_quality_score': 0.0
        }
