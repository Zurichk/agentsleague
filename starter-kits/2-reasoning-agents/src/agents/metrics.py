"""
Módulo de métricas para agentes AEP CertMaster.

Proporciona funcionalidades para trackear y medir el rendimiento
de los agentes en términos de latencia, uso de tokens, calidad, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import statistics


@dataclass
class AgentMetrics:
    """Métricas de rendimiento para un agente."""

    agent_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_times: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    last_call_timestamp: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def average_response_time(self) -> float:
        """Calcula el tiempo promedio de respuesta."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def median_response_time(self) -> float:
        """Calcula el tiempo mediano de respuesta."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def average_quality_score(self) -> float:
        """Calcula el puntaje promedio de calidad."""
        if not self.quality_scores:
            return 0.0
        return statistics.mean(self.quality_scores)

    @property
    def tokens_per_second(self) -> float:
        """Calcula tokens por segundo promedio."""
        if not self.response_times or self.total_tokens_used == 0:
            return 0.0
        total_time = sum(self.response_times)
        if total_time == 0:
            return 0.0
        return self.total_tokens_used / total_time

    def record_call(
        self,
        response_time: float,
        tokens_used: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        quality_score: Optional[float] = None
    ) -> None:
        """Registra una llamada al agente."""
        self.total_calls += 1
        self.last_call_timestamp = datetime.now()

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        self.total_tokens_used += tokens_used
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        if response_time > 0:
            self.response_times.append(response_time)

        if quality_score is not None and 0 <= quality_score <= 1:
            self.quality_scores.append(quality_score)

        # Mantener solo las últimas 1000 mediciones para no crecer indefinidamente
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        if len(self.quality_scores) > 1000:
            self.quality_scores = self.quality_scores[-1000:]

    def to_dict(self) -> Dict:
        """Convierte las métricas a diccionario."""
        return {
            'agent_name': self.agent_name,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': round(self.success_rate, 3),
            'total_tokens_used': self.total_tokens_used,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'average_response_time': round(self.average_response_time, 3),
            'median_response_time': round(self.median_response_time, 3),
            'average_quality_score': round(self.average_quality_score, 3),
            'tokens_per_second': round(self.tokens_per_second, 2),
            'last_call_timestamp': self.last_call_timestamp.isoformat() if self.last_call_timestamp else None
        }


class AgentMetricsCollector:
    """Colector centralizado de métricas de agentes."""

    def __init__(self):
        self.metrics: Dict[str, AgentMetrics] = {}

    def get_or_create_metrics(self, agent_name: str) -> AgentMetrics:
        """Obtiene o crea métricas para un agente."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        return self.metrics[agent_name]

    def record_agent_call(
        self,
        agent_name: str,
        response_time: float,
        tokens_used: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        quality_score: Optional[float] = None
    ) -> None:
        """Registra una llamada de agente."""
        metrics = self.get_or_create_metrics(agent_name)
        metrics.record_call(
            response_time=response_time,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            quality_score=quality_score
        )

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Obtiene métricas de un agente específico."""
        return self.metrics.get(agent_name)

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Obtiene todas las métricas como diccionarios."""
        return {name: metrics.to_dict() for name, metrics in self.metrics.items()}

    def get_summary_stats(self) -> Dict:
        """Obtiene estadísticas resumidas de todos los agentes."""
        if not self.metrics:
            return {}

        total_calls = sum(m.total_calls for m in self.metrics.values())
        total_successful = sum(
            m.successful_calls for m in self.metrics.values())
        total_tokens = sum(m.total_tokens_used for m in self.metrics.values())
        avg_response_times = [
            m.average_response_time for m in self.metrics.values() if m.response_times]

        return {
            'total_agents': len(self.metrics),
            'total_calls': total_calls,
            'overall_success_rate': round(total_successful / total_calls, 3) if total_calls > 0 else 0,
            'total_tokens_used': total_tokens,
            'average_response_time': round(statistics.mean(avg_response_times), 3) if avg_response_times else 0,
            'agents': list(self.metrics.keys())
        }


# Instancia global del colector de métricas
metrics_collector = AgentMetricsCollector()


def get_metrics_collector() -> AgentMetricsCollector:
    """Obtiene la instancia global del colector de métricas."""
    return metrics_collector


def calculate_quality_score(response: str, context: Dict) -> float:
    """
    Calcula un puntaje de calidad básico para una respuesta de agente.

    Args:
        response: La respuesta del agente
        context: Contexto adicional para evaluación

    Returns:
        Puntaje de calidad entre 0 y 1
    """
    if not response or not response.strip():
        return 0.0

    score = 0.5  # Puntaje base

    # Longitud apropiada (no demasiado corta, no demasiado larga)
    length = len(response)
    if 50 <= length <= 2000:
        score += 0.2
    elif length < 50:
        score -= 0.1
    else:
        score -= 0.05

    # Contiene información relevante
    relevant_keywords = context.get('expected_keywords', [])
    if relevant_keywords:
        found_keywords = sum(
            1 for keyword in relevant_keywords if keyword.lower() in response.lower())
        keyword_score = min(found_keywords / len(relevant_keywords), 1.0)
        score += keyword_score * 0.2

    # Estructura y claridad
    if any(indicator in response for indicator in ['•', '-', '1.', '2.', '3.']):
        score += 0.1

    return max(0.0, min(1.0, score))
