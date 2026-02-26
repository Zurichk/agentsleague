"""
Sistema de Rúbricas de Evaluación Automatizada - AEP CertMaster

Este módulo implementa rúbricas automatizadas para evaluar la calidad
de respuestas de agentes usando criterios objetivos y subjetivos.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field


class QualityDimension(Enum):
    """
    Dimensiones de calidad para evaluación.
    """
    ACCURACY = "accuracy"           # Precisión factual
    COMPLETENESS = "completeness"   # Completitud de respuesta
    RELEVANCE = "relevance"         # Relevancia al contexto
    CLARITY = "clarity"             # Claridad y comprensión
    STRUCTURE = "structure"         # Estructura y organización
    CREATIVITY = "creativity"        # Creatividad y originalidad


class QualityLevel(Enum):
    """
    Niveles de calidad.
    """
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 75-89%
    FAIR = "fair"           # 60-74%
    POOR = "poor"           # 40-59%
    VERY_POOR = "very_poor"  # 0-39%


class EvaluationCriterion(BaseModel):
    """
    Criterio de evaluación individual.

    Attributes:
        name: Nombre del criterio.
        description: Descripción del criterio.
        dimension: Dimensión de calidad que evalúa.
        weight: Peso relativo (0-1).
        scoring_guide: Guía de puntuación por niveles.
    """
    name: str
    description: str
    dimension: QualityDimension
    weight: float = Field(ge=0.0, le=1.0)
    scoring_guide: Dict[QualityLevel, str]


class AutomatedRubric(BaseModel):
    """
    Rúbrica automatizada completa.

    Attributes:
        name: Nombre de la rúbrica.
        description: Descripción de la rúbrica.
        criteria: Lista de criterios de evaluación.
        agent_type: Tipo de agente al que aplica.
        version: Versión de la rúbrica.
    """
    name: str
    description: str
    criteria: List[EvaluationCriterion]
    agent_type: str
    version: str = "1.0"


class EvaluationResult(BaseModel):
    """
    Resultado de evaluación de una respuesta.

    Attributes:
        overall_score: Puntuación general (0-100).
        overall_level: Nivel de calidad general.
        criteria_scores: Puntuaciones por criterio.
        feedback: Feedback generado automáticamente.
        strengths: Puntos fuertes identificados.
        weaknesses: Áreas de mejora identificadas.
        recommendations: Recomendaciones específicas.
    """
    overall_score: float
    overall_level: QualityLevel
    criteria_scores: Dict[str, Dict[str, Any]]
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class AutomatedRubricsEvaluator:
    """
    Evaluador automático de respuestas usando rúbricas.

    Esta clase implementa evaluación automática de calidad de respuestas
    de agentes usando criterios objetivos y análisis de patrones.
    """

    def __init__(self) -> None:
        """Inicializa el evaluador de rúbricas."""
        self.rubrics: Dict[str, AutomatedRubric] = {}
        self._initialize_default_rubrics()

    def _initialize_default_rubrics(self) -> None:
        """
        Inicializa las rúbricas por defecto para cada tipo de agente.
        """
        # Rúbrica para CuratorAgent
        curator_criteria = [
            EvaluationCriterion(
                name="relevance_to_certification",
                description="Relevancia de las rutas sugeridas para la certificación objetivo",
                dimension=QualityDimension.RELEVANCE,
                weight=0.3,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Todas las rutas son altamente relevantes y alineadas con requisitos de certificación",
                    QualityLevel.GOOD: "La mayoría de rutas son relevantes, algunas podrían ser más específicas",
                    QualityLevel.FAIR: "Algunas rutas relevantes, pero hay sugerencias genéricas o irrelevantes",
                    QualityLevel.POOR: "Pocas rutas relevantes, enfoque incorrecto en certificación",
                    QualityLevel.VERY_POOR: "Rutas completamente irrelevantes o incorrectas",
                }
            ),
            EvaluationCriterion(
                name="path_diversity",
                description="Diversidad y cobertura de diferentes aspectos del aprendizaje",
                dimension=QualityDimension.COMPLETENESS,
                weight=0.25,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Excelente cobertura de fundamentos, intermedio y avanzado",
                    QualityLevel.GOOD: "Buena cobertura de niveles básicos e intermedios",
                    QualityLevel.FAIR: "Cobertura básica aceptable pero limitada",
                    QualityLevel.POOR: "Cobertura muy limitada, falta profundidad",
                    QualityLevel.VERY_POOR: "Cobertura insuficiente, enfoque muy estrecho",
                }
            ),
            EvaluationCriterion(
                name="difficulty_progression",
                description="Progresión lógica de dificultad en las rutas sugeridas",
                dimension=QualityDimension.STRUCTURE,
                weight=0.2,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Progresión perfecta de principiante a avanzado",
                    QualityLevel.GOOD: "Buena progresión con algunos saltos lógicos",
                    QualityLevel.FAIR: "Progresión aceptable pero con algunas inconsistencias",
                    QualityLevel.POOR: "Progresión pobre, saltos grandes en dificultad",
                    QualityLevel.VERY_POOR: "Sin progresión lógica, rutas desordenadas",
                }
            ),
            EvaluationCriterion(
                name="recommendation_quality",
                description="Calidad y utilidad de las recomendaciones adicionales",
                dimension=QualityDimension.CLARITY,
                weight=0.15,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Recomendaciones específicas, accionables y bien fundamentadas",
                    QualityLevel.GOOD: "Recomendaciones útiles con buen razonamiento",
                    QualityLevel.FAIR: "Recomendaciones básicas pero aceptables",
                    QualityLevel.POOR: "Recomendaciones vagas o genéricas",
                    QualityLevel.VERY_POOR: "Recomendaciones irrelevantes o incorrectas",
                }
            ),
            EvaluationCriterion(
                name="response_structure",
                description="Estructura y organización de la respuesta",
                dimension=QualityDimension.STRUCTURE,
                weight=0.1,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Respuesta perfectamente estructurada y fácil de seguir",
                    QualityLevel.GOOD: "Buena estructura con secciones claras",
                    QualityLevel.FAIR: "Estructura aceptable pero podría mejorarse",
                    QualityLevel.POOR: "Estructura confusa o desorganizada",
                    QualityLevel.VERY_POOR: "Sin estructura discernible",
                }
            ),
        ]

        self.rubrics["curator"] = AutomatedRubric(
            name="Curator Agent Evaluation Rubric",
            description="Rúbrica para evaluar calidad de gestión de itinerarios de aprendizaje",
            criteria=curator_criteria,
            agent_type="curator",
        )

        # Rúbrica para StudyPlanAgent
        study_plan_criteria = [
            EvaluationCriterion(
                name="schedule_realism",
                description="Realismo y factibilidad del cronograma propuesto",
                dimension=QualityDimension.ACCURACY,
                weight=0.3,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Cronograma perfectamente realista y achievable",
                    QualityLevel.GOOD: "Cronograma muy realista con buenos ajustes",
                    QualityLevel.FAIR: "Cronograma aceptable pero podría ser más realista",
                    QualityLevel.POOR: "Cronograma poco realista, sobrecarga de trabajo",
                    QualityLevel.VERY_POOR: "Cronograma completamente irreal, imposible de seguir",
                }
            ),
            EvaluationCriterion(
                name="milestone_appropriateness",
                description="Apropiación y utilidad de los hitos definidos",
                dimension=QualityDimension.RELEVANCE,
                weight=0.25,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Hitos perfectamente alineados con objetivos de aprendizaje",
                    QualityLevel.GOOD: "Hitos útiles y bien posicionados",
                    QualityLevel.FAIR: "Hitos aceptables pero podrían refinarse",
                    QualityLevel.POOR: "Hitos poco útiles o mal posicionados",
                    QualityLevel.VERY_POOR: "Hitos irrelevantes o contraproducentes",
                }
            ),
            EvaluationCriterion(
                name="resource_integration",
                description="Integración efectiva de recursos y herramientas",
                dimension=QualityDimension.COMPLETENESS,
                weight=0.2,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Integración perfecta de múltiples recursos complementarios",
                    QualityLevel.GOOD: "Buena integración de recursos diversos",
                    QualityLevel.FAIR: "Integración básica aceptable",
                    QualityLevel.POOR: "Integración limitada de recursos",
                    QualityLevel.VERY_POOR: "Sin integración de recursos o recursos inadecuados",
                }
            ),
            EvaluationCriterion(
                name="flexibility_adaptability",
                description="Flexibilidad y capacidad de adaptación del plan",
                dimension=QualityDimension.STRUCTURE,
                weight=0.15,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Plan altamente flexible y adaptable a cambios",
                    QualityLevel.GOOD: "Buena flexibilidad con opciones de ajuste",
                    QualityLevel.FAIR: "Flexibilidad aceptable",
                    QualityLevel.POOR: "Rigidez excesiva, difícil de adaptar",
                    QualityLevel.VERY_POOR: "Plan completamente rígido e inadaptable",
                }
            ),
            EvaluationCriterion(
                name="motivation_elements",
                description="Elementos motivacionales incluidos en el plan",
                dimension=QualityDimension.CREATIVITY,
                weight=0.1,
                scoring_guide={
                    QualityLevel.EXCELLENT: "Elementos motivacionales perfectamente integrados",
                    QualityLevel.GOOD: "Buenos elementos motivacionales incluidos",
                    QualityLevel.FAIR: "Elementos motivacionales básicos",
                    QualityLevel.POOR: "Pocos elementos motivacionales",
                    QualityLevel.VERY_POOR: "Sin elementos motivacionales",
                }
            ),
        ]

        self.rubrics["study_plan"] = AutomatedRubric(
            name="Study Plan Agent Evaluation Rubric",
            description="Rúbrica para evaluar calidad de planes de estudio generados",
            criteria=study_plan_criteria,
            agent_type="study_plan",
        )

        # Rúbricas para otros agentes pueden agregarse aquí...

    def evaluate_response(
        self,
        agent_type: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evalúa una respuesta de agente usando la rúbrica correspondiente.

        Args:
            agent_type: Tipo de agente que generó la respuesta.
            response: Respuesta del agente a evaluar.
            context: Contexto adicional para la evaluación.

        Returns:
            Resultado completo de la evaluación.

        Raises:
            ValueError: Si no existe rúbrica para el tipo de agente.
        """
        if agent_type not in self.rubrics:
            raise ValueError(f"No rubric defined for agent type: {agent_type}")

        rubric = self.rubrics[agent_type]
        criteria_scores = {}

        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion in rubric.criteria:
            score, level, reasoning = self._evaluate_criterion(
                criterion, response, context
            )
            criteria_scores[criterion.name] = {
                "score": score,
                "level": level.value,
                "reasoning": reasoning,
                "weight": criterion.weight,
            }

            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight

        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        overall_level = self._score_to_level(overall_score)

        feedback = self._generate_feedback(criteria_scores, overall_level)
        strengths = self._identify_strengths(criteria_scores)
        weaknesses = self._identify_weaknesses(criteria_scores)
        recommendations = self._generate_recommendations(
            criteria_scores, agent_type)

        return EvaluationResult(
            overall_score=round(overall_score, 1),
            overall_level=overall_level,
            criteria_scores=criteria_scores,
            feedback=feedback,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def _evaluate_criterion(
        self,
        criterion: EvaluationCriterion,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, QualityLevel, str]:
        """
        Evalúa un criterio específico usando análisis automatizado.

        Args:
            criterion: Criterio a evaluar.
            response: Respuesta del agente.
            context: Contexto adicional.

        Returns:
            Tupla de (puntuación, nivel, razonamiento).
        """
        # Implementación específica por criterio y tipo de agente
        if criterion.name == "relevance_to_certification":
            return self._evaluate_relevance_to_certification(criterion, response, context)
        elif criterion.name == "path_diversity":
            return self._evaluate_path_diversity(criterion, response, context)
        elif criterion.name == "schedule_realism":
            return self._evaluate_schedule_realism(criterion, response, context)
        # ... otros criterios ...

        # Fallback: evaluación básica
        return self._basic_criterion_evaluation(criterion, response)

    def _evaluate_relevance_to_certification(
        self,
        criterion: EvaluationCriterion,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, QualityLevel, str]:
        """
        Evalúa relevancia de rutas para certificación.
        """
        learning_paths = response.get("learning_paths", [])
        target_cert = context.get(
            "target_certification", "") if context else ""

        if not learning_paths:
            return 20.0, QualityLevel.VERY_POOR, "No se encontraron rutas de aprendizaje"

        relevant_count = 0
        total_paths = len(learning_paths)

        for path in learning_paths:
            path_title = path.get("title", "").lower()
            cert_lower = target_cert.lower()

            # Verificar si el título contiene referencias a la certificación
            if cert_lower in path_title or any(skill.lower() in path_title for skill in ["azure", "aws", "cloud"]):
                relevant_count += 1

        relevance_ratio = relevant_count / total_paths

        if relevance_ratio >= 0.9:
            return 95.0, QualityLevel.EXCELLENT, f"{relevant_count}/{total_paths} rutas altamente relevantes"
        elif relevance_ratio >= 0.7:
            return 80.0, QualityLevel.GOOD, f"{relevant_count}/{total_paths} rutas mayoritariamente relevantes"
        elif relevance_ratio >= 0.5:
            return 65.0, QualityLevel.FAIR, f"{relevant_count}/{total_paths} rutas parcialmente relevantes"
        elif relevance_ratio >= 0.3:
            return 45.0, QualityLevel.POOR, f"{relevant_count}/{total_paths} rutas poco relevantes"
        else:
            return 20.0, QualityLevel.VERY_POOR, f"{relevant_count}/{total_paths} rutas mayoritariamente irrelevantes"

    def _evaluate_path_diversity(
        self,
        criterion: EvaluationCriterion,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, QualityLevel, str]:
        """
        Evalúa diversidad de rutas de aprendizaje.
        """
        learning_paths = response.get("learning_paths", [])

        if not learning_paths:
            return 20.0, QualityLevel.VERY_POOR, "No se encontraron rutas de aprendizaje"

        difficulties = set()
        modules_count = 0

        for path in learning_paths:
            diff = path.get("difficulty", "")
            if diff:
                difficulties.add(diff)
            modules = path.get("modules", [])
            modules_count += len(modules)

        # Puntaje basado en variedad de dificultades y cantidad de módulos
        # Máximo 50 puntos por dificultades
        difficulty_score = min(len(difficulties) * 25, 50)
        # Máximo 50 puntos por módulos
        modules_score = min(modules_count * 2, 50)

        total_score = difficulty_score + modules_score

        if total_score >= 90:
            return 95.0, QualityLevel.EXCELLENT, f"Excelente diversidad: {len(difficulties)} dificultades, {modules_count} módulos"
        elif total_score >= 75:
            return 80.0, QualityLevel.GOOD, f"Buena diversidad: {len(difficulties)} dificultades, {modules_count} módulos"
        elif total_score >= 60:
            return 65.0, QualityLevel.FAIR, f"Diversidad aceptable: {len(difficulties)} dificultades, {modules_count} módulos"
        elif total_score >= 40:
            return 45.0, QualityLevel.POOR, f"Diversidad limitada: {len(difficulties)} dificultades, {modules_count} módulos"
        else:
            return 25.0, QualityLevel.VERY_POOR, f"Poca diversidad: {len(difficulties)} dificultades, {modules_count} módulos"

    def _evaluate_schedule_realism(
        self,
        criterion: EvaluationCriterion,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, QualityLevel, str]:
        """
        Evalúa realismo del cronograma de estudio.
        """
        study_plan = response.get("study_plan", {})
        total_hours = study_plan.get("total_estimated_hours", 0)
        sessions_per_week = study_plan.get("sessions_per_week", 0)
        hours_per_week = context.get(
            "available_hours_per_week", 10) if context else 10

        if total_hours == 0 or sessions_per_week == 0:
            return 20.0, QualityLevel.VERY_POOR, "Información de cronograma insuficiente"

        # Calcular horas por semana requeridas
        required_hours_week = sessions_per_week * 1.5  # Asumiendo 1.5 horas por sesión

        # Comparar con horas disponibles
        if required_hours_week <= hours_per_week * 0.8:  # 80% de disponibilidad
            return 95.0, QualityLevel.EXCELLENT, f"Cronograma realista: {required_hours_week:.1f}h/semana vs {hours_per_week}h disponibles"
        elif required_hours_week <= hours_per_week:
            return 80.0, QualityLevel.GOOD, f"Cronograma desafiante pero posible: {required_hours_week:.1f}h/semana vs {hours_per_week}h disponibles"
        elif required_hours_week <= hours_per_week * 1.3:
            return 65.0, QualityLevel.FAIR, f"Cronograma exigente: {required_hours_week:.1f}h/semana vs {hours_per_week}h disponibles"
        elif required_hours_week <= hours_per_week * 1.6:
            return 45.0, QualityLevel.POOR, f"Cronograma muy exigente: {required_hours_week:.1f}h/semana vs {hours_per_week}h disponibles"
        else:
            return 20.0, QualityLevel.VERY_POOR, f"Cronograma irreal: {required_hours_week:.1f}h/semana vs {hours_per_week}h disponibles"

    def _basic_criterion_evaluation(
        self,
        criterion: EvaluationCriterion,
        response: Dict[str, Any],
    ) -> Tuple[float, QualityLevel, str]:
        """
        Evaluación básica de criterio usando heurísticas simples.
        """
        # Evaluación básica basada en completitud de respuesta
        response_keys = set(response.keys())
        expected_keys = {"learning_paths", "recommendations",
                         "search_query", "total_paths_found"}

        coverage = len(response_keys.intersection(
            expected_keys)) / len(expected_keys)

        if coverage >= 0.9:
            return 85.0, QualityLevel.GOOD, f"Respuesta completa ({coverage:.1%} cobertura)"
        elif coverage >= 0.7:
            return 70.0, QualityLevel.FAIR, f"Respuesta aceptable ({coverage:.1%} cobertura)"
        elif coverage >= 0.5:
            return 50.0, QualityLevel.POOR, f"Respuesta incompleta ({coverage:.1%} cobertura)"
        else:
            return 30.0, QualityLevel.VERY_POOR, f"Respuesta muy incompleta ({coverage:.1%} cobertura)"

    def _score_to_level(self, score: float) -> QualityLevel:
        """
        Convierte puntuación numérica a nivel de calidad.

        Args:
            score: Puntuación de 0-100.

        Returns:
            Nivel de calidad correspondiente.
        """
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR

    def _generate_feedback(
        self,
        criteria_scores: Dict[str, Dict[str, Any]],
        overall_level: QualityLevel,
    ) -> str:
        """
        Genera feedback automático basado en evaluación.

        Args:
            criteria_scores: Puntuaciones por criterio.
            overall_level: Nivel general.

        Returns:
            Feedback generado.
        """
        feedback_parts = []

        if overall_level == QualityLevel.EXCELLENT:
            feedback_parts.append(
                "¡Excelente respuesta! La calidad es excepcional.")
        elif overall_level == QualityLevel.GOOD:
            feedback_parts.append("Buena respuesta con fortalezas notables.")
        elif overall_level == QualityLevel.FAIR:
            feedback_parts.append(
                "Respuesta aceptable que cumple los requisitos básicos.")
        elif overall_level == QualityLevel.POOR:
            feedback_parts.append(
                "Respuesta por debajo de los estándares esperados.")
        else:
            feedback_parts.append("Respuesta requiere mejoras significativas.")

        # Agregar detalles específicos
        high_scores = [name for name,
                       data in criteria_scores.items() if data["score"] >= 80]
        low_scores = [name for name,
                      data in criteria_scores.items() if data["score"] < 60]

        if high_scores:
            feedback_parts.append(f"Destacan: {', '.join(high_scores)}.")

        if low_scores:
            feedback_parts.append(
                f"Áreas para mejorar: {', '.join(low_scores)}.")

        return " ".join(feedback_parts)

    def _identify_strengths(self, criteria_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Identifica puntos fuertes de la evaluación.

        Args:
            criteria_scores: Puntuaciones por criterio.

        Returns:
            Lista de fortalezas.
        """
        strengths = []
        for name, data in criteria_scores.items():
            if data["score"] >= 80:
                strengths.append(
                    f"{name.replace('_', ' ').title()}: {data['reasoning']}")

        return strengths[:3]  # Máximo 3 fortalezas

    def _identify_weaknesses(self, criteria_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Identifica áreas de mejora de la evaluación.

        Args:
            criteria_scores: Puntuaciones por criterio.

        Returns:
            Lista de debilidades.
        """
        weaknesses = []
        for name, data in criteria_scores.items():
            if data["score"] < 70:
                weaknesses.append(
                    f"{name.replace('_', ' ').title()}: {data['reasoning']}")

        return weaknesses[:3]  # Máximo 3 debilidades

    def _generate_recommendations(
        self,
        criteria_scores: Dict[str, Dict[str, Any]],
        agent_type: str,
    ) -> List[str]:
        """
        Genera recomendaciones específicas basadas en evaluación.

        Args:
            criteria_scores: Puntuaciones por criterio.
            agent_type: Tipo de agente.

        Returns:
            Lista de recomendaciones.
        """
        recommendations = []

        for name, data in criteria_scores.items():
            if data["score"] < 70:
                if "relevance" in name:
                    recommendations.append(
                        "Mejorar la selección de contenido relevante para la certificación objetivo.")
                elif "diversity" in name:
                    recommendations.append(
                        "Incluir mayor variedad de rutas y niveles de dificultad.")
                elif "schedule" in name:
                    recommendations.append(
                        "Ajustar el cronograma para que sea más realista y achievable.")
                elif "structure" in name:
                    recommendations.append(
                        "Mejorar la organización y estructura de la respuesta.")

        if not recommendations:
            recommendations.append(
                "Mantener el nivel de calidad actual y buscar oportunidades de mejora incremental.")

        return recommendations[:3]  # Máximo 3 recomendaciones


# Instancia global del evaluador
rubrics_evaluator = AutomatedRubricsEvaluator()
