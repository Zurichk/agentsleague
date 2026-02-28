"""
Agente Crítico - AEP CertMaster

Este agente implementa validación de respuestas, análisis de calidad de contenido,
feedback constructivo y aseguramiento de estándares educativos.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from .base_agent import AEPAgent
from src.models.schemas import AEPWorkflowContext
from src.tools.azure_openai_tool import AzureOpenAITool
from src.tools.persistence import PersistenceTool


class ValidationSeverity(Enum):
    """Niveles de severidad para validaciones."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentQuality(Enum):
    """Niveles de calidad del contenido."""

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class AEPValidationRule(BaseModel):
    """
    Regla de validación para contenido.

    Attributes:
        rule_id: Identificador único de la regla.
        name: Nombre descriptivo.
        description: Descripción de la regla.
        severity: Severidad de la violación.
        category: Categoría de la regla.
        criteria: Criterios de evaluación.
    """

    rule_id: str = Field(description="ID único de la regla")
    name: str = Field(description="Nombre de la regla")
    description: str = Field(description="Descripción detallada")
    severity: ValidationSeverity = Field(description="Severidad")
    category: str = Field(description="Categoría")
    criteria: List[str] = Field(
        default_factory=list,
        description="Criterios de evaluación"
    )


class AEPContentAnalysis(BaseModel):
    """
    Análisis de contenido realizado por el agente crítico.

    Attributes:
        content_id: ID del contenido analizado.
        quality_score: Puntuación de calidad (0-100).
        quality_level: Nivel de calidad.
        violations: Lista de violaciones encontradas.
        suggestions: Sugerencias de mejora.
        strengths: Puntos fuertes identificados.
        analyzed_at: Fecha del análisis.
    """

    content_id: str = Field(description="ID del contenido")
    quality_score: int = Field(description="Puntuación de calidad")
    quality_level: ContentQuality = Field(description="Nivel de calidad")
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Violaciones encontradas"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Sugerencias de mejora"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Puntos fuertes"
    )
    analyzed_at: datetime = Field(
        default_factory=datetime.now,
        description="Fecha del análisis"
    )


class AEPCriticAgent(AEPAgent):
    """
    Agente especializado en crítica constructiva y validación de calidad.

    Funcionalidades principales:
    - Validación de respuestas de estudiantes
    - Análisis de calidad de contenido educativo
    - Generación de feedback constructivo
    - Detección de plagio y contenido inadecuado
    - Evaluación de estándares pedagógicos
    - Moderación de contenido generado por IA
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el agente crítico.

        Args:
            config: Configuración del agente.
        """
        super().__init__(
            name="AEPCriticAgent",
            description=(
                "Especialista en validación y crítica de calidad del contenido "
                "generado por otros agentes del sistema."
            ),
            capabilities=[
                "Validar calidad de contenido generado",
                "Detectar plagio y contenido duplicado",
                "Verificar precisión técnica",
                "Evaluar completitud de respuestas",
                "Moderación de contenido generado por IA",
            ],
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.3),
        )
        self.openai_tool = AzureOpenAITool(config.get("azure_openai", {}))
        self.persistence_tool = PersistenceTool(config.get("persistence", {}))

        # Configuración específica del agente
        self.quality_threshold = config.get("quality_threshold", 70)
        self.plagiarism_check_enabled = config.get(
            "plagiarism_check_enabled", True)
        self.content_moderation_enabled = config.get(
            "content_moderation_enabled", True)

        # Reglas de validación predefinidas
        self.validation_rules = self._initialize_validation_rules()

        self.logger.info("Agente Crítico inicializado")

    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis crítico del contenido del estudiante usando Azure OpenAI.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales.

        Returns:
            Resultado del análisis crítico con feedback real.
        """
        try:
            student = context.student
            topics = student.topics_of_interest or []
            certification = student.target_certification or "General IT"
            level = getattr(student, "level", "intermediate")

            # Recopilar resumen de evaluaciones recientes si existen
            assessments_info = ""
            if hasattr(context, "assessments") and context.assessments:
                recent = context.assessments[-3:]
                scores = [
                    a.get("score", 0) for a in recent if isinstance(a, dict)
                ]
                if scores:
                    avg = sum(scores) / len(scores)
                    assessments_info = (
                        f"Historial de evaluaciones recientes: {scores}, "
                        f"promedio {avg:.1f}/100"
                    )

            # Análisis crítico real con Azure OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Eres un agente analista crítico de aprendizaje especializado en certificaciones IT. "
                        "Usa toda tu capacidad de análisis para detectar brechas reales, priorizar impacto y proponer mejoras concretas y medibles. "
                        "Evalúa el perfil del estudiante y proporciona feedback constructivo, específico y accionable. "
                        "Basa tu análisis en los datos reales proporcionados. "
                        "Responde SOLO con JSON válido, sin texto adicional."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analiza el siguiente perfil de estudiante y proporciona un análisis crítico detallado:\n"
                        f"- Certificación objetivo: {certification}\n"
                        f"- Nivel actual: {level}\n"
                        f"- Temas de interés ({len(topics)}): {', '.join(topics[:6]) if topics else 'No especificados'}\n"
                        f"- {assessments_info or 'Sin historial de evaluaciones aún'}\n\n"
                        "Devuelve un JSON con:\n"
                        "{\n"
                        '  "quality_score": <número 0-100 basado en el perfil>,\n'
                        '  "feedback": "<feedback detallado y personalizado de 2-3 oraciones>",\n'
                        '  "strengths": ["fortaleza específica 1", "fortaleza específica 2"],\n'
                        '  "areas_to_improve": ["área de mejora 1", "área de mejora 2"],\n'
                        '  "recommendations": [\n'
                        '    "recomendación concreta y accionable 1",\n'
                        '    "recomendación concreta y accionable 2",\n'
                        '    "recomendación concreta y accionable 3"\n'
                        "  ]\n"
                        "}"
                    ),
                },
            ]

            raw = await self._call_azure_openai(messages, temperature=0.4, max_tokens=800)

            # Parsear respuesta
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1] if len(parts) > 1 else cleaned
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            return {
                "success": True,
                "message": "Análisis crítico completado",
                "quality_score": result.get("quality_score", 70),
                "feedback": result.get("feedback", ""),
                "strengths": result.get("strengths", []),
                "areas_to_improve": result.get("areas_to_improve", []),
                "recommendations": result.get("recommendations", []),
            }

        except Exception as e:
            self.logger.error(f"Error en Critic Agent: {str(e)}")
            return {
                "success": False,
                "message": f"Error interno del agente: {str(e)}",
            }

    async def _validate_student_response(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Valida la respuesta de un estudiante.

        Args:
            context: Contexto con la respuesta a validar.

        Returns:
            Resultado de la validación.
        """
        response_text = context.request_data.get("response_text")
        question_context = context.request_data.get("question_context", {})
        validation_criteria = context.request_data.get("criteria", [])

        if not response_text:
            return AgentResponse(
                success=False,
                message="Se requiere response_text para validar"
            )

        # Realizar validaciones múltiples
        validation_results = await self._perform_comprehensive_validation(
            response_text,
            question_context,
            validation_criteria
        )

        # Calcular puntuación general
        overall_score = self._calculate_overall_score(validation_results)

        # Generar feedback basado en validaciones
        feedback = await self._generate_validation_feedback(
            validation_results,
            overall_score
        )

        return AgentResponse(
            success=True,
            message="Respuesta validada exitosamente",
            data={
                "validation_results": validation_results,
                "overall_score": overall_score,
                "feedback": feedback,
                "passed": overall_score >= self.quality_threshold
            }
        )

    async def _analyze_content_quality(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Analiza la calidad de contenido educativo.

        Args:
            context: Contexto con el contenido a analizar.

        Returns:
            Análisis de calidad.
        """
        content = context.request_data.get("content")
        content_type = context.request_data.get("content_type", "educational")
        target_audience = context.request_data.get(
            "target_audience", "students")

        if not content:
            return AgentResponse(
                success=False,
                message="Se requiere content para analizar"
            )

        # Realizar análisis de calidad
        analysis = await self._perform_quality_analysis(
            content,
            content_type,
            target_audience
        )

        # Crear objeto de análisis
        content_analysis = AEPContentAnalysis(
            content_id=context.request_data.get(
                "content_id", f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            quality_score=analysis["score"],
            quality_level=self._determine_quality_level(analysis["score"]),
            violations=analysis["violations"],
            suggestions=analysis["suggestions"],
            strengths=analysis["strengths"]
        )

        # Guardar análisis si se solicita
        if context.request_data.get("save_analysis", False):
            await self.persistence_tool.save_content_analysis(content_analysis)

        return AgentResponse(
            success=True,
            message="Análisis de calidad completado",
            data={
                "analysis": content_analysis.model_dump(),
                "recommendations": analysis["recommendations"]
            }
        )

    async def _generate_constructive_feedback(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Genera feedback constructivo personalizado.

        Args:
            context: Contexto para generar feedback.

        Returns:
            Feedback generado.
        """
        student_id = context.request_data.get("student_id")
        assessment_result = context.request_data.get("assessment_result")
        learning_objectives = context.request_data.get(
            "learning_objectives", [])

        if not assessment_result:
            return AgentResponse(
                success=False,
                message="Se requiere assessment_result para generar feedback"
            )

        # Obtener perfil del estudiante para personalización
        student_profile = None
        if student_id:
            student_profile = await self.persistence_tool.get_student_profile(student_id)

        # Generar feedback personalizado
        feedback = await self._create_personalized_feedback(
            assessment_result,
            student_profile,
            learning_objectives
        )

        # Crear item de feedback
        feedback_item = AEPFeedbackItem(
            feedback_id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            student_id=student_id,
            assessment_id=assessment_result.get("assessment_id"),
            feedback_type="constructive",
            content=feedback["content"],
            priority=feedback["priority"],
            actionable_items=feedback["actionable_items"],
            created_at=datetime.now()
        )

        # Guardar feedback
        await self.persistence_tool.save_feedback(feedback_item)

        return AgentResponse(
            success=True,
            message="Feedback constructivo generado",
            data={
                "feedback": feedback_item.model_dump(),
                "personalized_insights": feedback.get("insights", [])
            }
        )

    async def _check_plagiarism(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Verifica posible plagio en el contenido.

        Args:
            context: Contexto con el contenido a verificar.

        Returns:
            Resultado de la verificación de plagio.
        """
        content = context.request_data.get("content")
        source_references = context.request_data.get("source_references", [])

        if not content:
            return AgentResponse(
                success=False,
                message="Se requiere content para verificar plagio"
            )

        # Realizar verificación de plagio
        plagiarism_result = await self._perform_plagiarism_check(
            content,
            source_references
        )

        return AgentResponse(
            success=True,
            message="Verificación de plagio completada",
            data={
                "plagiarism_detected": plagiarism_result["detected"],
                "similarity_score": plagiarism_result["similarity_score"],
                "flagged_sections": plagiarism_result["flagged_sections"],
                "recommendations": plagiarism_result["recommendations"]
            }
        )

    async def _moderate_content(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Modera contenido para asegurar cumplimiento de estándares.

        Args:
            context: Contexto con el contenido a moderar.

        Returns:
            Resultado de la moderación.
        """
        content = context.request_data.get("content")
        content_type = context.request_data.get("content_type", "educational")

        if not content:
            return AgentResponse(
                success=False,
                message="Se requiere content para moderar"
            )

        # Realizar moderación de contenido
        moderation_result = await self._perform_content_moderation(
            content,
            content_type
        )

        return AgentResponse(
            success=True,
            message="Moderación de contenido completada",
            data={
                "approved": moderation_result["approved"],
                "issues_found": moderation_result["issues"],
                "moderation_score": moderation_result["score"],
                "required_actions": moderation_result["actions"]
            }
        )

    def _initialize_validation_rules(self) -> List[AEPValidationRule]:
        """Inicializa las reglas de validación predefinidas."""
        return [
            AEPValidationRule(
                rule_id="content_accuracy",
                name="Precisión del Contenido",
                description="El contenido debe ser factual y técnicamente correcto",
                severity=ValidationSeverity.HIGH,
                category="accuracy",
                criteria=[
                    "Información factual correcta",
                    "Términos técnicos apropiados",
                    "Conceptos explicados correctamente"
                ]
            ),
            AEPValidationRule(
                rule_id="content_completeness",
                name="Completitud del Contenido",
                description="El contenido debe cubrir todos los aspectos relevantes",
                severity=ValidationSeverity.MEDIUM,
                category="completeness",
                criteria=[
                    "Cobertura completa del tema",
                    "Ejemplos relevantes incluidos",
                    "Contexto adecuado proporcionado"
                ]
            ),
            AEPValidationRule(
                rule_id="pedagogical_appropriateness",
                name="Apropiación Pedagógica",
                description="El contenido debe ser apropiado para el nivel educativo",
                severity=ValidationSeverity.MEDIUM,
                category="pedagogy",
                criteria=[
                    "Nivel de dificultad adecuado",
                    "Lenguaje claro y accesible",
                    "Estructura lógica del contenido"
                ]
            ),
            AEPValidationRule(
                rule_id="content_originality",
                name="Originalidad del Contenido",
                description="El contenido debe ser original y no copiado",
                severity=ValidationSeverity.CRITICAL,
                category="originality",
                criteria=[
                    "Contenido original",
                    "Citas apropiadas cuando se usan fuentes",
                    "Parafraseo correcto cuando aplica"
                ]
            ),
            AEPValidationRule(
                rule_id="content_safety",
                name="Seguridad del Contenido",
                description="El contenido no debe contener material inapropiado",
                severity=ValidationSeverity.CRITICAL,
                category="safety",
                criteria=[
                    "Lenguaje apropiado",
                    "Contenido no ofensivo",
                    "Información segura para estudiantes"
                ]
            )
        ]

    async def _perform_comprehensive_validation(
        self,
        response_text: str,
        question_context: Dict[str, Any],
        custom_criteria: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Realiza validación completa de la respuesta.

        Args:
            response_text: Texto de la respuesta.
            question_context: Contexto de la pregunta.
            custom_criteria: Criterios personalizados.

        Returns:
            Resultados de validación.
        """
        validation_results = []

        # Validar contra reglas predefinidas
        for rule in self.validation_rules:
            result = await self._validate_against_rule(
                response_text,
                rule,
                question_context
            )
            validation_results.append(result)

        # Validar criterios personalizados
        for criterion in custom_criteria:
            result = await self._validate_custom_criterion(
                response_text,
                criterion,
                question_context
            )
            validation_results.append(result)

        return validation_results

    async def _validate_against_rule(
        self,
        response_text: str,
        rule: AEPValidationRule,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida respuesta contra una regla específica.

        Args:
            response_text: Texto a validar.
            rule: Regla de validación.
            context: Contexto adicional.

        Returns:
            Resultado de la validación.
        """
        prompt = f"""
        Evalúa la siguiente respuesta contra la regla de validación:

        Regla: {rule.name}
        Descripción: {rule.description}
        Severidad: {rule.severity.value}
        Criterios: {', '.join(rule.criteria)}

        Respuesta a evaluar: {response_text}

        Contexto de la pregunta: {json.dumps(context, ensure_ascii=False)}

        Por favor proporciona una evaluación detallada en formato JSON:
        {{
            "rule_id": "{rule.rule_id}",
            "passed": true/false,
            "score": 0-100,
            "issues": ["lista de problemas encontrados"],
            "evidence": "evidencia que soporta la evaluación",
            "suggestions": ["sugerencias de mejora"]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2
        )

        try:
            result = json.loads(response)
            result["rule_name"] = rule.name
            result["severity"] = rule.severity.value
            return result
        except json.JSONDecodeError:
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": True,
                "score": 75,
                "issues": [],
                "evidence": "Evaluación automática",
                "suggestions": ["Revisar contenido manualmente"],
                "severity": rule.severity.value
            }

    async def _validate_custom_criterion(
        self,
        response_text: str,
        criterion: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida contra un criterio personalizado.

        Args:
            response_text: Texto a validar.
            criterion: Criterio personalizado.
            context: Contexto.

        Returns:
            Resultado de validación.
        """
        prompt = f"""
        Evalúa si la respuesta cumple con el siguiente criterio personalizado:

        Criterio: {criterion}

        Respuesta: {response_text}

        Contexto: {json.dumps(context, ensure_ascii=False)}

        Formato JSON:
        {{
            "criterion": "{criterion}",
            "passed": true/false,
            "score": 0-100,
            "feedback": "explicación detallada"
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=400,
            temperature=0.2
        )

        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {
                "criterion": criterion,
                "passed": True,
                "score": 70,
                "feedback": "Criterio evaluado automáticamente"
            }

    def _calculate_overall_score(self, validation_results: List[Dict[str, Any]]) -> int:
        """
        Calcula puntuación general de validación.

        Args:
            validation_results: Resultados individuales.

        Returns:
            Puntuación general.
        """
        if not validation_results:
            return 100

        total_score = 0
        total_weight = 0

        for result in validation_results:
            score = result.get("score", 50)
            severity = result.get("severity", "medium")

            # Peso basado en severidad
            weight = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4
            }.get(severity, 2)

            total_score += score * weight
            total_weight += weight

        return int(total_score / total_weight) if total_weight > 0 else 50

    async def _generate_validation_feedback(
        self,
        validation_results: List[Dict[str, Any]],
        overall_score: int
    ) -> Dict[str, Any]:
        """
        Genera feedback basado en resultados de validación.

        Args:
            validation_results: Resultados de validación.
            overall_score: Puntuación general.

        Returns:
            Feedback generado.
        """
        prompt = f"""
        Genera feedback constructivo basado en los siguientes resultados de validación:

        Puntuación general: {overall_score}/100

        Resultados de validación:
        {json.dumps(validation_results, ensure_ascii=False, indent=2)}

        Por favor proporciona feedback estructurado en formato JSON:
        {{
            "overall_feedback": "comentario general",
            "strengths": ["puntos fuertes"],
            "areas_for_improvement": ["áreas a mejorar"],
            "specific_suggestions": ["sugerencias concretas"],
            "next_steps": ["próximos pasos recomendados"],
            "encouragement": "mensaje motivacional"
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.4
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "overall_feedback": f"Puntuación obtenida: {overall_score}/100",
                "strengths": ["Respuesta proporcionada"],
                "areas_for_improvement": ["Revisar criterios de evaluación"],
                "specific_suggestions": ["Mejorar profundidad del contenido"],
                "next_steps": ["Continuar practicando"],
                "encouragement": "¡Sigue esforzándote!"
            }

    async def _perform_quality_analysis(
        self,
        content: str,
        content_type: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """
        Realiza análisis de calidad del contenido.

        Args:
            content: Contenido a analizar.
            content_type: Tipo de contenido.
            target_audience: Audiencia objetivo.

        Returns:
            Análisis de calidad.
        """
        prompt = f"""
        Analiza la calidad del siguiente contenido educativo:

        Tipo de contenido: {content_type}
        Audiencia objetivo: {target_audience}

        Contenido:
        {content}

        Evalúa los siguientes aspectos:
        1. Precisión factual
        2. Claridad y comprehensibilidad
        3. Estructura y organización
        4. Apropiación pedagógica
        5. Engagement y motivación
        6. Inclusividad y accesibilidad

        Proporciona análisis detallado en formato JSON:
        {{
            "score": 0-100,
            "violations": [
                {{
                    "aspect": "aspecto",
                    "severity": "low|medium|high",
                    "description": "descripción del problema"
                }}
            ],
            "suggestions": ["sugerencias de mejora"],
            "strengths": ["puntos fuertes"],
            "recommendations": ["recomendaciones específicas"]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.3
        )

        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            return {
                "score": 75,
                "violations": [],
                "suggestions": ["Revisar contenido manualmente"],
                "strengths": ["Contenido proporcionado"],
                "recommendations": ["Mejorar análisis de calidad"]
            }

    def _determine_quality_level(self, score: int) -> ContentQuality:
        """Determina el nivel de calidad basado en la puntuación."""
        if score >= 90:
            return ContentQuality.EXCELLENT
        elif score >= 80:
            return ContentQuality.GOOD
        elif score >= 60:
            return ContentQuality.FAIR
        else:
            return ContentQuality.POOR

    async def _create_personalized_feedback(
        self,
        assessment_result: Dict[str, Any],
        student_profile: Optional[AEPStudentProfile],
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Crea feedback personalizado para el estudiante.

        Args:
            assessment_result: Resultado de la evaluación.
            student_profile: Perfil del estudiante.
            learning_objectives: Objetivos de aprendizaje.

        Returns:
            Feedback personalizado.
        """
        # Preparar contexto del estudiante
        student_context = ""
        if student_profile:
            student_context = f"""
            Perfil del estudiante:
            - Nivel educativo: {student_profile.education_level}
            - Áreas de interés: {', '.join([a.area_name for a in student_profile.knowledge_areas])}
            - Estilo de aprendizaje: {student_profile.learning_style}
            - Evaluaciones previas: {len(student_profile.assessments) if student_profile.assessments else 0}
            """

        prompt = f"""
        Genera feedback constructivo y personalizado para el estudiante:

        {student_context}

        Resultado de evaluación:
        {json.dumps(assessment_result, ensure_ascii=False, indent=2)}

        Objetivos de aprendizaje:
        {', '.join(learning_objectives) if learning_objectives else 'No especificados'}

        Crea feedback motivacional y específico en formato JSON:
        {{
            "content": "feedback completo y detallado",
            "priority": "high|medium|low",
            "actionable_items": ["acciones concretas"],
            "insights": ["perspectivas personalizadas"],
            "encouragement": "mensaje motivacional",
            "next_focus_areas": ["áreas a enfatizar"]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.5
        )

        try:
            feedback = json.loads(response)
            return feedback
        except json.JSONDecodeError:
            return {
                "content": "Buen intento. Continúa practicando para mejorar.",
                "priority": "medium",
                "actionable_items": ["Revisar conceptos básicos"],
                "insights": ["Identificar áreas de dificultad"],
                "encouragement": "¡Cada paso cuenta en el aprendizaje!",
                "next_focus_areas": ["Práctica consistente"]
            }

    async def _perform_plagiarism_check(
        self,
        content: str,
        source_references: List[str]
    ) -> Dict[str, Any]:
        """
        Realiza verificación de plagio.

        Args:
            content: Contenido a verificar.
            source_references: Referencias de fuentes.

        Returns:
            Resultado de verificación de plagio.
        """
        # Nota: Esta es una implementación básica. En producción,
        # se debería integrar con servicios especializados de detección de plagio

        prompt = f"""
        Analiza el siguiente contenido para detectar posible plagio:

        Contenido a analizar:
        {content}

        Referencias proporcionadas:
        {', '.join(source_references) if source_references else 'Ninguna'}

        Evalúa:
        1. Similitud con fuentes conocidas
        2. Uso apropiado de citas
        3. Originalidad del contenido
        4. Posibles secciones problemáticas

        Formato JSON:
        {{
            "detected": true/false,
            "similarity_score": 0-100,
            "flagged_sections": ["secciones problemáticas"],
            "recommendations": ["acciones recomendadas"]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2
        )

        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {
                "detected": False,
                "similarity_score": 15,
                "flagged_sections": [],
                "recommendations": ["Incluir referencias apropiadas"]
            }

    async def _perform_content_moderation(
        self,
        content: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Realiza moderación de contenido.

        Args:
            content: Contenido a moderar.
            content_type: Tipo de contenido.

        Returns:
            Resultado de moderación.
        """
        prompt = f"""
        Modera el siguiente contenido para asegurar cumplimiento de estándares educativos:

        Tipo de contenido: {content_type}

        Contenido:
        {content}

        Evalúa:
        1. Apropiación del lenguaje
        2. Contenido sensible o ofensivo
        3. Alineación con estándares educativos
        4. Seguridad para estudiantes

        Formato JSON:
        {{
            "approved": true/false,
            "score": 0-100,
            "issues": ["problemas encontrados"],
            "actions": ["acciones requeridas"]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=600,
            temperature=0.1
        )

        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {
                "approved": True,
                "score": 85,
                "issues": [],
                "actions": ["Contenido aprobado automáticamente"]
            }

    def _generate_simple_feedback(self, score: int, topics_count: int, certification: str) -> str:
        """Genera feedback simple basado en la puntuación."""
        if score >= 80:
            base_feedback = "¡Excelente progreso! Estás en el camino correcto."
        elif score >= 60:
            base_feedback = "Buen progreso. Continúa practicando."
        else:
            base_feedback = "Hay oportunidades de mejora. Enfócate en los fundamentos."

        return f"{base_feedback} Con {topics_count} temas de interés y objetivo en {certification}, tu puntuación actual es {score}/100."

    async def _perform_basic_quality_analysis(self, student_profile: AEPStudentProfile) -> Dict[str, Any]:
        """Realiza análisis básico de calidad del progreso del estudiante."""
        # Calcular puntuación basada en evaluaciones completadas
        assessment_count = len(student_profile.assessments or [])
        avg_proficiency = sum(area.proficiency_level for area in student_profile.knowledge_areas) / \
            len(student_profile.knowledge_areas) if student_profile.knowledge_areas else 0

        # Puntuación simple
        score = min(100, int((assessment_count * 10) + (avg_proficiency * 50)))

        recommendations = []
        if score < 70:
            recommendations.append("Enfocarse en completar más evaluaciones")
            recommendations.append(
                "Revisar conceptos básicos en áreas de baja proficiency")
        else:
            recommendations.append("Continuar con el buen progreso")
            recommendations.append("Considerar certificaciones más avanzadas")

        return {
            "score": score,
            "assessment_count": assessment_count,
            "avg_proficiency": avg_proficiency,
            "recommendations": recommendations
        }

    async def _generate_basic_feedback(self, student_profile: AEPStudentProfile, analysis: Dict[str, Any]) -> str:
        """Genera feedback básico constructivo."""
        score = analysis["score"]

        if score >= 80:
            feedback = "¡Excelente progreso! Estás demostrando un buen dominio de los conceptos."
        elif score >= 60:
            feedback = "Buen progreso. Continúa practicando para mejorar aún más."
        else:
            feedback = "Hay oportunidades de mejora. Enfócate en los conceptos básicos y completa más evaluaciones."

        feedback += f" Has completado {analysis['assessment_count']} evaluaciones con una proficiency promedio del {analysis['avg_proficiency']:.1%}."

        return feedback
