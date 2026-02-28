"""
Agente de Evaluación - AEP CertMaster

Este agente implementa evaluaciones inteligentes basadas en la taxonomía de Bloom,
preguntas adaptativas y análisis de respuestas para medir el progreso del estudiante.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from .base_agent import AEPAgent
from ..models.schemas import AEPWorkflowContext
from ..tools.azure_openai_tool import AzureOpenAITool
from ..tools.persistence import PersistenceTool


class BloomLevel(Enum):
    """Niveles de la taxonomía de Bloom."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class QuestionType(Enum):
    """Tipos de preguntas disponibles."""

    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CODING = "coding"
    SCENARIO = "scenario"


class AssessmentDifficulty(Enum):
    """Niveles de dificultad de evaluación."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AEPAssessmentAgent(AEPAgent):
    """
    Agente especializado en evaluaciones inteligentes.

    Funcionalidades principales:
    - Evaluaciones basadas en taxonomía de Bloom
    - Preguntas adaptativas según nivel del estudiante
    - Análisis automático de respuestas
    - Generación de feedback personalizado
    - Seguimiento de progreso por área de conocimiento
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el agente de evaluación.

        Args:
            config: Configuración del agente.
        """
        super().__init__(
            name="AEPAssessmentAgent",
            description=(
                "Especialista en evaluaciones inteligentes con preguntas "
                "adaptativas basadas en taxonomía de Bloom."
            ),
            capabilities=[
                "Generar evaluaciones adaptativas",
                "Analizar respuestas automáticamente",
                "Proporcionar feedback personalizado",
                "Seguimiento de progreso por área",
                "Ajustar dificultad dinámicamente",
            ],
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
        )
        self.openai_tool = AzureOpenAITool(config.get("azure_openai", {}))
        self.persistence_tool = PersistenceTool(config.get("persistence", {}))

        # Configuración específica del agente
        self.max_questions_per_assessment = config.get(
            "max_questions_per_assessment", 20)
        self.adaptive_threshold = config.get("adaptive_threshold", 0.7)
        self.feedback_depth = config.get("feedback_depth", "detailed")

        # Cache de preguntas por área y nivel
        self.question_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.logger.info("Agente de Evaluación inicializado")

    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta evaluación del estudiante generando preguntas reales con Azure OpenAI.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales.

        Returns:
            Resultado de la evaluación con preguntas reales.
        """
        try:
            student = context.student
            topics = student.topics_of_interest or []
            certification = student.target_certification or "General IT"
            level = getattr(student, "level", "intermediate")
            priority_areas = topics[:4] if topics else [certification]
            question_count = min(self.max_questions_per_assessment, max(
                3, len(priority_areas) * 2))

            # Generar preguntas reales con Azure OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en evaluaciones para certificaciones IT. "
                        "Usa toda tu capacidad para generar preguntas técnicamente rigurosas, no genéricas y alineadas al contexto del estudiante. "
                        "Genera preguntas de opción múltiple rigurosas y relevantes en JSON. "
                        "Responde SOLO con JSON válido, sin texto adicional."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Genera {question_count} preguntas de opción múltiple para:\n"
                        f"- Certificación: {certification}\n"
                        f"- Nivel del estudiante: {level}\n"
                        f"- Temas: {', '.join(priority_areas)}\n\n"
                        "Sigue los niveles de la taxonomía de Bloom (remember, understand, apply, analyze).\n"
                        "Devuelve un array JSON:\n"
                        "[\n"
                        "  {\n"
                        '    "question_id": "q1",\n'
                        '    "question": "texto de la pregunta",\n'
                        '    "options": {"a": "opción A", "b": "opción B", "c": "opción C", "d": "opción D"},\n'
                        '    "correct": "a",\n'
                        '    "explanation": "explicación de la respuesta correcta",\n'
                        '    "topic": "tema o área",\n'
                        '    "bloom_level": "remember|understand|apply|analyze",\n'
                        '    "difficulty": "beginner|intermediate|advanced"\n'
                        "  }\n"
                        "]"
                    ),
                },
            ]

            raw = await self._call_azure_openai(messages, temperature=0.5, max_tokens=3000)

            # Parsear respuesta
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1] if len(parts) > 1 else cleaned
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            questions = json.loads(cleaned)

            self.logger.info(
                f"Evaluación generada: {len(questions)} preguntas para {certification}"
            )

            return {
                "success": True,
                "message": "Evaluación generada exitosamente",
                "student_id": student.student_id,
                "target_certification": certification,
                "question_count": len(questions),
                "questions": questions,
                "topics_covered": priority_areas,
                # 2 minutos por pregunta
                "estimated_duration": len(questions) * 2,
                "difficulty": level,
            }

        except Exception as e:
            self.logger.error(f"Error en Assessment Agent: {str(e)}")
            return {
                "success": False,
                "message": f"Error interno del agente: {str(e)}",
            }

    async def _create_assessment(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Crea una nueva evaluación adaptativa.

        Args:
            context: Contexto de la solicitud.

        Returns:
            Respuesta con la evaluación creada.
        """
        student_id = context.request_data.get("student_id")
        if not student_id:
            return AgentResponse(
                success=False,
                message="Se requiere student_id para crear evaluación"
            )

        # Obtener perfil del estudiante
        student_profile = await self._get_student_profile(student_id)
        if not student_profile:
            return {
                "success": False,
                "message": f"Perfil de estudiante no encontrado: {student_id}"
            }

        # Determinar áreas de conocimiento a evaluar
        knowledge_areas = context.request_data.get("knowledge_areas", [])
        if not knowledge_areas:
            # Usar áreas del perfil del estudiante
            knowledge_areas = [
                area.area_name for area in student_profile.knowledge_areas]

        # Generar evaluación adaptativa
        assessment = await self._generate_adaptive_assessment(
            student_profile,
            knowledge_areas,
            context.request_data.get("difficulty", "intermediate")
        )

        # Guardar evaluación
        await self.persistence_tool.save_assessment(assessment)

        return AgentResponse(
            success=True,
            message="Evaluación creada exitosamente",
            data={
                "assessment": assessment.model_dump(),
                "question_count": len(assessment.questions),
                "estimated_duration": self._estimate_duration(assessment)
            }
        )

    async def _evaluate_response(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Evalúa la respuesta del estudiante a una pregunta.

        Args:
            context: Contexto con la respuesta del estudiante.

        Returns:
            Resultado de la evaluación.
        """
        assessment_id = context.request_data.get("assessment_id")
        question_id = context.request_data.get("question_id")
        student_response = context.request_data.get("response")

        if not all([assessment_id, question_id, student_response]):
            return AgentResponse(
                success=False,
                message="Se requieren assessment_id, question_id y response"
            )

        # Obtener evaluación y pregunta
        assessment = await self.persistence_tool.get_assessment(assessment_id)
        if not assessment:
            return AgentResponse(
                success=False,
                message=f"Evaluación no encontrada: {assessment_id}"
            )

        question = next(
            (q for q in assessment.questions if q.question_id == question_id),
            None
        )
        if not question:
            return AgentResponse(
                success=False,
                message=f"Pregunta no encontrada: {question_id}"
            )

        # Evaluar respuesta
        evaluation_result = await self._evaluate_student_response(
            question,
            student_response
        )

        # Actualizar resultado en la evaluación
        result = AEPAssessmentResult(
            question_id=question_id,
            student_response=student_response,
            score=evaluation_result["score"],
            feedback=evaluation_result["feedback"],
            evaluated_at=datetime.now(),
            time_spent=context.request_data.get("time_spent", 0)
        )

        assessment.results.append(result)

        # Calcular progreso adaptativo
        if evaluation_result["score"] >= self.adaptive_threshold:
            # Pregunta más difícil siguiente
            next_difficulty = self._increase_difficulty(question.difficulty)
        else:
            # Pregunta más fácil siguiente
            next_difficulty = self._decrease_difficulty(question.difficulty)

        # Guardar evaluación actualizada
        await self.persistence_tool.save_assessment(assessment)

        return AgentResponse(
            success=True,
            message="Respuesta evaluada exitosamente",
            data={
                "evaluation": evaluation_result,
                "next_difficulty": next_difficulty,
                "assessment_progress": self._calculate_progress(assessment)
            }
        )

    async def _get_progress_report(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Genera un reporte de progreso del estudiante.

        Args:
            context: Contexto de la solicitud.

        Returns:
            Reporte de progreso.
        """
        student_id = context.request_data.get("student_id")
        if not student_id:
            return AgentResponse(
                success=False,
                message="Se requiere student_id para generar reporte"
            )

        # Obtener todas las evaluaciones del estudiante
        assessments = await self.persistence_tool.get_student_assessments(student_id)

        if not assessments:
            return AgentResponse(
                success=True,
                message="No hay evaluaciones para este estudiante",
                data={"assessments_completed": 0}
            )

        # Calcular métricas de progreso
        progress_report = await self._calculate_progress_metrics(assessments)

        return AgentResponse(
            success=True,
            message="Reporte de progreso generado",
            data=progress_report
        )

    async def _generate_adaptive_question(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Genera una pregunta adaptativa basada en el rendimiento anterior.

        Args:
            context: Contexto de la solicitud.

        Returns:
            Pregunta adaptativa generada.
        """
        student_id = context.request_data.get("student_id")
        knowledge_area = context.request_data.get("knowledge_area")
        current_difficulty = context.request_data.get(
            "current_difficulty", "intermediate")

        if not all([student_id, knowledge_area]):
            return AgentResponse(
                success=False,
                message="Se requieren student_id y knowledge_area"
            )

        # Obtener historial de rendimiento
        performance_history = await self._get_performance_history(
            student_id,
            knowledge_area
        )

        # Determinar dificultad óptima
        optimal_difficulty = self._calculate_optimal_difficulty(
            performance_history,
            current_difficulty
        )

        # Generar pregunta adaptativa
        question = await self._generate_question(
            knowledge_area,
            optimal_difficulty,
            context.request_data.get("bloom_level", "apply")
        )

        return AgentResponse(
            success=True,
            message="Pregunta adaptativa generada",
            data={
                "question": question,
                "optimal_difficulty": optimal_difficulty,
                "adaptation_reason": self._explain_adaptation(performance_history)
            }
        )

    async def _generate_adaptive_assessment(
        self,
        student_profile: AEPStudentProfile,
        knowledge_areas: List[str],
        base_difficulty: str
    ) -> AEPAssessment:
        """
        Genera una evaluación adaptativa completa.

        Args:
            student_profile: Perfil del estudiante.
            knowledge_areas: Áreas de conocimiento a evaluar.
            base_difficulty: Dificultad base.

        Returns:
            Evaluación generada.
        """
        questions = []

        for area in knowledge_areas:
            # Determinar dificultad inicial basada en el perfil
            area_knowledge = next(
                (k for k in student_profile.knowledge_areas if k.area_name == area),
                None
            )

            difficulty = self._determine_initial_difficulty(
                area_knowledge,
                base_difficulty
            )

            # Generar preguntas por nivel de Bloom
            bloom_levels = [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                            BloomLevel.APPLY, BloomLevel.ANALYZE]

            questions_per_level = self.max_questions_per_assessment // len(
                bloom_levels)

            for bloom_level in bloom_levels:
                level_questions = await self._generate_questions_for_level(
                    area,
                    difficulty,
                    bloom_level.value,
                    questions_per_level
                )
                questions.extend(level_questions)

        return AEPAssessment(
            assessment_id=f"assessment_{student_profile.student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            student_id=student_profile.student_id,
            title=f"Evaluación Adaptativa - {', '.join(knowledge_areas)}",
            description="Evaluación personalizada basada en tu perfil de aprendizaje",
            questions=questions,
            total_questions=len(questions),
            estimated_duration=self._estimate_duration_from_questions(
                questions),
            created_at=datetime.now(),
            status="active"
        )

    async def _generate_questions_for_level(
        self,
        knowledge_area: str,
        difficulty: str,
        bloom_level: str,
        count: int
    ) -> List[AEPAssessmentQuestion]:
        """
        Genera preguntas para un nivel específico de Bloom.

        Args:
            knowledge_area: Área de conocimiento.
            difficulty: Nivel de dificultad.
            bloom_level: Nivel de Bloom.
            count: Número de preguntas.

        Returns:
            Lista de preguntas generadas.
        """
        questions = []

        # Verificar cache primero
        cache_key = f"{knowledge_area}_{difficulty}_{bloom_level}"
        if cache_key in self.question_cache:
            cached_questions = self.question_cache[cache_key]
            questions.extend(cached_questions[:count])
            count -= len(questions)

        if count > 0:
            # Generar nuevas preguntas usando AI
            prompt = self._build_question_generation_prompt(
                knowledge_area,
                difficulty,
                bloom_level,
                count
            )

            response = await self.openai_tool.generate_completion(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )

            new_questions = self._parse_generated_questions(response)
            questions.extend(new_questions)

            # Actualizar cache
            if cache_key not in self.question_cache:
                self.question_cache[cache_key] = []
            self.question_cache[cache_key].extend(new_questions)

        return questions[:count]

    async def _evaluate_student_response(
        self,
        question: AEPAssessmentQuestion,
        student_response: str
    ) -> Dict[str, Any]:
        """
        Evalúa la respuesta del estudiante usando AI.

        Args:
            question: La pregunta original.
            student_response: Respuesta del estudiante.

        Returns:
            Resultado de la evaluación.
        """
        prompt = f"""
        Evalúa la siguiente respuesta del estudiante:

        Pregunta: {question.question_text}
        Tipo de pregunta: {question.question_type}
        Nivel de Bloom: {question.bloom_level}
        Dificultad: {question.difficulty}

        Respuesta del estudiante: {student_response}

        Respuesta correcta esperada: {question.correct_answer}

        Por favor proporciona:
        1. Puntuación (0-100)
        2. Feedback detallado
        3. Puntos fuertes
        4. Áreas de mejora
        5. Sugerencias para estudiar más

        Formato JSON:
        {{
            "score": 85,
            "feedback": "Buen intento...",
            "strengths": ["..."],
            "improvements": ["..."],
            "study_suggestions": ["..."]
        }}
        """

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3
        )

        try:
            evaluation = json.loads(response)
            return evaluation
        except json.JSONDecodeError:
            # Fallback evaluation
            return {
                "score": 50,
                "feedback": "Respuesta evaluada. Revisa los conceptos básicos.",
                "strengths": ["Intento realizado"],
                "improvements": ["Profundizar en el tema"],
                "study_suggestions": ["Repasar material básico"]
            }

    async def _calculate_progress_metrics(
        self,
        assessments: List[AEPAssessment]
    ) -> Dict[str, Any]:
        """
        Calcula métricas de progreso del estudiante.

        Args:
            assessments: Lista de evaluaciones.

        Returns:
            Métricas calculadas.
        """
        total_assessments = len(assessments)
        completed_assessments = len(
            [a for a in assessments if a.status == "completed"])
        total_questions = sum(len(a.questions) for a in assessments)
        total_correct = sum(
            len([r for r in a.results if r.score >= 70]) for a in assessments)

        # Calcular promedio por área de conocimiento
        area_performance = {}
        for assessment in assessments:
            for result in assessment.results:
                question = next(
                    (q for q in assessment.questions if q.question_id ==
                     result.question_id),
                    None
                )
                if question:
                    area = question.knowledge_area
                    if area not in area_performance:
                        area_performance[area] = {"total": 0, "correct": 0}
                    area_performance[area]["total"] += 1
                    if result.score >= 70:
                        area_performance[area]["correct"] += 1

        # Calcular tendencias
        recent_scores = []
        for assessment in sorted(assessments, key=lambda x: x.created_at, reverse=True)[:5]:
            avg_score = sum(r.score for r in assessment.results) / \
                len(assessment.results) if assessment.results else 0
            recent_scores.append(avg_score)

        return {
            "total_assessments": total_assessments,
            "completed_assessments": completed_assessments,
            "completion_rate": completed_assessments / total_assessments if total_assessments > 0 else 0,
            "overall_accuracy": total_correct / total_questions if total_questions > 0 else 0,
            "area_performance": {
                area: {
                    "accuracy": data["correct"] / data["total"],
                    "questions_answered": data["total"]
                }
                for area, data in area_performance.items()
            },
            "recent_trend": self._calculate_trend(recent_scores),
            "strengths": [area for area, data in area_performance.items()
                          if data["correct"] / data["total"] >= 0.8],
            "weaknesses": [area for area, data in area_performance.items()
                           if data["correct"] / data["total"] < 0.6]
        }

    def _determine_initial_difficulty(
        self,
        knowledge_area: Optional[AEPKnowledgeArea],
        base_difficulty: str
    ) -> str:
        """
        Determina la dificultad inicial basada en el conocimiento del estudiante.

        Args:
            knowledge_area: Área de conocimiento del estudiante.
            base_difficulty: Dificultad base solicitada.

        Returns:
            Dificultad determinada.
        """
        if not knowledge_area:
            return base_difficulty

        proficiency = knowledge_area.proficiency_level

        if proficiency >= 0.8:
            return "advanced"
        elif proficiency >= 0.6:
            return "intermediate"
        elif proficiency >= 0.4:
            return "beginner"
        else:
            return "beginner"

    def _increase_difficulty(self, current: str) -> str:
        """Aumenta el nivel de dificultad."""
        levels = ["beginner", "intermediate", "advanced", "expert"]
        try:
            idx = levels.index(current)
            return levels[min(idx + 1, len(levels) - 1)]
        except ValueError:
            return current

    def _decrease_difficulty(self, current: str) -> str:
        """Disminuye el nivel de dificultad."""
        levels = ["beginner", "intermediate", "advanced", "expert"]
        try:
            idx = levels.index(current)
            return levels[max(idx - 1, 0)]
        except ValueError:
            return current

    def _calculate_progress(self, assessment: AEPAssessment) -> float:
        """Calcula el progreso de una evaluación."""
        if not assessment.questions:
            return 0.0
        answered = len(assessment.results)
        return answered / len(assessment.questions)

    def _estimate_duration(self, assessment: AEPAssessment) -> int:
        """Estima la duración de una evaluación en minutos."""
        return assessment.estimated_duration

    def _estimate_duration_from_questions(self, questions: List[AEPAssessmentQuestion]) -> int:
        """Estima duración basada en preguntas."""
        # Promedio: 2 minutos por pregunta
        return len(questions) * 2

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calcula la tendencia de puntuaciones."""
        if len(scores) < 2:
            return "insufficient_data"

        # Calcular pendiente simple
        slope = (scores[0] - scores[-1]) / len(scores)

        if slope > 5:
            return "improving"
        elif slope < -5:
            return "declining"
        else:
            return "stable"

    async def _get_student_profile(self, student_id: str) -> Optional[AEPStudentProfile]:
        """Obtiene el perfil del estudiante."""
        return await self.persistence_tool.get_student_profile(student_id)

    async def _get_performance_history(
        self,
        student_id: str,
        knowledge_area: str
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de rendimiento."""
        assessments = await self.persistence_tool.get_student_assessments(student_id)
        history = []

        for assessment in assessments:
            for result in assessment.results:
                question = next(
                    (q for q in assessment.questions if q.question_id ==
                     result.question_id),
                    None
                )
                if question and question.knowledge_area == knowledge_area:
                    history.append({
                        "score": result.score,
                        "difficulty": question.difficulty,
                        "bloom_level": question.bloom_level,
                        "date": result.evaluated_at
                    })

        return history

    def _calculate_optimal_difficulty(
        self,
        performance_history: List[Dict[str, Any]],
        current_difficulty: str
    ) -> str:
        """Calcula la dificultad óptima basada en el historial."""
        if not performance_history:
            return current_difficulty

        recent_scores = [h["score"] for h in performance_history[-5:]]
        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score >= 80:
            return self._increase_difficulty(current_difficulty)
        elif avg_score <= 60:
            return self._decrease_difficulty(current_difficulty)
        else:
            return current_difficulty

    def _explain_adaptation(self, performance_history: List[Dict[str, Any]]) -> str:
        """Explica por qué se adaptó la dificultad."""
        if not performance_history:
            return "Primera evaluación - dificultad base"

        recent_scores = [h["score"] for h in performance_history[-3:]]
        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score >= 80:
            return "Buen rendimiento reciente - aumentando dificultad"
        elif avg_score <= 60:
            return "Dificultad alta - ajustando a nivel inferior"
        else:
            return "Rendimiento consistente - manteniendo dificultad"

    async def _generate_question(
        self,
        knowledge_area: str,
        difficulty: str,
        bloom_level: str
    ) -> AEPAssessmentQuestion:
        """Genera una pregunta individual."""
        prompt = self._build_question_generation_prompt(
            knowledge_area,
            difficulty,
            bloom_level,
            1
        )

        response = await self.openai_tool.generate_completion(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7
        )

        questions = self._parse_generated_questions(response)
        return questions[0] if questions else self._create_fallback_question(knowledge_area)

    def _build_question_generation_prompt(
        self,
        knowledge_area: str,
        difficulty: str,
        bloom_level: str,
        count: int
    ) -> str:
        """Construye el prompt para generación de preguntas."""
        return f"""
        Genera {count} pregunta(s) de evaluación para el área: {knowledge_area}
        Nivel de dificultad: {difficulty}
        Nivel de Bloom: {bloom_level}

        Formato requerido para cada pregunta (JSON):
        {{
            "question_id": "unique_id",
            "question_text": "Texto de la pregunta",
            "question_type": "multiple_choice|true_false|short_answer",
            "options": ["A) Opción 1", "B) Opción 2", "C) Opción 3", "D) Opción 4"] (solo para multiple_choice),
            "correct_answer": "Respuesta correcta",
            "explanation": "Explicación de por qué es correcta",
            "knowledge_area": "{knowledge_area}",
            "difficulty": "{difficulty}",
            "bloom_level": "{bloom_level}",
            "points": 10
        }}

        Asegúrate de que las preguntas sean apropiadas para el nivel de Bloom especificado:
        - remember: recordar hechos
        - understand: explicar conceptos
        - apply: usar en situaciones nuevas
        - analyze: descomponer y analizar
        - evaluate: juzgar y criticar
        - create: crear algo nuevo

        Genera preguntas de alta calidad educativa.
        """

    def _parse_generated_questions(self, response: str) -> List[AEPAssessmentQuestion]:
        """Parsea preguntas generadas por AI."""
        try:
            # Intentar parsear como JSON array
            data = json.loads(response)
            if isinstance(data, list):
                return [AEPAssessmentQuestion(**q) for q in data]

            # Intentar parsear como objeto único
            return [AEPAssessmentQuestion(**data)]

        except (json.JSONDecodeError, TypeError):
            # Fallback: crear pregunta básica
            return [self._create_fallback_question("general")]

    def _create_fallback_question(self, knowledge_area: str) -> AEPAssessmentQuestion:
        """Crea una pregunta de fallback."""
        return AEPAssessmentQuestion(
            question_id=f"fallback_{datetime.now().strftime('%H%M%S')}",
            question_text=f"¿Cuál es un concepto importante en {knowledge_area}?",
            question_type="short_answer",
            correct_answer="Depende del contexto específico",
            explanation="Pregunta general para evaluación básica",
            knowledge_area=knowledge_area,
            difficulty="beginner",
            bloom_level="remember",
            points=5
        )

    async def _generate_basic_questions(self, topics: List[str]) -> List[AEPAssessmentQuestion]:
        """Genera preguntas básicas para los temas dados."""
        questions = []

        for topic in topics[:3]:  # Limitar a 3 temas
            # Crear 2 preguntas por tema
            for i in range(2):
                question = AEPAssessmentQuestion(
                    question_id=f"q_{topic}_{i}_{datetime.now().strftime('%H%M%S')}",
                    question_text=f"¿Qué sabes sobre {topic}? Proporciona una explicación breve.",
                    question_type="short_answer",
                    correct_answer="Respuesta evaluada por el instructor",
                    explanation="Pregunta abierta para evaluación cualitativa",
                    knowledge_area=topic,
                    difficulty="intermediate",
                    bloom_level="understand",
                    points=10
                )
                questions.append(question)

        return questions
