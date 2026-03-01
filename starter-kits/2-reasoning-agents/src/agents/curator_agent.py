"""
Curator Agent — Busca y gestiona itinerarios de aprendizaje en Microsoft Learn.

Este agente es el primero en la secuencia del workflow. Su responsabilidad
es encontrar itinerarios de aprendizaje relevantes basados en los temas de interés
del estudiante y la certificación objetivo.
"""

from __future__ import annotations

import json
import uuid
from urllib.parse import quote_plus
from typing import Any, Dict, List

import httpx

from .base_agent import AEPAgent
from src.tools.certifications import certifications_tool
from src.tools.mslearn_mcp import mslearn_mcp_tool
from src.models.schemas import (
    AEPLearningModule,
    AEPLearningPath,
    AEPWorkflowContext,
)


class CuratorAgent(AEPAgent):
    """
    Agente de gestión de itinerarios de aprendizaje.

    Busca y filtra itinerarios de aprendizaje relevantes en Microsoft Learn,
    considerando los temas de interés del estudiante y la certificación
    objetivo. Presenta opciones con puntuaciones de relevancia.
    """

    def __init__(self) -> None:
        """Inicializa el Curator Agent."""
        super().__init__(
            name="CuratorAgent",
            description=(
                "Especialista en gestionar itinerarios de aprendizaje relevantes "
                "en Microsoft Learn para estudiantes de certificaciones."
            ),
            capabilities=[
                "Buscar itinerarios de aprendizaje en Microsoft Learn",
                "Filtrar por relevancia a certificación objetivo",
                "Evaluar dificultad y duración",
                "Presentar itinerarios con recomendaciones",
            ],
            max_tokens=1024,
            temperature=0.3,  # Bajo para respuestas consistentes
        )

    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ejecuta la gestión de itinerarios de aprendizaje.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales (opcionales).

        Returns:
            Diccionario con itinerarios gestionados y recomendaciones.
        """
        self.log_reasoning(
            "INICIO",
            "Iniciando gestión de itinerarios de aprendizaje",
            f"Estudiante: {context.student.name}, Certificación: {context.student.target_certification}",
        )

        # 1. Obtener información de la certificación objetivo (consulta en tiempo real)
        cert_info = await self._get_certification_info(
            context.student.target_certification
        )

        # 2. Generar búsqueda inteligente de rutas
        search_query = self._build_search_query(
            context.student.topics_of_interest,
            cert_info,
        )

        # 3. Buscar rutas (simulado por ahora, luego MCP)
        learning_paths = await self._search_learning_paths(
            search_query,
            context.student.level,
            certification=context.student.target_certification,
        )

        # 4. Filtrar y puntuar por relevancia
        curated_paths = self._filter_and_score_paths(
            learning_paths,
            context.student.topics_of_interest,
            cert_info,
        )

        # 5. Preparar recomendaciones
        recommendations = self._generate_recommendations(
            curated_paths,
            context.student,
        )

        result = {
            "learning_paths": curated_paths,
            "recommendations": recommendations,
            "search_query": search_query,
            "total_paths_found": len(learning_paths),
            "paths_curated": len(curated_paths),
        }

        self.log_reasoning(
            "COMPLETADO",
            f"Curadas {len(curated_paths)} rutas de {len(learning_paths)} encontradas",
            f"Recomendación principal: {recommendations[0] if recommendations else 'Ninguna'}",
        )

        return result

    async def _get_certification_info(self, cert_id: str) -> Dict[str, Any]:
        """
        Obtiene información de la certificación consultando la API oficial
        de Microsoft Learn en tiempo real.

        Args:
            cert_id: ID de la certificación (e.g. "AZ-900").

        Returns:
            Información de la certificación o valores por defecto.
        """
        self.log_reasoning(
            "CERT_LOOKUP",
            f"Consultando certificación '{cert_id}' en Microsoft Learn",
        )

        cert = await certifications_tool.get_certification_by_id(cert_id)

        if cert:
            return cert

        # Certificación no encontrada, usar valores genéricos
        self.logger.warning(
            f"Certificación {cert_id} no encontrada en el catálogo oficial"
        )
        return {
            "cert_id": cert_id,
            "name": f"Certificación {cert_id}",
            "skills_measured": ["Conceptos generales"],
            "level": "intermediate",
            "estimated_study_hours": 40.0,
        }

    def _build_search_query(
        self,
        topics: List[str],
        cert_info: Dict[str, Any],
    ) -> str:
        """
        Construye una consulta compacta y útil para la herramienta MCP.

        La API de Microsoft Learn no se beneficia de palabras de relleno como
        "Microsoft Learn learning path:"; en cambio, basta con un listado de
        términos clave (temas de interés + certificado). Este método también
        incluye el código de certificación si está presente para filtrar más
        eficientemente.

        Args:
            topics: Temas de interés del estudiante.
            cert_info: Información de la certificación, puede incluir
                "cert_id" y "skills_measured".

        Returns:
            Cadena de búsqueda optimizada para pasar al MCP tool.
        """
        terms: list[str] = []
        # tomar hasta 5 temas únicos
        for t in topics[:5]:
            if t and t.lower() not in (x.lower() for x in terms):
                terms.append(t)

        # agregar habilidades de la certificación si existen
        for s in cert_info.get("skills_measured", [])[:5]:
            if s and s.lower() not in (x.lower() for x in terms):
                terms.append(s)

        # añadir el código de certificación explícitamente (AZ-900, MB-800, etc.)
        cert_id = cert_info.get("cert_id") or cert_info.get("certification")
        if cert_id and cert_id not in terms:
            terms.append(cert_id)

        search_query = " ".join(terms).strip()
        if not search_query:
            search_query = "Microsoft Learn"

        self.log_reasoning(
            "SEARCH_QUERY",
            f"Construyendo consulta con {len(terms)} términos",
            f"Query: {search_query}",
        )

        return search_query

    async def _search_learning_paths(
        self,
        search_query: str,
        student_level: str,
        certification: str | None = None,
    ) -> List[AEPLearningPath]:
        """
        Genera rutas de aprendizaje usando Azure OpenAI.

        Usa el modelo para crear rutas de aprendizaje estructuradas y relevantes
        basadas en la consulta de búsqueda. En futuro podrá integrarse con
        Microsoft Learn MCP Server para datos en tiempo real.

        Args:
            search_query: Consulta de búsqueda.
            student_level: Nivel del estudiante.

        Returns:
            Lista de rutas de aprendizaje generadas.
        """
        self.log_reasoning(
            "SEARCH_LEARNING_PATHS",
            f"Generando rutas de aprendizaje para nivel '{student_level}'",
            f"Query: {search_query}",
        )

        # Primer intento: obtener rutas reales mediante la herramienta MCP
        def _normalize_url(url: str) -> str:
            if not url:
                return url
            if url.startswith("/"):
                return f"https://learn.microsoft.com{url}"
            if url.startswith("http"):
                return url
            return f"https://learn.microsoft.com/{url.lstrip(' /')}"

        try:
            raw_paths = await mslearn_mcp_tool.search_learning_paths(
                query=search_query,
                certification=certification,
                max_results=5,
            )
            if raw_paths:
                learning_paths: List[AEPLearningPath] = []
                for item in raw_paths:
                    modules: List[AEPLearningModule] = []
                    try:
                        details = await mslearn_mcp_tool.get_learning_path_details(
                            item.get("url", ""),
                        )
                        for child in details.get("children", []) if details else []:
                            modules.append(
                                AEPLearningModule(
                                    title=child.get("title", ""),
                                    url=_normalize_url(child.get("url", "")),
                                    duration_minutes=int(
                                        child.get("duration_minutes", 0) or 0
                                    ),
                                    description=child.get("summary", ""),
                                    skills_covered=[],
                                )
                            )
                    except Exception:
                        pass

                    learning_paths.append(
                        AEPLearningPath(
                            path_id=item.get(
                                "uid") or f"lp-{uuid.uuid4().hex[:8]}",
                            title=item.get("title", ""),
                            description=item.get("summary", ""),
                            modules=modules,
                            estimated_hours=float(
                                item.get("number_of_children", 0) or 0
                            ),
                            relevance_score=0.0,
                        )
                    )
                if learning_paths:
                    learning_paths = await self._validate_learning_paths_urls(
                        learning_paths,
                        search_query,
                    )
                    self.log_reasoning(
                        "SEARCH_COMPLETE",
                        f"Encontradas {len(learning_paths)} rutas reales",
                        "Via Microsoft Learn API",
                    )
                    return learning_paths
        except Exception as e:
            self.logger.warning(
                f"Fallo en mslearn_mcp_tool: {e}"
            )

        # si no hubo resultados reales, seguimos con el LLM de respaldo
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un experto en certificaciones Microsoft y rutas de aprendizaje de Microsoft Learn. "
                    "Usa toda tu capacidad de análisis para priorizar precisión, cobertura temática y relevancia real para el perfil recibido. "
                    "Genera rutas de aprendizaje detalladas y relevantes en formato JSON. "
                    "<<IMPORTANTE>> los valores de \"url\" deben ser vínculos válidos que apunten a "
                    "https://learn.microsoft.com (idealmente dentro de /training/paths/...). No inventes dominios, subdominios ni extensiones inexistentes. "
                    "Si no conoces la URL exacta, incluye al menos un enlace real a la página de búsqueda de Learn correspondiente. "
                    "Responde SOLO con JSON válido, sin texto adicional."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Genera 3 rutas de aprendizaje de Microsoft Learn para:\n"
                    f"- Búsqueda: '{search_query}'\n"
                    f"- Nivel del estudiante: {student_level}\n\n"
                    "Devuelve un array JSON con este formato exacto:\n"
                    "[\n"
                    "  {\n"
                    '    "path_id": "lp-descriptive-id",\n'
                    '    "title": "Nombre de la Ruta",\n'
                    '    "description": "Descripción detallada de la ruta",\n'
                    '    "estimated_hours": 8.0,\n'
                    '    "relevance_score": 0.90,\n'
                    '    "modules": [\n'
                    "      {\n"
                    '        "title": "Nombre del Módulo",\n'
                    '        "url": "https://learn.microsoft.com/training/paths/...",\n'
                    '        "duration_minutes": 60,\n'
                    '        "description": "Descripción del módulo",\n'
                    '        "skills_covered": ["skill1", "skill2"]\n'
                    "      }\n"
                    "    ]\n"
                    "  }\n"
                    "]"
                ),
            },
        ]

        try:
            raw = await self._call_azure_openai(
                messages, temperature=0.3, max_tokens=2000
            )

            # Limpiar y parsear JSON
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1] if len(parts) > 1 else cleaned
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            paths_data: List[Dict[str, Any]] = json.loads(cleaned)

            learning_paths: List[AEPLearningPath] = []
            for p in paths_data:
                modules = [
                    AEPLearningModule(
                        title=m.get("title", ""),
                        url=m.get(
                            "url", "https://learn.microsoft.com/training/"),
                        duration_minutes=int(m.get("duration_minutes", 60)),
                        description=m.get("description", ""),
                        skills_covered=m.get("skills_covered", []),
                    )
                    for m in p.get("modules", [])
                ]
                learning_paths.append(
                    AEPLearningPath(
                        path_id=p.get(
                            "path_id") or f"lp-{uuid.uuid4().hex[:8]}",
                        title=p.get("title", ""),
                        description=p.get("description", ""),
                        modules=modules,
                        estimated_hours=float(p.get("estimated_hours", 8.0)),
                        relevance_score=float(p.get("relevance_score", 0.8)),
                    )
                )

            learning_paths = await self._validate_learning_paths_urls(
                learning_paths,
                search_query,
            )

            self.log_reasoning(
                "SEARCH_COMPLETE",
                f"Generadas {len(learning_paths)} rutas de aprendizaje",
                "Via Azure OpenAI",
            )
            return learning_paths

        except Exception as e:
            self.logger.error(f"Error generando rutas de aprendizaje: {e}")
            return [
                AEPLearningPath(
                    path_id=f"lp-fallback-{uuid.uuid4().hex[:8]}",
                    title=f"Ruta para {search_query[:50]}",
                    description=f"Ruta generada para: {search_query}",
                    modules=[
                        AEPLearningModule(
                            title="Buscar módulos en Microsoft Learn",
                            url=self._build_learn_search_url(search_query),
                            duration_minutes=30,
                            description=(
                                "Enlace de búsqueda oficial para encontrar "
                                "módulos válidos."
                            ),
                            skills_covered=[],
                        )
                    ],
                    estimated_hours=8.0,
                    relevance_score=0.7,
                )
            ]

    async def _validate_learning_paths_urls(
        self,
        learning_paths: List[AEPLearningPath],
        search_query: str,
    ) -> List[AEPLearningPath]:
        """
        Valida URLs de módulos y reemplaza enlaces inválidos.

        Args:
            learning_paths: Rutas candidatas.
            search_query: Consulta de búsqueda original.

        Returns:
            Lista de rutas con URLs verificadas.
        """
        if not learning_paths:
            return learning_paths

        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
        ) as client:
            for path in learning_paths:
                for module in path.modules:
                    module.url = await self._ensure_valid_learn_url(
                        module.url,
                        search_query,
                        client,
                        module.title,
                    )

        return learning_paths

    async def _ensure_valid_learn_url(
        self,
        url: str,
        search_query: str,
        client: httpx.AsyncClient,
        module_title: str = "",
    ) -> str:
        """
        Devuelve una URL válida de Learn o una búsqueda oficial de fallback.

        Args:
            url: URL original.
            search_query: Consulta principal.
            client: Cliente HTTP reutilizable.
            module_title: Título del módulo para búsqueda específica.

        Returns:
            URL verificada o URL de búsqueda en Learn.
        """
        normalized = (url or "").strip()
        if not normalized:
            return self._build_learn_search_url(module_title or search_query)

        if not normalized.startswith("https://learn.microsoft.com"):
            return self._build_learn_search_url(module_title or search_query)

        is_valid = await self._is_url_reachable(normalized, client)
        if is_valid:
            return normalized

        en_us_variant = normalized.replace("/es-es/", "/en-us/")
        if en_us_variant != normalized:
            en_us_valid = await self._is_url_reachable(en_us_variant, client)
            if en_us_valid:
                return en_us_variant

        return self._build_learn_search_url(module_title or search_query)

    async def _is_url_reachable(
        self,
        url: str,
        client: httpx.AsyncClient,
    ) -> bool:
        """
        Comprueba si una URL responde con estado válido.

        Args:
            url: URL a comprobar.
            client: Cliente HTTP reutilizable.

        Returns:
            True si responde 2xx/3xx; en otro caso False.
        """
        try:
            response = await client.head(url)
            status = response.status_code
            if 200 <= status < 400:
                return True
            if status in (403, 405):
                get_response = await client.get(url)
                get_status = get_response.status_code
                return 200 <= get_status < 400
            return False
        except httpx.HTTPError:
            return False

    def _build_learn_search_url(self, query: str) -> str:
        """
        Construye URL de búsqueda oficial de Microsoft Learn.

        Args:
            query: Términos de búsqueda.

        Returns:
            URL de búsqueda en Learn.
        """
        q = quote_plus((query or "microsoft learn").strip())
        return f"https://learn.microsoft.com/en-us/training/browse/?terms={q}"

    def _filter_and_score_paths(
        self,
        paths: List[AEPLearningPath],
        student_topics: List[str],
        cert_info: Dict[str, Any],
    ) -> List[AEPLearningPath]:
        """
        Filtra y puntúa rutas por relevancia.

        Args:
            paths: Rutas encontradas.
            student_topics: Temas de interés del estudiante.
            cert_info: Información de la certificación.

        Returns:
            Rutas filtradas y ordenadas por relevancia.
        """
        scored_paths = []

        for path in paths:
            # Calcular relevancia basada en temas
            relevance = self._calculate_relevance(
                path, student_topics, cert_info
            )
            path.relevance_score = relevance

            # Solo incluir rutas con relevancia > 0.2 (ajustado para testing)
            if relevance > 0.2:
                scored_paths.append(path)

        # Ordenar por relevancia descendente
        scored_paths.sort(key=lambda p: p.relevance_score, reverse=True)

        self.log_reasoning(
            "FILTERING",
            f"Filtradas {len(scored_paths)} rutas de {len(paths)}",
            f"Mejor puntuación: {scored_paths[0].relevance_score if scored_paths else 0:.2f}",
        )

        return scored_paths

    def _calculate_relevance(
        self,
        path: AEPLearningPath,
        student_topics: List[str],
        cert_info: Dict[str, Any],
    ) -> float:
        """
        Calcula la relevancia de una ruta de aprendizaje.

        Args:
            path: Ruta a evaluar.
            student_topics: Temas del estudiante.
            cert_info: Información de la certificación.

        Returns:
            Puntuación de relevancia (0-1).
        """
        relevance = 0.0
        cert_skills = cert_info.get("skills_measured", [])

        # Buscar coincidencias en título y descripción
        text_to_check = (
            path.title.lower() + " " + path.description.lower()
        )

        # Contar coincidencias con temas del estudiante
        student_matches = sum(
            1 for topic in student_topics
            if topic.lower() in text_to_check
        )

        # Contar coincidencias con skills de certificación
        cert_matches = sum(
            1 for skill in cert_skills
            if skill.lower() in text_to_check
        )

        # Calcular puntuación
        total_possible = len(student_topics) + len(cert_skills)
        if total_possible > 0:
            relevance = (student_matches + cert_matches) / total_possible

        # Bonus por módulos relevantes
        module_bonus = min(len(path.modules) * 0.05, 0.2)
        relevance += module_bonus

        return min(relevance, 1.0)  # Máximo 1.0

    def _generate_recommendations(
        self,
        curated_paths: List[AEPLearningPath],
        student: Any,  # AEPStudentProfile
    ) -> List[str]:
        """
        Genera recomendaciones basadas en las rutas curadas.

        Args:
            curated_paths: Rutas curadas.
            student: Perfil del estudiante.

        Returns:
            Lista de recomendaciones.
        """
        if not curated_paths:
            return ["No se encontraron rutas relevantes. Considera ajustar tus temas de interés."]

        recommendations = []
        top_path = curated_paths[0]

        recommendations.append(
            f"Recomendación principal: '{top_path.title}' "
            f"(Relevancia: {top_path.relevance_score:.1%})"
        )

        if len(curated_paths) > 1:
            recommendations.append(
                f"Alternativa: '{curated_paths[1].title}' "
                f"(Relevancia: {curated_paths[1].relevance_score:.1%})"
            )

        # Recomendaciones basadas en nivel del estudiante
        if student.level == "beginner" and top_path.estimated_hours > 10:
            recommendations.append(
                "Como principiante, considera dividir esta ruta en sesiones más cortas."
            )

        return recommendations
