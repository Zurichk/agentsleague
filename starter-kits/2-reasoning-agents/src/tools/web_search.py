"""
Herramienta de búsqueda en Microsoft Learn para AEP CertMaster.

Implementa búsqueda real en learn.microsoft.com mediante sus APIs públicas:
  - /api/search  → búsqueda por keywords (sin autenticación)
  - /api/catalog → recupera detalles de certificaciones por código de examen
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

from src.utils.logger import setup_logger

logger = setup_logger("tools.web_search")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
AEP_MSLEARN_SEARCH_URL: str = "https://learn.microsoft.com/api/search"
AEP_MSLEARN_CATALOG_URL: str = "https://learn.microsoft.com/api/catalog/"
AEP_MSLEARN_TIMEOUT_SECONDS: int = 15
AEP_MSLEARN_DEFAULT_LOCALE: str = "es-ES"

# Patrón para detectar códigos de examen tipo "AZ-900", "MB-800", "SC-300"
_EXAM_CODE_RE = re.compile(r"\b([A-Z]{2,3}-\d{3,4}[A-Z0-9]?)\b")


class WebSearchTool:
    """
    Herramienta de búsqueda en Microsoft Learn.

    Utiliza las APIs públicas de learn.microsoft.com; no requiere
    ninguna clave de API ni configuración adicional.

    Métodos públicos
    ----------------
    search_certification_resources(certification, query, max_results)
        Busca recursos para preparar una certificación.
    search_learning_materials(topic, certification, max_results)
        Busca módulos y rutas de aprendizaje sobre un tema.
    search_community_discussions(topic, certification, max_results)
        Busca documentación técnica relacionada con un tema.
    get_resource_quality_score(url)
        Puntuación heurística de calidad según el dominio.
    """

    def __init__(self) -> None:
        """Inicializa la herramienta de búsqueda en Microsoft Learn."""
        logger.info(
            "🔍 WebSearchTool inicializado "
            "(Microsoft Learn API — sin clave requerida)"
        )

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    async def search_certification_resources(
        self,
        certification: str,
        query: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca recursos de estudio para una certificación.

        Primero intenta recuperar información oficial del catálogo MS Learn
        si ``certification`` contiene un código de examen reconocible
        (ej: "AZ-900"). Complementa con resultados de búsqueda por keyword.

        Args:
            certification: Código o nombre de la certificación.
            query: Términos adicionales opcionales.
            max_results: Número máximo de resultados.

        Returns:
            Lista de dicts con title, url, snippet, source y type.
        """
        results: List[Dict[str, Any]] = []

        # Búsqueda en el catálogo si hay código de examen explícito
        exam_match = _EXAM_CODE_RE.search(certification.upper())
        if exam_match:
            catalog_items = await self._search_catalog(
                exam_code=exam_match.group(1),
                max_results=max_results,
            )
            results.extend(catalog_items)

        # Búsqueda por keyword complementaria
        search_query = (
            f"{certification} certification study resources Microsoft Learn"
        )
        if query:
            search_query += f" {query}"

        remaining = max(1, max_results - len(results))
        keyword_items = await self._mslearn_search(search_query, remaining)
        results.extend(keyword_items)

        logger.info(
            f"✅ search_certification_resources: "
            f"{len(results)} resultado(s) para '{certification}'"
        )
        return results[:max_results]

    async def search_learning_materials(
        self,
        topic: str,
        certification: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca módulos y rutas de aprendizaje sobre un tema en MS Learn.

        Args:
            topic: Tema o habilidad a buscar.
            certification: Código de certificación relacionado (opcional).
            max_results: Número máximo de resultados.

        Returns:
            Lista de dicts con title, url, snippet, source y type.
        """
        search_query = f"{topic} learn module tutorial"
        if certification:
            search_query += f" {certification}"

        logger.info(f"🔍 Buscando materiales para: '{search_query[:80]}'")
        return await self._mslearn_search(search_query, max_results)

    async def search_community_discussions(
        self,
        topic: str,
        certification: Optional[str] = None,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Busca documentación técnica relacionada con un tema en MS Learn.

        Args:
            topic: Tema a buscar.
            certification: Certificación relacionada (opcional).
            max_results: Número máximo de resultados.

        Returns:
            Lista de dicts con title, url, snippet, source y type.
        """
        search_query = f"{topic} documentation guide"
        if certification:
            search_query += f" {certification} exam"

        logger.info(
            f"🔍 Buscando documentación para: '{search_query[:80]}'"
        )
        return await self._mslearn_search(search_query, max_results)

    async def get_resource_quality_score(self, url: str) -> float:
        """
        Evalúa la calidad heurística de un recurso según su dominio.

        Args:
            url: URL del recurso a evaluar.

        Returns:
            Puntuación de calidad entre 0.0 y 1.0.
        """
        lower_url = url.lower()
        if "learn.microsoft.com" in lower_url:
            return 0.95
        elif "microsoft.com" in lower_url or "docs.microsoft.com" in lower_url:
            return 0.90
        elif "github.com" in lower_url:
            return 0.80
        elif (
            "stackoverflow.com" in lower_url
            or "techcommunity.microsoft.com" in lower_url
        ):
            return 0.75
        else:
            return 0.60

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    async def _mslearn_search(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda en learn.microsoft.com/api/search.

        Args:
            query: Texto de búsqueda.
            max_results: Número máximo de resultados.

        Returns:
            Lista de resultados con campos normalizados.
        """
        params = {
            "search": query,
            "locale": AEP_MSLEARN_DEFAULT_LOCALE,
            "$top": max_results,
            "expandScope": "true",
        }
        try:
            async with httpx.AsyncClient(
                timeout=AEP_MSLEARN_TIMEOUT_SECONDS
            ) as client:
                resp = await client.get(
                    AEP_MSLEARN_SEARCH_URL, params=params
                )
                resp.raise_for_status()
                data = resp.json()

            items = data.get("results", [])
            results: List[Dict[str, Any]] = [
                {
                    "title": item.get("title", "Sin título"),
                    "url": item.get("url", ""),
                    "snippet": item.get("summary", ""),
                    "source": "learn.microsoft.com",
                    "type": item.get("@type", "web"),
                }
                for item in items
                if item.get("url")
            ]
            logger.debug(
                f"🔍 MS Learn Search: {len(results)} resultado(s) "
                f"para '{query[:60]}'"
            )
            return results[:max_results]

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"❌ MS Learn Search HTTP {exc.response.status_code}: {exc}"
            )
            return []
        except httpx.RequestError as exc:
            logger.error(f"❌ MS Learn Search error de red: {exc}")
            return []

    async def _search_catalog(
        self, exam_code: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Consulta el catálogo de MS Learn filtrando por código de examen.

        Llama a /api/catalog/?locale=es-ES&type=certifications,courses
        y busca coincidencias donde el exam_code aparezca en el campo
        ``exams`` o en el ``uid`` del elemento.

        Args:
            exam_code: Código de examen, ej: "MB-800".
            max_results: Número máximo de entradas a devolver.

        Returns:
            Lista de dicts normalizados de certificaciones/cursos.
        """
        params = {
            "locale": AEP_MSLEARN_DEFAULT_LOCALE,
            "type": "certifications,courses",
        }
        exam_lower = exam_code.lower().replace("-", "")

        try:
            async with httpx.AsyncClient(
                timeout=AEP_MSLEARN_TIMEOUT_SECONDS
            ) as client:
                resp = await client.get(
                    AEP_MSLEARN_CATALOG_URL, params=params
                )
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"❌ MS Learn Catalog HTTP {exc.response.status_code}: {exc}"
            )
            return []
        except httpx.RequestError as exc:
            logger.error(f"❌ MS Learn Catalog error de red: {exc}")
            return []

        results: List[Dict[str, Any]] = []

        for collection in ("certifications", "courses"):
            for item in data.get(collection, []):
                uid: str = item.get("uid", "").lower()
                exams: List[str] = [
                    e.lower().replace("-", "").replace("exam.", "")
                    for e in item.get("exams", [])
                ]
                # Coincidencia si el código aparece en uid o en la lista exams
                if exam_lower in uid or exam_lower in exams:
                    raw_subtitle: str = item.get("subtitle", "") or ""
                    clean_snippet = re.sub(
                        r"<[^>]+>", " ", raw_subtitle
                    ).strip()
                    clean_snippet = re.sub(r"\s+", " ", clean_snippet)[:300]

                    results.append({
                        "title": item.get("title", "Sin título"),
                        "url": item.get("url", ""),
                        "snippet": clean_snippet,
                        "source": "learn.microsoft.com/catalog",
                        "type": item.get("type", collection.rstrip("s")),
                    })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        logger.debug(
            f"📋 Catálogo MS Learn: {len(results)} entrada(s) "
            f"para código '{exam_code}'"
        )
        return results[:max_results]


# Instancia global
web_search_tool = WebSearchTool()
