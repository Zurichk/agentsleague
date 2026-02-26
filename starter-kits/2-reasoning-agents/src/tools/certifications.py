"""
Herramienta para consultar certificaciones Microsoft en tiempo real.

Obtiene el catálogo oficial desde la API pública de Microsoft Learn
y lo normaliza al formato interno del sistema.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx

from src.utils.logger import setup_logger

logger = setup_logger("tools.certifications")

# URL pública de la API de catálogo de Microsoft Learn
_CATALOG_API_URL = (
    "https://learn.microsoft.com/api/catalog/"
    "?type=certifications&locale=en-us"
)

# Mapeo de niveles de la API al formato interno
_LEVEL_MAP: Dict[str, str] = {
    "beginner": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
}


class CertificationsTool:
    """
    Consulta el catálogo oficial de certificaciones Microsoft desde
    la API pública de Microsoft Learn.

    Incluye caché en memoria con TTL configurable para evitar
    llamadas repetidas a la API durante una misma sesión.
    """

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        """
        Inicializa la herramienta.

        Args:
            cache_ttl_seconds: Tiempo de vida del caché en segundos (1h por defecto).
        """
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl = cache_ttl_seconds
        logger.info("🏅 CertificationsTool inicializado")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def fetch_all_certifications(self) -> List[Dict[str, Any]]:
        """
        Devuelve todas las certificaciones Microsoft disponibles.

        Usa caché para no llamar a la API en cada invocación.

        Returns:
            Lista de certificaciones normalizadas al formato interno.
        """
        if self._is_cache_valid():
            logger.debug("📦 Usando caché de certificaciones")
            return self._cache  # type: ignore[return-value]

        logger.info(
            "🌐 Consultando catálogo de certificaciones en Microsoft Learn...")

        raw = await self._fetch_from_api()
        normalized = [self._normalize(c) for c in raw]

        self._cache = normalized
        self._cache_timestamp = time.monotonic()

        logger.info(f"✅ Catálogo cargado: {len(normalized)} certificaciones")
        return normalized

    async def get_certification_by_id(
        self, cert_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Busca una certificación por su código de examen (p.ej. "AZ-900").

        Args:
            cert_id: Código del examen.

        Returns:
            Información de la certificación o None si no se encuentra.
        """
        all_certs = await self.fetch_all_certifications()
        cert_id_upper = cert_id.upper()

        for cert in all_certs:
            if cert["cert_id"] == cert_id_upper:
                return cert

        logger.warning(
            f"⚠️ Certificación '{cert_id}' no encontrada en el catálogo oficial")
        return None

    async def search_certifications(
        self,
        query: str = "",
        level: Optional[str] = None,
        product: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filtra certificaciones por texto, nivel o producto.

        Args:
            query: Texto libre a buscar en nombre/descripción.
            level: Nivel deseado ('beginner', 'intermediate', 'advanced').
            product: Producto Microsoft (ej. 'azure', 'microsoft-365').

        Returns:
            Lista de certificaciones que cumplen los filtros.
        """
        all_certs = await self.fetch_all_certifications()

        results = all_certs

        if query:
            q = query.lower()
            results = [
                c for c in results
                if q in c["name"].lower() or q in c["description"].lower()
                or any(q in s.lower() for s in c.get("skills_measured", []))
            ]

        if level:
            results = [c for c in results if c.get("level") == level]

        if product:
            p = product.lower()
            results = [
                c for c in results
                if any(p in pr.lower() for pr in c.get("products", []))
            ]

        return results

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        if self._cache is None:
            return False
        return (time.monotonic() - self._cache_timestamp) < self._cache_ttl

    async def _fetch_from_api(self) -> List[Dict[str, Any]]:
        """Realiza la llamada HTTP a la API de Microsoft Learn."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(_CATALOG_API_URL)
                response.raise_for_status()
                data = response.json()
                return data.get("certifications", [])
        except httpx.HTTPError as exc:
            logger.error(f"❌ Error al consultar API de Microsoft Learn: {exc}")
            return []
        except Exception as exc:
            logger.error(
                f"❌ Error inesperado al obtener certificaciones: {exc}")
            return []

    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza una entrada de la API al formato interno del sistema.

        La API de Microsoft Learn devuelve entradas con campos como:
          uid, title, summary, levels, roles, products, exams, url
        """
        # Extraer código de examen (ej. "AZ-900") desde la lista de exámenes
        exam_codes: List[str] = []
        for exam in raw.get("exams", []):
            code = exam.get("display_name") or exam.get("uid", "")
            if code:
                exam_codes.append(code.upper())

        cert_id = exam_codes[0] if exam_codes else raw.get(
            "uid", "UNKNOWN").upper()

        # Niveles
        raw_levels: List[str] = raw.get("levels", [])
        level = _LEVEL_MAP.get(raw_levels[0].lower(
        ), "intermediate") if raw_levels else "intermediate"

        # Habilidades medidas — la API expone "study_guide" u otras secciones;
        # usamos roles + products como proxy razonable cuando no hay lista explícita
        roles: List[str] = raw.get("roles", [])
        products: List[str] = raw.get("products", [])
        skills = roles + [p.replace("-", " ").title() for p in products]

        return {
            "cert_id": cert_id,
            "name": raw.get("title", cert_id),
            "description": raw.get("summary", ""),
            "level": level,
            "exam_url": raw.get("url", ""),
            "skills_measured": skills,
            "products": products,
            "roles": roles,
            "locale": raw.get("locale", "en-us"),
        }


# Instancia global
certifications_tool = CertificationsTool()
