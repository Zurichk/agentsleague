"""
Herramienta de integración con Microsoft Learn (solo API real).

Consulta la API pública del catálogo de Microsoft Learn para obtener rutas
y módulos en tiempo real.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from src.utils.logger import setup_logger

logger = setup_logger("tools.mslearn_mcp")

_CATALOG_BASE = "https://learn.microsoft.com/api/catalog/"
_CACHE_TTL = 1800  # 30 minutos
_CATALOG_TYPE_MAP: Dict[str, str] = {
    "learningpaths": "learningPaths",
    "modules": "modules",
}
_CATALOG_RESPONSE_KEYS: Dict[str, str] = {
    "learningPaths": "learningPaths",
    "modules": "modules",
}


class MicrosoftLearnMCPTool:
    """Cliente de lectura del catálogo de Microsoft Learn."""

    def __init__(self) -> None:
        self.base_url = "https://learn.microsoft.com"
        self._paths_cache: Optional[List[Dict[str, Any]]] = None
        self._paths_cache_ts: float = 0.0
        self._modules_cache: Optional[List[Dict[str, Any]]] = None
        self._modules_cache_ts: float = 0.0
        logger.info("MicrosoftLearnMCPTool inicializado (solo API real)")

    async def search_learning_paths(
        self,
        query: str,
        certification: Optional[str] = None,
        max_results: int = 10,
        language: str = "en-us",
    ) -> List[Dict[str, Any]]:
        """Busca rutas de aprendizaje en Microsoft Learn."""
        paths = await self._get_all_paths(language)
        if not paths:
            logger.warning("No se pudieron obtener rutas desde la API real")
            return []

        if certification:
            cert_upper = certification.upper()
            filtered_paths = [
                item for item in paths
                if cert_upper in (item.get("certification_levels") or "").upper()
                or cert_upper in " ".join(item.get("products", [])).upper()
                or cert_upper in item.get("uid", "").upper()
                or cert_upper in item.get("title", "").upper()
            ]
            if filtered_paths:
                paths = filtered_paths

        if query:
            q_lower = query.lower()
            tokens = [token for token in q_lower.split() if len(token) > 2]

            def _score(path: Dict[str, Any]) -> int:
                text = (
                    path.get("title", "") + " " +
                    path.get("summary", "") + " " +
                    " ".join(path.get("roles", [])) + " " +
                    " ".join(path.get("products", []))
                ).lower()
                return sum(1 for token in tokens if token in text)

            scored = [(path, _score(path)) for path in paths]
            scored = [(path, score) for path, score in scored if score > 0]
            scored.sort(key=lambda item: item[1], reverse=True)
            paths = [path for path, _ in scored]

        return paths[:max_results]

    async def get_learning_path_details(
        self,
        path_url: str,
        language: str = "en-us",
    ) -> Optional[Dict[str, Any]]:
        """Obtiene detalles de una ruta a partir de su URL o UID."""
        paths = await self._get_all_paths(language)
        if not paths:
            return None

        normalized_input = self._normalize_learn_url(path_url)

        for item in paths:
            item_url = self._normalize_learn_url(item.get("url", ""))
            item_uid = item.get("uid", "")
            if item_url == normalized_input or item_uid in path_url:
                module_uids = item.get("modules", [])
                if not module_uids:
                    return item

                modules = await self._get_all_modules(language)
                modules_by_uid = {
                    module.get("uid", ""): module for module in modules
                }
                children: List[Dict[str, Any]] = []
                for module_uid in module_uids:
                    module = modules_by_uid.get(module_uid)
                    if not module:
                        continue
                    children.append(
                        {
                            "uid": module.get("uid", ""),
                            "title": module.get("title", ""),
                            "url": module.get("url", ""),
                            "duration_minutes": int(
                                module.get("duration_in_minutes", 0) or 0
                            ),
                            "summary": module.get("summary", ""),
                        }
                    )

                enriched_item = dict(item)
                enriched_item["children"] = children
                return enriched_item
        return None

    async def search_modules(
        self,
        query: str,
        certification: Optional[str] = None,
        max_results: int = 20,
        language: str = "en-us",
    ) -> List[Dict[str, Any]]:
        """Busca módulos individuales en Microsoft Learn."""
        modules = await self._get_all_modules(language)
        if not modules:
            logger.warning("No se pudieron obtener módulos desde la API real")
            return []

        filtered_modules = modules
        if certification:
            cert_upper = certification.upper()
            cert_filtered_modules = [
                item for item in filtered_modules
                if cert_upper in item.get("title", "").upper()
                or cert_upper in item.get("uid", "").upper()
            ]
            if cert_filtered_modules:
                filtered_modules = cert_filtered_modules

        if query:
            q_lower = query.lower()
            tokens = [token for token in q_lower.split() if len(token) > 2]

            def _score(module: Dict[str, Any]) -> int:
                text = (
                    module.get("title", "") + " " + module.get("summary", "")
                ).lower()
                return sum(1 for token in tokens if token in text)

            scored = [(module, _score(module)) for module in filtered_modules]
            scored = [(module, score) for module, score in scored if score > 0]
            scored.sort(key=lambda item: item[1], reverse=True)
            filtered_modules = [module for module, _ in scored]

        return filtered_modules[:max_results]

    async def get_popular_paths(
        self,
        certification: Optional[str] = None,
        limit: int = 5,
        language: str = "en-us",
    ) -> List[Dict[str, Any]]:
        """Devuelve rutas populares ordenadas por número de módulos."""
        paths = await self._get_all_paths(language)
        if not paths:
            return []

        if certification:
            cert_upper = certification.upper()
            filtered_paths = [
                item for item in paths
                if cert_upper in item.get("title", "").upper()
                or cert_upper in " ".join(item.get("products", [])).upper()
            ]
            if filtered_paths:
                paths = filtered_paths

        paths.sort(
            key=lambda item: item.get("number_of_children", 0),
            reverse=True,
        )
        return paths[:limit]

    def _is_cache_valid(self, ts: float) -> bool:
        return (time.monotonic() - ts) < _CACHE_TTL

    async def _get_all_paths(self, language: str = "en-us") -> List[Dict[str, Any]]:
        if self._paths_cache is not None and self._is_cache_valid(self._paths_cache_ts):
            return self._paths_cache

        raw = await self._fetch_catalog("learningpaths", language)
        if raw:
            self._paths_cache = raw
            self._paths_cache_ts = time.monotonic()
        return raw

    async def _get_all_modules(self, language: str = "en-us") -> List[Dict[str, Any]]:
        if self._modules_cache is not None and self._is_cache_valid(self._modules_cache_ts):
            return self._modules_cache

        raw = await self._fetch_catalog("modules", language)
        if raw:
            self._modules_cache = raw
            self._modules_cache_ts = time.monotonic()
        return raw

    async def _fetch_catalog(self, type_: str, language: str) -> List[Dict[str, Any]]:
        api_type = _CATALOG_TYPE_MAP.get(type_, type_)
        response_key = _CATALOG_RESPONSE_KEYS.get(api_type, api_type)
        url = f"{_CATALOG_BASE}?type={api_type}&locale={language}"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
                return payload.get(response_key, [])
        except httpx.HTTPError as exc:
            logger.error(f"Error HTTP al obtener catálogo '{type_}': {exc}")
            return []
        except Exception as exc:
            logger.error(
                f"Error inesperado al obtener catálogo '{type_}': {exc}")
            return []

    @staticmethod
    def _normalize_learn_url(url: str) -> str:
        if not url:
            return ""

        no_query = url.split("?", maxsplit=1)[0]
        return no_query.rstrip("/")


mslearn_mcp_tool = MicrosoftLearnMCPTool()
