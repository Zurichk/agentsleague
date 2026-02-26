"""
Herramientas e integraciones para AEP CertMaster.

Este módulo contiene todas las herramientas externas y servicios
de integración utilizados por los agentes:

- Microsoft Learn Certifications API (catálogo oficial en tiempo real)
- Microsoft Learn MCP Server
- Bing Search API
- Azure Email Service
- Azure Cosmos DB (persistencia)
- Generación de calendarios (.ics)
"""

from __future__ import annotations

from .calendar import CalendarTool, calendar_tool
from .certifications import CertificationsTool, certifications_tool
from .email import EmailTool, email_tool
from .mslearn_mcp import MicrosoftLearnMCPTool, mslearn_mcp_tool
from .persistence import PersistenceTool, persistence_tool
from .web_search import WebSearchTool, web_search_tool

__version__ = "0.1.0"

__all__ = [
    # Tools
    "CalendarTool",
    "CertificationsTool",
    "EmailTool",
    "MicrosoftLearnMCPTool",
    "PersistenceTool",
    "WebSearchTool",

    # Instances
    "calendar_tool",
    "certifications_tool",
    "email_tool",
    "mslearn_mcp_tool",
    "persistence_tool",
    "web_search_tool",
]
