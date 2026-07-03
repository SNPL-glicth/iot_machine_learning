"""ExtractorFactory — crea extractores de entidades para texto.

DESIGN NOTE: Diseño nuevo, no reconstrucción.
El módulo semantic_extraction/ original fue eliminado sin rastro en git
(corrupción de objetos). No existe contrato original para ExtractorFactory.

Decisión: factory que retorna RegexEntityExtractor (regex-only, sin NLP)
como extractor predeterminado ("compuesto" = un solo extractor regex
que cubre equipos, métricas, fechas, alertas y operaciones).

LIMITACIÓN CONOCIDA: domain_hint="general" está hardcodeado. No usa el
clasificador de dominio bilingüe (español/inglés) existente en el pipeline
de análisis. El RegexEntityExtractor ignora el domain_hint en la práctica
(su extract() lo acepta pero solo afecta domain_detected en el resultado).
Si en el futuro se conecta un extractor por dominio (ej. extractor específico
para infraestructura vs trading), el factory deberá aceptar un domain_hint
dinámico o usar el clasificador automático. Por ahora no es bloqueante porque
el extractor regex es agnóstico al dominio.
"""

from __future__ import annotations

from .composite_entity_extractor import RegexEntityExtractor


class ExtractorFactory:
    """Factory para crear extractores de entidades semánticas.

    Methods:
        create_composite_extractor: Retorna el extractor predeterminado
            (regex-only, cubre todos los dominios básicos).
    """

    @staticmethod
    def create_composite_extractor() -> RegexEntityExtractor:
        """Crear extractor compuesto (regex-only, sin NLP).

        LIMITACIÓN: domain_hint="general" hardcodeado.
        No utiliza el clasificador de dominio bilingüe existente.
        Ver docstring del módulo para detalle.

        Returns:
            Instancia de RegexEntityExtractor configurada para uso general.
        """
        return RegexEntityExtractor(domain_hint="general")
