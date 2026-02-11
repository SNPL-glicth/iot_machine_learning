"""Capa de Dominio — Lógica de negocio pura UTSAE.

Arquitectura hexagonal: esta capa NO conoce infraestructura.
Solo define entidades, value objects, ports (interfaces) y servicios de dominio.

Dependencias permitidas: solo stdlib + typing.
Dependencias prohibidas: sqlalchemy, fastapi, numpy, sklearn, redis.
"""
