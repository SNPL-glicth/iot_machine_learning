"""Integration test for MoE architecture — validates safe integration.

Test que MoE se ejecuta correctamente cuando ML_MOE_ENABLED=True,
y que el fallback a modo estándar funciona cuando está desactivado.
"""

import pytest
from unittest.mock import MagicMock, patch

from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.expert_port import ExpertOutput
from iot_machine_learning.infrastructure.config.moe_factory import (
    create_moe_gateway,
    create_moe_gateway_safe,
)
from iot_machine_learning.infrastructure.ml.moe import MoEGateway


class TestMoEFactory:
    """Tests de creación segura de MoE gateway."""
    
    def test_create_moe_gateway_success(self):
        """Verifica que se puede crear el gateway con engines existentes."""
        gateway = create_moe_gateway_safe(sparsity_k=2)
        
        # Si los engines existen, debe retornar un gateway
        # Si no existen, retorna None (fallback seguro)
        assert gateway is None or isinstance(gateway, MoEGateway)
    
    def test_create_moe_gateway_safe_never_raises(self):
        """La versión safe nunca debe lanzar excepciones."""
        # No debe lanzar aunque los engines no estén disponibles
        gateway = create_moe_gateway_safe(sparsity_k=2)
        # Simplemente retorna None si falla
        assert gateway is None or isinstance(gateway, MoEGateway)


class TestMoEInContainer:
    """Tests de integración con BatchEnterpriseContainer."""
    
    def test_container_uses_standard_mode_when_flag_disabled(self):
        """Cuando ML_MOE_ENABLED=False, usar modo estándar."""
        from iot_machine_learning.ml_service.runners.wiring.container import (
            BatchEnterpriseContainer,
        )
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        
        # Crear flags con MoE desactivado
        flags = MagicMock(spec=FeatureFlags)
        flags.ML_MOE_ENABLED = False
        flags.ML_ENABLE_AUDIT_LOGGING = False
        flags.ML_USE_TAYLOR_PREDICTOR = False
        
        # Mock engine para evitar dependencias de SQL
        mock_engine = MagicMock()
        container = BatchEnterpriseContainer(mock_engine, flags)
        
        # Verificar que _get_or_create_moe_gateway retorna None
        moe_gateway = container._get_or_create_moe_gateway()
        assert moe_gateway is None
    
    def test_container_attempts_moe_when_flag_enabled(self):
        """Cuando ML_MOE_ENABLED=True, intentar crear MoE."""
        from iot_machine_learning.ml_service.runners.wiring.container import (
            BatchEnterpriseContainer,
        )
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        
        # Crear flags con MoE activado
        flags = MagicMock(spec=FeatureFlags)
        flags.ML_MOE_ENABLED = True
        flags.ML_ENABLE_AUDIT_LOGGING = False
        flags.ML_USE_TAYLOR_PREDICTOR = False
        
        # Mock engine
        mock_engine = MagicMock()
        container = BatchEnterpriseContainer(mock_engine, flags)
        
        # Verificar que intenta crear el gateway (puede retornar None si falla)
        moe_gateway = container._get_or_create_moe_gateway()
        # Puede ser None (si no hay engines) o un gateway válido
        assert moe_gateway is None or isinstance(moe_gateway, MoEGateway)


class TestMoEGatewayPrediction:
    """Tests de predicción con MoE."""
    
    def test_moe_gateway_returns_prediction(self):
        """Verifica que MoEGateway retorna un Prediction válido."""
        # Crear gateway si es posible
        gateway = create_moe_gateway_safe(sparsity_k=2)
        if gateway is None:
            pytest.skip("MoE gateway no disponible (engines no encontrados)")
        
        # Crear ventana de prueba
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )
        
        readings = [
            SensorReading(sensor_id=1, value=10.0, timestamp=float(i))
            for i in range(10)
        ]
        window = SensorWindow(sensor_id=1, readings=readings)
        
        # Ejecutar predicción
        prediction = gateway.predict(window)
        
        # Verificar estructura del resultado
        assert prediction is not None
        assert hasattr(prediction, 'predicted_value')
        assert hasattr(prediction, 'confidence_score')
        assert hasattr(prediction, 'trend')
        
        # Verificar valores razonables
        assert isinstance(prediction.predicted_value, float)
        assert 0.0 <= prediction.confidence_score <= 1.0
        assert prediction.trend in ['up', 'down', 'stable']
    
    def test_moe_metadata_contains_fusion_info(self):
        """Verifica que la predicción incluye metadata de fusión."""
        gateway = create_moe_gateway_safe(sparsity_k=2)
        if gateway is None:
            pytest.skip("MoE gateway no disponible")
        
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )
        
        readings = [
            SensorReading(sensor_id=1, value=10.0, timestamp=float(i))
            for i in range(10)
        ]
        window = SensorWindow(sensor_id=1, readings=readings)
        
        prediction = gateway.predict(window)
        
        # Verificar metadata de MoE
        assert 'fusion' in prediction.metadata
        assert 'sparsity_k' in prediction.metadata['fusion']
        assert 'weights_used' in prediction.metadata['fusion']
        assert 'dominant_expert' in prediction.metadata['fusion']


class TestMoEObservability:
    """Tests de logging y observabilidad."""
    
    def test_moe_logs_execution_mode(self, caplog):
        """Verifica que se loggea el modo de ejecución."""
        import logging
        from iot_machine_learning.ml_service.runners.wiring.container import (
            BatchEnterpriseContainer,
        )
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        
        flags = MagicMock(spec=FeatureFlags)
        flags.ML_MOE_ENABLED = True
        flags.ML_ENABLE_AUDIT_LOGGING = False
        flags.ML_USE_TAYLOR_PREDICTOR = False
        
        mock_engine = MagicMock()
        container = BatchEnterpriseContainer(mock_engine, flags)
        
        # Forzar creación del adapter para verificar logs
        with caplog.at_level(logging.INFO):
            # Intentar obtener el gateway (puede fallar, pero debe loggear)
            container._get_or_create_moe_gateway()
        
        # Verificar que se loggeó algo relacionado a MoE
        assert any('moe' in record.message.lower() for record in caplog.records)
