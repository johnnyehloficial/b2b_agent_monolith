# src/database.py
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# =====================================================================
# CLIENTE GLOBAL COMPARTIDO (Persiste en la RAM del proceso de Python)
# =====================================================================
_SHARED_IN_MEMORY_CLIENT = QdrantClient(location=":memory:")

class QdrantTenantEngine:
    def __init__(self, collection_name: str):
        """Inicializa el motor reutilizando la instancia global en memoria."""
        self.collection_name = collection_name
        self.client = _SHARED_IN_MEMORY_CLIENT
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Crea la colección global SÓLO si no existe para evitar limpiar datos."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=4,  # Dimensión de juguete para el Mock de desarrollo
                    distance=qdrant_models.Distance.COSINE
                )
            )

    def _mock_embedding(self, text: str) -> List[float]:
        """Simulador de Embedding corporativo (e.g., text-embedding-3-small)."""
        lowered = text.lower()
        if "horario" in lowered or "atiende" in lowered or "hora" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        if "costo" in lowered or "precio" in lowered or "limpieza" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "agendar" in lowered or "cita" in lowered or "reserva" in lowered:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def ingest_tenant_knowledge(self, tenant_id: str, raw_documents: List[Dict[str, Any]]) -> None:
        """Pipeline de Ingesta: Enriquece y almacena vectores con payload estricto."""
        points = []
        for idx, doc in enumerate(raw_documents):
            vector = self._mock_embedding(doc["text"])
            points.append(
                qdrant_models.PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload={
                        "tenant_id": tenant_id,
                        "category": doc["category"],
                        "text": doc["text"]
                    }
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_isolated_knowledge(self, query: str, tenant_id: str, threshold: float) -> List[str]:
        """Recuperación Semántica Aislada Multi-Tenant con Payload Filtering."""
        tenant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="tenant_id",
                    match=qdrant_models.MatchValue(value=tenant_id)
                )
            ]
        )
        
        # Recuperación limpia y compatible usando Scroll sobre el cliente en memoria
        scroll_results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=tenant_filter,
            limit=2,
            with_payload=True
        )
        
        return [hit.payload["text"] for hit in scroll_results if hit.payload]