# src/state.py
from typing import Annotated, Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# =====================================================================
# 1. EXTRACTOR DE ENTIDADES / SLOTS (PYDANTIC V2)
# =====================================================================
class BookingSlots(BaseModel):
    """
    Esquema estricto para almacenar y validar de forma incremental 
    los datos requeridos para la reserva del cliente.
    """
    customer_name: Optional[str] = Field(
        None, 
        description="Nombre completo del cliente o paciente que solicita la cita."
    )
    phone: Optional[str] = Field(
        None, 
        description="Número telefónico o de WhatsApp de contacto válido."
    )
    service: Optional[str] = Field(
        None, 
        description="Tratamiento o servicio específico que el cliente desea reservar."
    )
    date_time: Optional[str] = Field(
        None, 
        description="Fecha y hora sugerida o confirmada para la cita."
    )


# =====================================================================
# 2. DEFINICIÓN DEL ESTADO CORE DEL GRAFO (LANGGRAPH)
# =====================================================================
class AgentState(TypedDict):
    """
    Fuente única de verdad de la sesión conversacional.
    Mantiene la memoria, las configuraciones del negocio y los slots extraídos.
    """
    # Almacena el historial completo del chat, añadiendo mensajes nuevos automáticamente
    messages: Annotated[list, add_messages]
    
    # Identificador único para el aislamiento estricto de datos (Multi-Tenant Lite)
    tenant_id: str
    
    # Diccionario agnóstico con las reglas cargadas desde el config.json
    config: Dict[str, Any]
    
    # Lista de textos exactos recuperados desde Qdrant para evitar alucinaciones
    context_docs: List[str]
    
    # Clasificación de la intención detectada en el último turno del usuario
    intent: Literal["faq", "book", "unknown"]
    
    # Instancia del modelo Pydantic para el seguimiento de los datos de la cita
    booking_slots: BookingSlots
    
    # Mensaje o payload final estructurado listo para enviarse al usuario
    final_response: str