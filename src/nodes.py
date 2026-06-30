# src/nodes.py
from typing import Dict, Any
from src.state import AgentState, BookingSlots
from src.database import QdrantTenantEngine

# =====================================================================
# SIMULADOR DE API EXTERNA (MOCK DE CALENDARIO)
# =====================================================================
class GoogleCalendarMock:
    """Simulador de lectura y escritura en la API de Google Calendar o Calendly."""
    @staticmethod
    def check_and_book(date_time: str, service: str) -> bool:
        # En producción real esto ejecutaría una llamada HTTP POST / REST API request
        if "10:00 AM" in date_time or "00" in date_time:
            return True
        return False


# =====================================================================
# CODIFICACIÓN DE NODOS CORE DEL FLUJO CONVERSACIONAL
# =====================================================================

def node_retriever(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 1: Intercepta la consulta entrante y extrae el contexto de Qdrant.
    Aplica el filtro por tenant_id garantizando un aislamiento de datos estricto.
    """
    last_message = state["messages"][-1].content if state["messages"] else ""
    tenant_id = state["tenant_id"]
    
    collection = state["config"]["vector_db"]["collection_name"]
    threshold = state["config"]["vector_db"]["similarity_threshold"]
    
    # Instanciamos el conector y ejecutamos la búsqueda contextual
    db_engine = QdrantTenantEngine(collection_name=collection)
    context = db_engine.search_isolated_knowledge(
        query=last_message, 
        tenant_id=tenant_id, 
        threshold=threshold
    )
    
    return {"context_docs": context}


def node_analyzer(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 2: Enrutador semántico y Extractor de Entidades (Slots).
    Simula la inferencia JSON estructurada de modelos como Llama-3-8B / DeepSeek-V3.
    """
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    current_slots = state["booking_slots"]
    
    # Clonamos los slots existentes en la sesión para preservar la idempotencia del estado
    updated_slots = BookingSlots(**current_slots.model_dump())
    
    # Clasificación heurística/semántica determinista de la intención
    if any(k in last_message for k in ["agendar", "cita", "turno", "reserva", "mañana", "fecha"]):
        intent = "book"
    else:
        intent = "faq"
        
    # Extracción simulada de slots del usuario (Simulación de NER vía Structured Outputs)
    if "limpieza" in last_message:
        updated_slots.service = "Limpieza dental"
    if "ortodoncia" in last_message:
        updated_slots.service = "Ortodoncia"
    if "juan" in last_message:
        updated_slots.customer_name = "Juan Perez"
    if "999" in last_message:
        updated_slots.phone = "999555123"
    if "mañana" in last_message:
        updated_slots.date_time = "2026-07-01 10:00 AM"
        
    return {"intent": intent, "booking_slots": updated_slots}


def node_execute_booking(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 3: Evalúa el llenado de slots obligatorios exigidos en config.json.
    Si faltan datos, activa una repregunta; si está completo, confirma en la API Mock.
    """
    slots = state["booking_slots"]
    required_fields = state["config"]["business_rules"]["slots_required"]
    
    # Identificar dinámicamente campos ausentes basándose en las reglas de negocio
    missing = [field for field in required_fields if getattr(slots, field) is None]
    
    if not missing:
        # Slots completos: Interactuar con el backend de Google Calendar
        success = GoogleCalendarMock.check_and_book(
            date_time=slots.date_time, 
            service=slots.service
        )
        if success:
            response = f"¡Perfecto! Su cita para {slots.service} ha sido agendada con éxito para el {slots.date_time}."
        else:
            response = "Lo lamento, ese horario ya no se encuentra disponible. Por favor, indique otra hora."
    else:
        # Gestión del bucle conversacional: Solicitar de forma ordenada la primera variable ausente
        prompts = {
            "customer_name": "Por favor, facilíteme su nombre completo para registrar la reserva.",
            "phone": "Necesito un número de teléfono móvil para enviarle la confirmación.",
            "service": "Indique qué servicio desea contratar (Limpieza dental, Curación, Ortodoncia).",
            "date_time": "Por favor, indique la fecha y hora deseada para su cita."
        }
        response = prompts.get(missing[0], "Faltan datos obligatorios para proceder.")
        
    return {"final_response": response}


def node_respond(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 4: Capa terminal de formateo de salida para el cliente.
    Funde e inyecta la información recuperada de Qdrant si la intención es una FAQ.
    """
    if state["intent"] == "faq":
        if state["context_docs"]:
            # RAG puro inyectado directo al flujo de respuesta libre de alucinaciones
            clean_output = f"[FAQ Validada]: {state['context_docs'][0]}"
        else:
            clean_output = "Disculpe, no dispongo de esa información exacta en nuestro catálogo de atención."
    else:
        # Respuestas transaccionales generadas por el nodo ejecutor de reservas
        clean_output = state["final_response"]
        
    return {"final_response": clean_output}