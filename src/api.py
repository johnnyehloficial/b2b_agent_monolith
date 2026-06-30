# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from src.state import BookingSlots
from src.graph import compiled_graph
from main import bootstrap_pilot_tenant

app = FastAPI(
    title="Multi-Tenant B2B AI Agent Router",
    version="1.1.0",
    description="Endpoint REST con persistencia de memoria de sesión (Stateful Threads)."
)

# Inicialización de la base de conocimiento de Qdrant al arrancar
tenant_config = bootstrap_pilot_tenant()

# =====================================================================
# REGISTRO GLOBAL DE MEMORIA (Simula un Checkpointer de producción en RAM)
# =====================================================================
_SESSION_DB: Dict[str, Dict[str, Any]] = {}


# Esquemas de Datos Pydantic v2
class MessageRequest(BaseModel):
    session_id: str = Field(..., example="whatsapp_+51999555123", description="ID único de chat por usuario")
    tenant_id: str = Field(..., example="clinica_dental_peru_001")
    message: str = Field(..., example="Quiero agendar una limpieza")

class MessageResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    current_slots: dict


@app.post("/api/v1/chat", response_model=MessageResponse)
async def handle_user_message(payload: MessageRequest):
    """
    Procesa el mensaje del usuario recuperando el historial y los slots 
    guardados para ese 'session_id' específico, manteniendo la continuidad del flujo.
    """
    # 1. Validación estricta de aislamiento Multi-Tenant
    if payload.tenant_id != tenant_config["tenant_id"]:
        raise HTTPException(
            status_code=403, 
            detail=f"Tenant ID '{payload.tenant_id}' no autorizado en este nodo."
        )

    # 2. Recuperar o inicializar el estado histórico del hilo conversacional
    if payload.session_id not in _SESSION_DB:
        _SESSION_DB[payload.session_id] = {
            "messages": [],
            "tenant_id": payload.tenant_id,
            "config": tenant_config,
            "context_docs": [],
            "intent": "unknown",
            "booking_slots": BookingSlots(),
            "final_response": ""
        }
    
    # Extraemos el estado persistente de este usuario de manera idempotente
    user_state = _SESSION_DB[payload.session_id]
    
    # 3. Inyectar el nuevo mensaje recibido en el historial de LangGraph
    user_state["messages"].append(HumanMessage(content=payload.message))

    try:
        # 4. Invocación del Grafo pasándole el estado acumulado
        final_state = compiled_graph.invoke(user_state)
        
        # 5. Guardar los cambios calculados por LangGraph de vuelta en nuestra base de datos de sesión
        _SESSION_DB[payload.session_id]["messages"].append(AIMessage(content=final_state["final_response"]))
        _SESSION_DB[payload.session_id]["booking_slots"] = final_state["booking_slots"]
        _SESSION_DB[payload.session_id]["intent"] = final_state["intent"]
        
        return MessageResponse(
            session_id=payload.session_id,
            intent=final_state["intent"],
            response=final_state["final_response"],
            current_slots=final_state["booking_slots"].model_dump()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la ejecución del grafo conversacional: {str(e)}"
        )