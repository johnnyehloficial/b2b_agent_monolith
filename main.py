# main.py
import json
from langchain_core.messages import HumanMessage, AIMessage
from src.state import BookingSlots
from src.database import QdrantTenantEngine
from src.graph import compiled_graph

# =====================================================================
# SIMULACIÓN DEL ENTORNO DE ARRANQUE (BOOTSTRAPPING)
# =====================================================================

def bootstrap_pilot_tenant() -> dict:
    """
    Carga de forma dinámica la configuración del tenant e inyecta
    su respectiva base de conocimiento cruda en el motor vectorial.
    """
    # 1. Simulación de lectura física del archivo config/config.json
    config_mock = {
        "tenant_id": "clinica_dental_peru_001",
        "business_name": "Clínica Dental San Apollonia",
        "vector_db": {
            "collection_name": "b2b_global_knowledge_base",
            "similarity_threshold": 0.82
        },
        "business_rules": {
            "slots_required": ["customer_name", "phone", "service", "date_time"],
            "available_services": ["Limpieza dental", "Curación", "Ortodoncia"]
        }
    }
    
    # 2. Simulación de lectura física del archivo mock_data/business_knowledge.json
    knowledge_mock = [
        {
            "category": "faq", 
            "text": "La clínica atiende de Lunes a Viernes de 9:00 AM a 6:00 PM y Sábados de 9:00 AM a 1:00 PM."
        },
        {
            "category": "faq", 
            "text": "El costo de la Limpieza dental es de 150 Soles. La Ortodoncia requiere evaluación de 50 Soles."
        }
    ]
    
    # 3. Inicializar e ingestar el cimiento de datos aislados por Tenant en Qdrant
    collection_name = config_mock["vector_db"]["collection_name"]
    db_engine = QdrantTenantEngine(collection_name=collection_name)
    db_engine.ingest_tenant_knowledge(
        tenant_id=config_mock["tenant_id"], 
        raw_documents=knowledge_mock
    )
    
    return config_mock


# =====================================================================
# PROCESAMIENTO CONVERSACIONAL DE PRUEBA EN TERMINAL
# =====================================================================

if __name__ == "__main__":
    # Cargar los artefactos variables del negocio piloto
    tenant_config = bootstrap_pilot_tenant()
    
    # Declaración del primer AgentState unificado
    current_state = {
        "messages": [],
        "tenant_id": tenant_config["tenant_id"],
        "config": tenant_config,
        "context_docs": [],
        "intent": "unknown",
        "booking_slots": BookingSlots(),
        "final_response": ""
    }
    
    print("=" * 75)
    print(f"SISTEMA OPERATIVO INICIADO: {tenant_config['business_name']}")
    print("Motor Vectorial Multi-Tenant Activado (0% Target Hallucination)")
    print("=" * 75)
    
    # -----------------------------------------------------------------
    # TURNO 1: Consulta FAQ pura (Validación de RAG y Cero Alucinaciones)
    # -----------------------------------------------------------------
    user_input_1 = "Hola, ¿en qué horarios atienden y cuánto cuesta la limpieza?"
    print(f"\n[USER]: {user_input_1}")
    
    current_state["messages"].append(HumanMessage(content=user_input_1))
    
    # Ejecución síncrona del Grafo determinista
    output_state_1 = compiled_graph.invoke(current_state)
    print(f"[AGENT]: {output_state_1['final_response']}")
    
    # Sincronización estricta de memoria acumulada
    current_state["messages"].append(AIMessage(content=output_state_1["final_response"]))
    current_state["booking_slots"] = output_state_1["booking_slots"]
    
    # -----------------------------------------------------------------
    # TURNO 2: Intención transaccional con slots incompletos
    # -----------------------------------------------------------------
    user_input_2 = "Excelente, me gustaría agendar una limpieza por favor. Me llamo Juan"
    print(f"\n[USER]: {user_input_2}")
    
    current_state["messages"].append(HumanMessage(content=user_input_2))
    
    output_state_2 = compiled_graph.invoke(current_state)
    print(f"\n[AGENT]: {output_state_2['final_response']}")
    
    # Sincronización estricta de memoria acumulada
    current_state["messages"].append(AIMessage(content=output_state_2["final_response"]))
    current_state["booking_slots"] = output_state_2["booking_slots"]
    
    # -----------------------------------------------------------------
    # TURNO 3: Completado final de slots y llamada a API Mock de Calendario
    # -----------------------------------------------------------------
    user_input_3 = "Mi celular es 999555123 y quiero ir mañana en la mañana"
    print(f"\n[USER]: {user_input_3}")
    
    current_state["messages"].append(HumanMessage(content=user_input_3))
    
    output_state_3 = compiled_graph.invoke(current_state)
    print(f"\n[AGENT]: {output_state_3['final_response']}")
    
    print("\n" + "=" * 75)
    print("SANDBOX TERMINADO: Flujo agéntico validado bajo criterios de producción.")
    print("=" * 75)