# src/graph.py
from typing import Literal
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import node_retriever, node_analyzer, node_execute_booking, node_respond

# =====================================================================
# ENRUTAMIENTO CONDICIONAL DETERMINISTA
# =====================================================================
def conditional_router(state: AgentState) -> Literal["node_execute_booking", "node_respond"]:
    """
    Borde condicional que evalúa de forma rígida el estado de la intención:
    - Si el usuario requiere transaccionar una agenda ("book"), deriva al validador de slots.
    - Si es una duda (FAQ) o intención desconocida, pasa de inmediato a responder.
    """
    if state["intent"] == "book":
        return "node_execute_booking"
    return "node_respond"


# =====================================================================
# CONSTRUCCIÓN Y COMPILACIÓN DEL WORKFLOW (STATEGRAPH)
# =====================================================================

# 1. Inicializar el constructor del grafo utilizando el esquema estricto del estado
workflow = StateGraph(AgentState)

# 2. Registrar los componentes funcionales (Nodos) en la topología
workflow.add_node("node_retriever", node_retriever)
workflow.add_node("node_analyzer", node_analyzer)
workflow.add_node("node_execute_booking", node_execute_booking)
workflow.add_node("node_respond", node_respond)

# 3. Configurar el Punto de Entrada obligatorio del agente
workflow.set_entry_point("node_retriever")

# 4. Establecer la arista secuencial de datos hacia análisis semántico
workflow.add_edge("node_retriever", "node_analyzer")

# 5. Configurar los caminos dinámicos mediante el enrutador condicional
workflow.add_conditional_edges(
    "node_analyzer",
    conditional_router,
    {
        "node_execute_booking": "node_execute_booking",
        "node_respond": "node_respond"
    }
)

# 6. Conectar el nodo de agenda con la estación de despacho terminal
workflow.add_edge("node_execute_booking", "node_respond")

# 7. Declarar el Fin absoluto del ciclo del grafo conversacional
workflow.add_edge("node_respond", END)

# 8. Compilar la topología en un objeto ejecutable en memoria
compiled_graph = workflow.compile()