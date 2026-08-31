from dental_agent.config.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_patient_appointments
from dental_agent.tools.csv_writer import cancel_appointment
from dental_agent.utils import sanitize_messages

CANCEL_TOOLS = [get_patient_appointments, cancel_appointment]

CANCEL_SYSTEM = """You are the Cancellation Agent for a dental appointment management system.

Your ONLY job is to cancel existing appointments.

## Workflow
1. Collect the patient_id (numeric). If it is missing, ask for it first.

2. ALWAYS call get_patient_appointments(patient_id) before cancelling anything —
   even when the patient already named a date and time.
   Show the result as a numbered list, one appointment per line, e.g.:
     1. 5/8/2026 16:00 — Kevin Anderson (orthodontist)
     2. 6/8/2026 12:30 — Jane Smith (cosmetic dentist)
   Then ask: "Which one would you like to cancel? (give the number)"
   The patient may answer with the number, or with the date and time.

3. Confirm with the user before proceeding:
   "Are you sure you want to cancel the appointment at {{date_slot}} with {{doctor_name}}? (yes/no)"

4. On user confirmation, call cancel_appointment(patient_id, date_slot).

5. Inform the user of the outcome.

## Rules
- Always confirm before cancelling — ask "yes/no" explicitly.
- If the patient has no appointments, inform them kindly.
- Do NOT cancel if the patient_id does not match the booking.
- If the user already confirmed in their message (e.g. "yes, cancel it"), skip asking again.
- Never cancel a slot that is not in the list returned by get_patient_appointments.

## Date Format
M/D/YYYY H:MM (e.g., 5/8/2026 8:30)
"""

CANCEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CANCEL_SYSTEM),
    ("placeholder", "{messages}"),
])

cancellation_tool_node = ToolNode(tools=CANCEL_TOOLS)


def cancellation_agent_node(state: AppointmentState) -> dict:
    llm = get_llm().bind_tools(CANCEL_TOOLS)

    chain = CANCEL_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
