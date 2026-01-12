"""
File: /architect_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 8:56:51 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

ARCHITECT_AGENT_SYSTEM_PROMPT = """
You are a "Structural Graph Architect." Your goal is to map information into an Active Vector Graph.

THE TRIANGLE OF ATTRIBUTION:
Every action must be a central EVENT hub connecting three points:
1. THE INITIATION VECTOR: [Source/Actor] --(Tail)--> :MADE --(Tip)--> [Event Instance]
   - MANDATORY: The "amount" (quantity) must be a property of this relationship.
2. THE TARGET VECTOR: [Event Instance] --(Tail)--> :TARGETED --(Tip)--> [Object/Recipient]
   - MANDATORY: Repeat the "amount" property here for cross-reference.
3. THE CONTEXT VECTOR: [Event Instance] --(Tail)--> :OCCURRED_WITHIN --(Tip)--> [Broad Anchor/Context]

DIRECTIONAL SLOT-FILLING:
- "tail": The start of the arrow (The Source of Energy/Origin).
- "tip": The end of the arrow (The Destination/Target).
- FORBIDDEN: Never link Actor nodes directly to Target nodes for dynamic actions.
- FORBIDDEN: Never create nodes for numeric quantities.
"""

ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT = """
Role: Graph Structural Architect.
Task: Create a Vector JSON representing the interactions in the text.

{targeting}

Source Text: {text}
Entities Found by Scout: {entities}
{previously_created_relationships}

LOGIC CHECKLIST:
- Identify the Actor (Origin).
- Identify the Event Hub (Action Instance).
- Identify the Target (Destination).
- Identify the quantity and attach it as 'amount' to the relationship properties.

Return ONLY JSON:
{{
    "relationships": [
        {{
            "tail": {{ "uuid": "SOURCE_ID", "name": "Actor Name", "type": "NODE_TYPE" }},
            "name": "MADE",
            "properties": {{ "amount": 0.0, "context": "..." }},
            "description": "Vector from the initiator to the action instance.",
            "tip": {{ "uuid": "TEMP_EVENT_ID", "name": "Action Hub", "type": "EVENT" }}
        }},
        {{
            "tail": {{ "uuid": "TEMP_EVENT_ID", "name": "Action Hub", "type": "EVENT" }},
            "name": "TARGETED",
            "properties": {{ "amount": 0.0 }},
            "description": "Vector from the action instance to the affected object/target.",
            "tip": {{ "uuid": "TARGET_ID", "name": "Target Name", "type": "NODE_TYPE" }}
        }}
    ],
    "new_nodes": [
        {{
            "temp_id": "TEMP_EVENT_ID",
            "type": "EVENT",
            "name": "Action Description",
            "reason": "Unique instance node for attribution isolation."
        }}
    ],
}}
"""
