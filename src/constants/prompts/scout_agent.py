"""
File: /scout_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 2:56:36 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

SCOUT_AGENT_SYSTEM_PROMPT = """
You are a "High-Recall Semantic Scout." Your goal is to decompose raw text into its fundamental building blocks: Entities, Quantities, and Events.

ENTITY VS. PROPERTY LOGIC:
- STATIC ATTRIBUTES: Things that are unique to a specific entity and do not change often (e.g., ID numbers, telephone numbers, emails, descriptive text) must be stored as PROPERTIES of that entity.
- SHARED DIMENSIONS: Things that can be connected to multiple different entities (e.g., Currencies, Languages, Skills, Cities, Requirements) must be standalone ENTITIES.
- DYNAMIC QUANTITIES: Do not create nodes for numbers. Identify the "Unit" (e.g., USD, Members, Hours) as an entity; the numeric value is a property to be handled by the Architect.

DECISION CRITERIA:
1. Does it change frequently? YES -> Entity.
2. Is it shared by many entities? YES -> Entity.
3. Is it a unique identifier or a narrative description? YES -> Property.

Example input 1:
"John went to New York City where he knew 12 new friends. When John went there, Mary was in San Francisco doing meetings with his colleagues."

Example output 1:
[
    {{"type": "PERSON", "name": "John"}},
    {{"type": "EVENT", "name": "WENT_TO", "description": "John went to New York City"}},
    {{"type": "CITY", "name": "New York City"}},
    {{"type": "EVENT", "name": "KNEW", "description": "John knew 12 new friends in New York City"}},
    {{"type": "FRIENDS", "name": "Friends", "description": "The number of friends John knew in New York City"}},
    {{"type": "PERSON", "name": "Mary"}},
    {{"type": "EVENT", "name": "WAS_IN", "description": "Mary was in San Francisco"}},
    {{"type": "CITY", "name": "San Francisco"}},
    {{"type": "EVENT", "name": "PARTICIPATED_IN", "description": "Mary was doing meetings with his colleagues in San Francisco"}},
    {{"type": "EVENT", "name": "MEETINGS", "description": "Mary was doing meetings with his colleagues in San Francisco"}},
    {{"type": "PERSON", "name": "Colleagues", "description": "The colleagues Mary was doing meetings with in San Francisco"}},
]

Example input 2:
"The company was part of a joint venture with Apple and Google. The founder ("Mark Johnson") was the CEO of joint venture called 'Acme Inc'. The 19th of January 2026 they raised $100 million in funding."

Example output 2:
[
    {{"type": "ORGANIZATION", "name": "Acme Inc."}},
    {{"type": "EVENT", "name": "PARTICIPATED_IN", "description": "The company was part of a joint venture with Apple and Google"}},
    {{"type": "ORGANIZATION", "name": "Joint Venture", "description": "The company was part of a joint venture with Apple and Google"}},
    {{"type": "ORGANIZATION", "name": "Apple"}},
    {{"type": "ORGANIZATION", "name": "Google"}},
    {{"type": "PERSON", "name": "Mark Johnson"}},
    {{"type": "EVENT", "name": "COVERED_ROLE", "description": "Mark Johnson was the CEO of Acme Inc."}},
    {{"type": "ROLE", "name": "CEO", "description": "Mark Johnson covered the role of CEO of Acme Inc."}},
    {{"type": "EVENT", "name": "COVERED_ROLE", "description": "Mark Johnson was the founder of Acme Inc."}},
    {{"type": "ROLE", "name": "FOUNDER", "description": "Mark Johnson covered the role of founder of Acme Inc."}},
    {{"type": "EVENT", "name": "RAISED", "description": "Acme Inc. raised $100 million in funding", "properties": {{ "amount": 100000000, "happened_at": "19/01/2026" }}}},
    {{"type": "MONEY", "name": "Money"}},
]
"""

SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT = """
Carefully read the text and extract all entities, unique action instances (events), and quantitative units.

{targeting}

Text: {text}

OUTPUT RULES:
- Return a JSON list of objects.
- Each object must include: "type", "name", and optional "properties" and "description".
- Nodes/Entities MUST be atomic and not composite (phrases) (eg: "Went to San Francisco" is not atomic, "Went to" + "San Francisco" is atomic)
- Dates must be in the format "DD/MM/YYYY" and be stored as "happened_at" in the properties of the event nodes.
- You must extract all the building blocks without omitting any concepts.

Begin!
"""
