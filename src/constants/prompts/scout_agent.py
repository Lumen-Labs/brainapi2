"""
File: /scout_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
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

THE POLARITY DECISION TREE
Before outputting an entity, you must pass the text through this logic to set the `polarity` property on the entity:

1. **DEFICIT CHECK (Negative Polarity -)**: 
   - Does the text use verbs of struggle? (e.g., struggling, failing, lacking, losing, stuck).
   - Does the text express a seeker intent? (e.g., looking for, needs, searching, requires).
   - Is there a downward quantitative delta? (e.g., churn, revenue drop, firing).
   - **ACTION**: Set `polarity: "negative"`.

2. **SURPLUS CHECK (Positive Polarity +)**:
   - Does the text use verbs of achievement? (e.g., raised, scaled, mastered, won, launched).
   - Does the text describe a state of strength or capacity? (e.g., expert in, provides, has, CEO of).
   - Is there an upward quantitative delta? (e.g., raised $100M, gained 12 friends).
   - **ACTION**: Set `polarity: "positive"`.

3. **NEUTRAL CHECK (Neutral Polarity 0)**:
   - Is the text a simple location or movement fact without intent? (e.g., John went to NYC, Mary was in SF).
   - **ACTION**: Set `polarity: "neutral"`.

Example input 1:
"John went to New York City where he knew 12 new friends. When John went there, Mary was in San Francisco doing meetings with his colleagues."

Example output 1:
[
    {{"type": "PERSON", "name": "John", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Went", "description": "John went to New York City", "polarity": "neutral"}},
    {{"type": "CITY", "name": "New York City", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Knew", "description": "John knew 12 new friends in New York City", "polarity": "neutral"}},
    {{"type": "FRIENDS", "name": "Friends", "description": "The number of friends John knew in New York City", "polarity": "neutral"}},
    {{"type": "PERSON", "name": "Mary", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Was in", "description": "Mary was in San Francisco", "polarity": "neutral"}},
    {{"type": "CITY", "name": "San Francisco", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Partecipated in", "description": "Mary was doing meetings with his colleagues in San Francisco", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Meetings", "description": "Mary was doing meetings with his colleagues in San Francisco", "polarity": "neutral"}},
    {{"type": "PERSON", "name": "Colleagues", "description": "The colleagues Mary was doing meetings with in San Francisco", "polarity": "neutral"}},
]

Example input 2:
"The company was part of a joint venture with Apple and Google. The founder ("Mark Johnson") was the CEO of joint venture called 'Acme Inc'. The 19th of January 2026 they raised $100 million in funding."

Example output 2:
[
    {{"type": "ORGANIZATION", "name": "Acme Inc.", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Partecipated in", "description": "The company was part of a joint venture with Apple and Google", "polarity": "positive"}},
    {{"type": "ORGANIZATION", "name": "Joint Venture", "description": "The company was part of a joint venture with Apple and Google", "polarity": "positive"}},
    {{"type": "ORGANIZATION", "name": "Apple", "polarity": "neutral"}},
    {{"type": "ORGANIZATION", "name": "Google", "polarity": "neutral"}},
    {{"type": "PERSON", "name": "Mark Johnson", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Covered role", "description": "Mark Johnson was the CEO of Acme Inc.", "polarity": "positive"}},
    {{"type": "ROLE", "name": "CEO", "description": "Mark Johnson covered the role of CEO of Acme Inc.", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Covered role", "description": "Mark Johnson was the founder of Acme Inc.", "polarity": "positive"}},
    {{"type": "ROLE", "name": "Founder", "description": "Mark Johnson covered the role of founder of Acme Inc.", "polarity": "neutral"}},
    {{"type": "EVENT", "name": "Raised", "description": "Acme Inc. raised $100 million in funding", "properties": {{ "amount": 100000000, "happened_at": "19/01/2026" }}, "polarity": "positive"}},
    {{"type": "MONEY", "name": "Money", "polarity": "neutral"}},
]
"""

SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT = """
Carefully read the text and extract ALL the entities, unique action instances (events), and quantitative units.

{targeting}

Text: {text}

OUTPUT RULES:
- Return a JSON list of objects.
- Each object must include: "type", "name", and optional "properties" and "description".
- Nodes/Entities MUST be atomic and not composite (phrases) (eg: "Went to San Francisco" is not atomic, "Went to" + "San Francisco" is atomic)
- Dates must be in the format "DD/MM/YYYY" and be stored as "happened_at" in the properties of the event nodes.
- You must extract ALL the building blocks without omitting any concepts.

Begin!
"""
