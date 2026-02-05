<p align="center">
  <img src="https://img.shields.io/badge/version-2.4.4--dev-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/python-3.11+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/license-AGPLv3%20%2B%20Commons%20Clause-purple?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🧠 BrainAPI</h1>

<p align="center">
  <strong>A cognitive memory layer for AI applications</strong>
  <br/>
  <em>Transform unstructured data into actionable knowledge with human-inspired retrieval</em>
</p>

<p align="center">
  <a href="#-usecases">Use Cases</a> •
  <a href="#-what-is-brainapi">Overview</a> •
  <a href="#-core-philosophy-the-triangle-of-attribution">Core Philosophy</a> •
  <a href="#-the-agentic-swarm">Agents</a> •
  <a href="#-sdks--packages">SDKs</a> •
  <a href="https://brainapi.lumen-labs.ai/docs/quickstart">Quick Start ↗</a> •
  <a href="https://brainapi.lumen-labs.ai/docs/rest">API Docs ↗</a>
</p>

<p align="center">
  <img src="https://pbs.twimg.com/media/G6jXa2HXsAAJhDO?format=jpg&amp;name=4096x4096" alt="BrainAPI Concept Art" width="1000" style="border-radius:4px;box-shadow:0 2px 12px rgba(0,0,0,0.09);margin-bottom:16px;"/>
</p>

## 🏃‍♂️ Local Development & Running BrainAPI2

You can run **BrainAPI2 locally** in two main ways:

### 1. Clone or Download the Project

- **Clone via Git:**
  ```sh
  git clone https://github.com/lumen-labs/brainapi2.git
  cd brainapi2
  ```
- **Or Download ZIP:**  
  [Download ZIP from GitHub](https://github.com/lumen-labs/brainapi2/archive/refs/heads/main.zip), then unzip and `cd` into the project folder.

- **Install dependencies (recommended with Python 3.11+):**

  ```sh
  poetry install
  ```

- **Use the Makefile commands to start the API & Workers:**

  ```sh
  make start-all
  ```

---

### 2. Run Using the Container Image

- **Pull the prebuilt container:**

  ```sh
  docker pull ghcr.io/lumen-labs/brainapi:latest
  ```

- **Start all required services and the BrainAPI container:**
  Use the included `example-docker-compose.yaml` to start up BrainAPI and all dependencies together:

  ```sh
  docker compose -f example-docker-compose.yaml up -d
  ```

---

> This will bring up all necessary databases and services so data ingestion, querying, and API access all work out-of-the-box.

See the [`example-docker-compose.yaml`](./example-docker-compose.yaml) file for configuration details, ports, and volumes.
For step-by-step instructions, visit the [Quick Start Guide](https://brainapi.lumen-labs.ai/docs/quickstart).

---

<p align="center">
  <a href="https://www.youtube.com/watch?v=wWwTFU5-qeA">
    <img src="https://img.youtube.com/vi/wWwTFU5-qeA/0.jpg" alt="BrainAPI Overview Video" width="100%" style="aspect-ratio: 16/9;object-fit: cover;border-radius:4px;box-shadow:0 2px 12px rgba(0,0,0,0.09);margin-bottom:16px;"/>
    <br/>
    <strong>▶ Watch: BrainAPI (non-technical) Overview Video</strong>
  </a>
</p>

---

## 📖 Use Cases

### 🚀 Example Use Cases

BrainAPI enables advanced knowledge representation and semantic retrieval for a wide range of AI-powered applications. Here are a few core use cases:

---

#### 1. **Recommendation Systems**

Leverage BrainAPI's graph of actions, relationships, and temporal contexts to produce precise recommendations for users, whether for content, products, collaborators, or actions.

- **How:** By tracing _events_ such as purchases, likes, shares, or user journeys, BrainAPI can recommend similar or complementary items considering social influence, context, and personalized behavioural paths.
- **Example:**
  - _Text Input:_  
    `"Alice bought a book called 'Neural Networks 101' during the Spring AI Symposium."`
  - _Ingested JSON:_
    ```json
    {
      "actor": "Alice",
      "event": "purchased",
      "target": "Neural Networks 101",
      "context": "Spring AI Symposium"
    }
    ```
  - _Constructed Graph:_
    ```
    (Alice)-[:MADE {date: "2024-04-12"}]->(Purchase Event)-[:TARGETED]->(Neural Networks 101)
                                          \
                                           \-[:OCCURRED_WITHIN]->(Spring AI Symposium)
    ```
  - _Recommendation Example:_  
    _Bob also attended the Spring AI Symposium and might be interested in the same books as Alice._

---

#### 2. **Search Engines**

Move beyond keyword search and retrieve information via deep semantic connections, matching intent, events, and multi-hop reasoning.

- **How:** The event-centric model lets you ask nuanced questions that traditional full-text search can't answer—enabling contextual, temporal, and relational queries.
- **Example:**
  - _Text Input:_  
    `"Tesla presented their latest battery at the 2023 Battery Expo in Berlin."`
  - _Ingested JSON:_
    ```json
    {
      "actor": "Tesla",
      "event": "presented",
      "target": "latest battery",
      "context": {
        "event": "2023 Battery Expo",
        "location": "Berlin"
      }
    }
    ```
  - _Constructed Graph:_
    ```
    (Tesla)-[:MADE]->(Presentation Event)-[:TARGETED]->(Latest Battery)
                                        \
                                         \-[:OCCURRED_WITHIN]->(2023 Battery Expo)-[:HELD_IN]->(Berlin)
    ```
  - _Retrieval Example:_  
    _query: "What products did Tesla present in Berlin in 2023?"_<br>
    _result: "The latest battery was presented at the 2023 Battery Expo in Berlin."_

---

#### 3. **AI Memory for AI-Apps & Agents**

Equip your agents and apps with persistent, structured memory, allowing nuanced contextual understanding, continuity, and knowledge grounding.

- **How:** Store, retrieve, and update multi-turn conversations, observations, and learned facts as an evolving graph, empowering agents to reason, plan, and act over long time horizons.
- **Example:**
  - _Text Input Sequence:_
    1. `"The user's favorite tool is VSCode."`
    2. `"She also uses GitHub Copilot for code suggestions."`
  - _Ingested JSON:_
    ```json
    [
      {
        "actor": "User",
        "event": "prefers",
        "target": "VSCode"
      },
      {
        "actor": "User",
        "event": "uses",
        "target": "GitHub Copilot",
        "context": "code suggestions"
      }
    ]
    ```
  - _Constructed Graph:_
    ```
    (User)-[:MADE]->(Preference Event)-[:TARGETED]->(VSCode)
    (User)-[:MADE]->(Usage Event)-[:TARGETED]->(GitHub Copilot)
                                      \
                                       \-[:OCCURRED_WITHIN]->(Code Suggestions)
    ```
  - _Retrieval + Recommendation Query:_
    - _"Which productivity tools does the user rely on for coding?"_
    - _"Recommend AI tools that integrate with VSCode."_

---

### 🧩 **Sample: From Unstructured Data to Graph, Then Query & Recommend**

#### **Step 1: Ingest Text & JSON**

```json
{
  "actor": "Emily",
  "event": "organized",
  "target": "AI Ethics Meetup",
  "context": "London",
  "date": "2024-03-08"
}
```

#### **Step 2: BrainAPI Graph Representation**

```
(Emily)-[:MADE {date: "2024-03-08"}]->(Organizing Event)-[:TARGETED]->(AI Ethics Meetup)
                                        \
                                         \-[:OCCURRED_WITHIN]->(London)
```

#### **Step 3: Retrieve Information**

- _Query_: `"Who organized AI events in London in March 2024?"`
- _Result_:
  - `Emily organized the 'AI Ethics Meetup' in London on 2024-03-08.`

#### **Step 4: Generate Recommendations**

- _Query_: `"What other events has Emily organized, or what similar events are happening in London?"`
- _Result_:
  - List of past/future meetups in London, relevant organizers, or AI-themed events.

---

BrainAPI makes it easy to convert unstructured information into actionable, queryable, and recommendable knowledge — powering the next generation of memory-augmented AI applications.

---

## 📖 What is BrainAPI?

BrainAPI is an advanced **Knowledge Graph Ecosystem** designed for high-precision semantic reasoning and relational analysis across multi-domain datasets. Unlike traditional graphs that rely on static entity-to-entity links, BrainAPI uses a dynamic **Event-Centric architecture** — treating actions, interactions, and state changes as central nodes to capture multi-dimensional context, temporal tracking, and complex multi-hop reasoning.

> **Why Event-Centric?**  
> Move beyond simple keyword retrieval toward **"Action-Path Reasoning."** BrainAPI identifies not just _that_ two entities are connected, but _how_ they interacted, at what _magnitude_, and within what _environment_.

---

## 🔺 Core Philosophy: The Triangle of Attribution

Every action in the graph is modeled as a central hub connecting three critical points through directed energy vectors:

```
                    ┌─────────────────┐
                    │   EVENT HUB     │
                    │  (Action Node)  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │    ACTOR     │  │    TARGET    │  │   CONTEXT    │
    │   (Source)   │  │ (Recipient)  │  │  (Anchor)    │
    └──────────────┘  └──────────────┘  └──────────────┘
         :MADE            :TARGETED       :OCCURRED_WITHIN
```

| Vector         | Relationship                 | Description                                                                       |
| -------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| **Initiation** | `:MADE` / `:INITIATED`       | Connects the **Actor** to the **Event Hub**. Carries quantitative `amount` data.  |
| **Targeting**  | `:TARGETED` / `:DIRECTED_AT` | Connects the **Event Hub** to the **Target** (recipient/destination).             |
| **Context**    | `:OCCURRED_WITHIN`           | Connects the **Event Hub** to a **Persistent Anchor** (org, location, timeframe). |

---

## 🤖 The Agentic Swarm

BrainAPI transforms unstructured text into rigorous graph schemas through a specialized multi-agent pipeline:

| Agent               | Role                  | Responsibility                                                                       |
| :------------------ | :-------------------- | :----------------------------------------------------------------------------------- |
| 🔍 **Scout**        | Semantic Fact-Finding | Identifies raw entities, distinguishes static properties from dynamic shared anchors |
| 🏛️ **Architect**    | Structural Mapping    | Translates facts into the Triangle of Attribution, enforcing vector directionality   |
| 🧹 **Janitor**      | Directional Police    | Audits graph units, resolves UUIDs, flips inverted relationships violating ontology  |
| 🔄 **Consolidator** | Micro-Swarm Auditor   | Performs deduplication and hub merging via collaborative voting (MAKGED)             |

---

## 🔎 Retrieval & Intelligence Layer

<details>
<summary><strong>KGLA — Knowledge Graph Enhanced Language Agents</strong></summary>

Bridges structured facts and natural language. Extracts multi-hop paths and translates them into human-readable explanations using rich `description` properties stored in nodes and relationships.

</details>

<details>
<summary><strong>RGP — Relational Graph Perceiver with Temporal Sampling</strong></summary>

Applies **Temporal Subgraph Sampling** to prioritize contextually recent events while enabling "Non-Local Temporal Matching" — finding entities that shared similar challenges during the same chronological windows.

</details>

<details>
<summary><strong>HippoRAG2 — Subgraph Localization</strong></summary>

Uses **Personalized PageRank** to navigate large, disparate data clusters. By traversing abstract "Concept Nodes," bridges disconnected subgraphs to discover structurally distant but semantically related information.

</details>

<details>
<summary><strong>Quantitative Synergy Scoring</strong></summary>

Ranks results using a multi-factor formula balancing semantic similarity, temporal recency, and quantitative alignment:

$$Score = (Similarity \times W_1) + (Recency \times W_2) + (PropertyAlignment)$$

Retrieval based not just on _what_ an entity is, but on the _scale_ and _timing_ of their recorded actions.

</details>

---

## 📦 SDKs & Packages

Integrate BrainAPI into your applications using our official client libraries:

| Platform    | Package                                                               | Status                                                                                    |
| ----------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Python**  | [`lumen_brain`](https://pypi.org/project/lumen_brain/)                | ![PyPI](https://img.shields.io/pypi/v/lumen_brain?label=v0.x&color=orange) ⚠️ Pre-release |
| **Node.js** | [`lumen-brain`](https://www.npmjs.com/package/@lumenlabs/lumen-brain) | ![npm](https://img.shields.io/npm/v/lumen-brain?label=v0.x&color=orange) ⚠️ Pre-release   |

> **Note:** Both SDKs are currently at version 0.x and under active development. For production use cases, we recommend using the [REST API](https://brainapi.lumen-labs.ai/docs/rest) directly until v1.0 releases.

---

## 🤝 Contributing

<!-- Add contributing guidelines -->

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under **AGPLv3 + Commons Clause** — free for personal, research, and non-commercial use. Commercial usage (SaaS, embedding, redistribution) requires an [Enterprise License](mailto:hello@lumen-labs.ai) from Lumen Platforms Inc.

See the [LICENSE](LICENSE) file for full details.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/lumen-labs">Lumen Labs</a></sub>
</p>
