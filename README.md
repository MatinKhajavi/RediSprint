# RediSprint

An A2A (Agent-to-Agent) Sprint Planning System that automates Jira ticket creation from Google Sheets using AI-powered context enrichment.

Built for the [Agents with Superpowers Context Engineering Hackathon](https://sf.aitinkerers.org/p/agents-with-superpowers-context-engineering-hackathon-w-redis) in San Francisco (November 2024) - Winner of the one-day hackathon.

## Overview

This system uses three collaborative agents that communicate via Redis Streams:

1. Sprint Reader Agent - Reads sprint planning data from Google Sheets
2. Context Enricher Agent - Searches previous tickets and codebase to enrich ticket descriptions with relevant context using Redis vector search
3. Jira Creator Agent - Creates comprehensive Jira tickets with AI-generated descriptions

## Key Features

- AI-powered context enrichment using GPT-4o
- Semantic search over past tickets and codebase using vector embeddings
- Priority-based ticket creation based on client impact
- Asynchronous agent communication via Redis Streams
- Automatic latest sprint detection
- Web UI for easy execution and real-time monitoring

## Configuration

### Google Sheet Format

Your Google Sheet should have these columns (in order):

| Sprint | Type | Title | Client Impact | Effort | Owner | Status | Assignee |
|--------|------|-------|---------------|--------|-------|--------|----------|
| Sprint 5 | Story | Add user authentication | High | 5 days | John | Planned | john@company.com |

### Advanced Configuration

Additional settings in `a2a_agents.py` (usually don't need to change):

- `QUEUE_PREFIX`: Redis queue prefix (default: "a2a:sprint:")
- Search parameters: Number of tickets and code files to retrieve
- Agent polling intervals and timeouts

### Priority Mapping

The system automatically maps Client Impact to Jira Priority:

| Client Impact | Jira Priority |
|---------------|---------------|
| Critical      | Highest       |
| High          | High          |
| Medium        | Medium        |
| Low           | Low           |

## How It Works

1. Agent 1 reads your Google Sheet and identifies the latest sprint
2. For each ticket in the latest sprint:
   - Agent 2 performs semantic search on past tickets (top 5 results)
   - Agent 2 performs semantic search on codebase (top 3 files)
   - Agent 2 uses GPT-4o to generate a comprehensive ticket description
   - Agent 2 forwards enriched data to Agent 3
3. Agent 3 creates the Jira ticket with all enriched information


## Acknowledgments

Built for the Agents with Superpowers Context Engineering Hackathon hosted by AI Tinkerers SF in collaboration with Redis.

Winner of the one-day hackathon (November 2024).
