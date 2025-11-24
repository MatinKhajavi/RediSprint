# a2a_agents.py
"""
A2A (Agent-to-Agent) Sprint Planning System
Three agents collaborate via Redis Streams:
1. SprintReaderAgent: Reads Google Sheets sprint data
2. ContextEnricherAgent: Searches tickets/code and enriches descriptions
3. JiraCreatorAgent: Creates Jira tickets
"""
import os, asyncio, textwrap, json
from dotenv import load_dotenv
from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider
from agents import Agent, Runner
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import HybridQuery
from redisvl.utils.vectorize import OpenAITextVectorizer
from a2a_redis import RedisStreamsQueueManager
from a2a_redis.utils import create_redis_client

load_dotenv()

# ---- Configuration ----
REDIS_URL = os.environ["REDIS_URL"]
QUEUE_PREFIX = "a2a:sprint:"
QUEUE_ENRICH = "enrich-tickets"
QUEUE_CREATE = "create-jira"

# Google Sheets
SPREADSHEET_ID = "1nnErycyHRayc1OY_YWq06dOkaYTTPY9hTn6E3QwSXN4"
SHEET_RANGE = "Sheet1!A1:J1000"

# Composio
COMPOSIO_USER_ID = os.environ["COMPOSIO_USER_ID"]
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")

# Jira
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "SPRINT")
JIRA_DEFAULT_COMPONENT = os.environ.get("JIRA_COMPONENT_DEFAULT", "")

# Initialize clients
# Two separate Composio clients:
# 1. For Agent/Runner pattern (Jira agent) - with OpenAIAgentsProvider
composio_agents = Composio(api_key=COMPOSIO_API_KEY, provider=OpenAIAgentsProvider()) if COMPOSIO_API_KEY else Composio(provider=OpenAIAgentsProvider())
# 2. For direct OpenAI API calls (Sheet reader) - without provider
composio_direct = Composio(api_key=COMPOSIO_API_KEY) if COMPOSIO_API_KEY else Composio()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Redis search indices
tickets_idx = SearchIndex.from_existing("tickets_idx", redis_url=REDIS_URL)
code_idx = SearchIndex.from_existing("code_idx", redis_url=REDIS_URL)
embedder = OpenAITextVectorizer(model="text-embedding-3-small", api_config={"api_key": os.environ["OPENAI_API_KEY"]})


# Sheet columns: Sprint | Type | Title | Client Impact | Effort | Owner | Status | Assignee
COLS = ["Sprint", "Type", "Title", "Client Impact", "Effort", "Owner", "Status", "Assignee"]

# Priority mapping for client impact
PRIORITY_MAP = {
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Critical": 0
}

# ===== Helper Functions =====

def get_priority_from_impact(client_impact: str) -> int:
    """Convert client impact to numeric priority (lower = higher priority)"""
    for key in PRIORITY_MAP:
        if key.lower() in client_impact.lower():
            return PRIORITY_MAP[key]
    return 2  # default to Medium

def find_latest_sprint(rows):
    """Find the latest sprint from sheet rows"""
    sprints = []
    for row in rows:
        if len(row) > 0 and row[0]:  # Sprint column (first column)
            sprint_name = row[0].strip()
            if sprint_name and sprint_name not in sprints:
                sprints.append(sprint_name)
    
    # Return the last sprint (assumes sheet is ordered or newest at bottom)
    return sprints[-1] if sprints else None

def search_relevant_context(query_text, component=None, k_tickets=5, k_code=3):
    """Search Redis indices for relevant tickets and code"""
    qemb = embedder.embed(query_text)
    
    # Ensure list, not numpy array
    if hasattr(qemb, "tolist"):
        qemb = qemb.tolist()
    
    # Build optional filter for component
    from redisvl.query.filter import Tag
    comp_filter = (Tag("component") == component) if component else None
    
    # Search tickets: hybrid text+vector over title | body
    tq = HybridQuery(
        text=query_text,
        text_field_name="title|body",
        vector=qemb,
        vector_field_name="embedding",
        num_results=k_tickets,
        return_fields=["id","title","assignee","issuetype","component","body"],
        filter_expression=comp_filter,
    )
    tickets_hits = tickets_idx.query(tq)
    
    # Search code: hybrid over path | body
    cq = HybridQuery(
        text=query_text,
        text_field_name="path|body",
        vector=qemb,
        vector_field_name="embedding",
        num_results=k_code,
        return_fields=["id","repo","path","lang","body"]
    )
    code_hits = code_idx.query(cq)
    
    return tickets_hits, code_hits

def generate_enriched_description(row_data, tickets_hits, code_hits):
    """Use LLM to generate comprehensive Jira description based on past tickets and code"""
    title = row_data.get("title", "")
    itype = row_data.get("type", "Task")
    impact = row_data.get("client_impact", "")
    effort = row_data.get("effort", "")
    sprint = row_data.get("sprint", "")
    reporter = row_data.get("owner", "")
    assignee = row_data.get("assignee", "")
    
    # Format relevant tickets with their descriptions
    refs_tickets = "\n\n".join([
        f"**{h['id']}: {h['title']}**\n{h.get('body', 'No description')[:500]}..."
        for h in tickets_hits[:3]
    ]) or "No relevant past tickets found"
    
    # Format relevant code files with actual code snippets
    refs_code = "\n\n".join([
        f"File: {h['path']} ({h['lang']})\n```{h['lang']}\n{h.get('body', '')[:500]}\n```"
        for h in code_hits[:3]
    ]) or "No relevant code files found"
    
    prompt = f"""
You are writing a Jira ticket description for: "{title}"

Context:
- Type: {itype}
- Priority: {impact}
- Story Points: {effort}
- Sprint: {sprint}

Similar past tickets for reference:
{refs_tickets}

CODE ANALYSIS - Relevant codebase files found:
{refs_code}

Write a clear, concise Jira ticket description in PLAIN TEXT (no markdown headers). Keep it under 350 words.

IMPORTANT: You MUST analyze the actual code shown above and reference specific:
- Function names, class names, or methods that need modification
- File paths where changes will be made
- Current implementation details visible in the code snippets

Structure your description as:

1. OVERVIEW (2-3 sentences): What needs to be done and why

2. CODE ANALYSIS: Based on the codebase snippets above, identify which specific functions/classes/files need changes. Reference actual function names and explain what modifications are needed. This section should prove you've analyzed the real code.

3. WHY IT MATTERS: Justify the {impact} priority with concrete impact on users or system

4. ACCEPTANCE CRITERIA (simple bullet list with dashes):
- Specific, testable outcomes
- Include at least one criterion about the code changes

5. TESTING: Brief note on how to verify the changes work

Write in clear paragraphs. Be specific about code - this proves we've done semantic search on the codebase!
"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": textwrap.dedent(prompt)}]
    )
    
    return response.choices[0].message.content


# ===== AGENT 1: Sprint Reader Agent =====
async def sprint_reader_agent(enrich_queue):
    """
    Agent 1: Reads Google Sheets to get sprint data
    Sends raw ticket data to enrichment queue
    """
    print("\n🔍 [AGENT 1: Sprint Reader] Starting...")
    
    # Get Google Sheets tools (simpler approach from test_search_enrich.py)
    # Use composio_direct (without provider) for direct OpenAI API calls
    tools = composio_direct.tools.get(user_id=COMPOSIO_USER_ID, toolkits=["GOOGLESHEETS"])
    
    task_prompt = f"Get all data from spreadsheet {SPREADSHEET_ID}, range {SHEET_RANGE}"
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": task_prompt}],
        tools=tools,
    )
    
    # Handle tool calls manually
    values = None
    if completion.choices[0].message.tool_calls:
        for tool_call in completion.choices[0].message.tool_calls:
            result = composio_direct.tools.execute(
                slug=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                user_id=COMPOSIO_USER_ID,
                dangerously_skip_version_check=True
            )
            
            if result.get('successful'):
                data = result.get('data', {})
                value_ranges = data.get('valueRanges', [])
                if value_ranges:
                    values = value_ranges[0].get('values', [])
    
    if not values or len(values) < 2:
        print("[AGENT 1] ❌ No data found in sheet")
        return
    
    # Parse header and rows
    header = [h.strip() for h in values[0]]
    rows = values[1:]
    
    print(f"[AGENT 1] 📊 Found {len(rows)} rows with columns: {header}")
    
    # Find latest sprint
    latest_sprint = find_latest_sprint(rows)
    print(f"[AGENT 1] 🎯 Latest sprint identified: {latest_sprint}")
    
    # Normalize rows to standard format
    def parse_row(row_values):
        row_dict = {header[i]: row_values[i] if i < len(row_values) else "" 
                   for i in range(len(header))}
        return {
            "sprint": row_dict.get(COLS[0], ""),
            "type": row_dict.get(COLS[1], "Task"),
            "title": row_dict.get(COLS[2], ""),
            "client_impact": row_dict.get(COLS[3], ""),
            "effort": row_dict.get(COLS[4], ""),
            "owner": row_dict.get(COLS[5], ""),
            "status": row_dict.get(COLS[6], ""),
            "assignee": row_dict.get(COLS[7], ""),
        }
    
    # Filter for latest sprint and enqueue
    count = 0
    for row in rows:
        parsed = parse_row(row)
        
        # Skip empty rows or non-matching sprints
        if not parsed["title"].strip():
            continue
        if latest_sprint and parsed["sprint"] != latest_sprint:
            continue
        
        # Calculate priority
        priority = get_priority_from_impact(parsed["client_impact"])
        parsed["priority"] = priority
        
        # Send to enrichment queue
        await enrich_queue.enqueue_event({
            "type": "EnrichTicket",
            "data": parsed
        })
        count += 1
        print(f"[AGENT 1] ✅ Enqueued: {parsed['title'][:60]}... (Priority: {priority}, Impact: {parsed['client_impact']})")
        await asyncio.sleep(0.1)  # Small delay to avoid overwhelming Redis
    
    print(f"\n[AGENT 1] 🎉 Finished! Sent {count} tickets for enrichment\n")


# ===== AGENT 2: Context Enricher Agent =====
async def context_enricher_agent(enrich_queue, create_queue):
    """
    Agent 2: Searches previous tickets and codebase
    Enriches ticket descriptions with context
    Sends enriched data to Jira creation queue
    """
    print("\n🔎 [AGENT 2: Context Enricher] Starting...")
    
    processed = 0
    empty_polls = 0
    max_empty_polls = 2  # Exit after 2 consecutive empty polls (agents run sequentially)
    
    while True:
        event = None
        try:
            # a2a-redis doesn't support block_ms; poll with no_wait instead
            try:
                event = await enrich_queue.dequeue_event(no_wait=True)
                empty_polls = 0  # Reset counter when we get an event
            except RuntimeError:
                # No events available right now
                await asyncio.sleep(2)  # Shorter polling since agents run sequentially
                event = None
                # Check if we should exit after processing
                if processed > 0:
                    empty_polls += 1
                    if empty_polls >= max_empty_polls:
                        print(f"\n[AGENT 2] 🎉 Finished enriching {processed} tickets\n")
                        break
                continue
        except Exception as e:
            print(f"[AGENT 2] Dequeue error: {e}")
            continue
        
        if not event:
            continue
        
        try:
            if event.get("type") == "EnrichTicket":
                data = event["data"]
                print(f"\n[AGENT 2] 🔍 Enriching: {data['title'][:60]}...")
                
                # Build search query
                query = f"{data['title']}\n{data['client_impact']}\n{data['effort']}"
                
                # Search relevant context
                tickets_hits, code_hits = search_relevant_context(
                    query,
                    component=JIRA_DEFAULT_COMPONENT or None,
                    k_tickets=5,
                    k_code=3
                )
                
                print(f"[AGENT 2]   📋 Found {len(tickets_hits)} relevant tickets")
                print(f"[AGENT 2]   💻 Found {len(code_hits)} relevant code files")
                
                # Generate enriched description
                enriched_desc = generate_enriched_description(data, tickets_hits, code_hits)
                
                # Prepare for Jira creation
                jira_payload = {
                    "row_data": data,
                    "description": enriched_desc,
                    "tickets_context": [
                        {"id": h["id"], "title": h["title"]} for h in tickets_hits[:3]
                    ],
                    "code_context": [
                        {"path": h["path"], "lang": h["lang"]} for h in code_hits[:3]
                    ]
                }
                
                # Send to Jira creation queue
                event_data = {
                    "type": "CreateJiraTicket",
                    "payload": jira_payload
                }
                await create_queue.enqueue_event(event_data)
                print(f"[AGENT 2] 📤 Enqueued event to queue: {QUEUE_PREFIX}{QUEUE_CREATE}")
                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming Redis
                
                processed += 1
                print(f"[AGENT 2] ✅ Enriched and forwarded to Jira creator")
            else:
                print(f"[AGENT 2] ⚠️  Unknown event type: {event.get('type')}")
            
        except Exception as e:
            print(f"[AGENT 2] ❌ Error processing event: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always call task_done to acknowledge the event
            enrich_queue.task_done()  # Not async, don't await


# ===== AGENT 3: Jira Creator Agent =====
async def jira_creator_agent(create_queue):
    """
    Agent 3: Creates Jira tickets using Composio with Agent/Runner pattern
    """
    print("\n🎫 [AGENT 3: Jira Creator] Starting...")
    
    # Get Jira tools (use composio_agents with OpenAIAgentsProvider for Agent/Runner)
    jira_tools = composio_agents.tools.get(user_id=COMPOSIO_USER_ID, toolkits=["JIRA"])
    
    print(f"[AGENT 3] Available Jira tools: {len(jira_tools)}")
    
    # Create Jira agent
    jira_agent = Agent(
        name="Jira Creator Agent",
        instructions="""You are a helpful assistant that creates Jira tickets with the provided information. use "project_key": "SCRUM" and create the issues under the sprint they are""",
        tools=jira_tools,
    )
    
    created = 0
    empty_polls = 0
    max_empty_polls = 2  # Exit after 2 consecutive empty polls (agents run sequentially)
    
    while True:
        event = None
        try:
            # a2a-redis doesn't support block_ms; poll with no_wait instead
            try:
                event = await create_queue.dequeue_event(no_wait=True)
                empty_polls = 0  # Reset counter when we get an event
                print(f"[AGENT 3] 📬 Received event: {event.get('type') if event else 'None'}")
            except RuntimeError:
                # No events available right now
                await asyncio.sleep(2)  # Shorter polling since agents run sequentially
                event = None
                empty_polls += 1
                print(f"[AGENT 3] 💤 No events (poll {empty_polls}/{max_empty_polls})")
        except Exception as e:
            print(f"[AGENT 3] ❌ Dequeue error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if not event:
            # Exit after multiple empty polls (whether we created tickets or not)
            if empty_polls >= max_empty_polls:
                print(f"\n[AGENT 3] 🎉 Finished! Created {created} Jira tickets\n")
                break
            continue
        
        try:
            if event.get("type") == "CreateJiraTicket":
                payload = event["payload"]
                row_data = payload["row_data"]
                description = payload["description"]
                
                print(f"\n[AGENT 3] 🎫 Creating Jira ticket: {row_data['title'][:60]}...")
                
                # Prepare Jira creation prompt
                jira_prompt = f"""
Create a Jira ticket with the following details:

Project Key: {JIRA_PROJECT_KEY}
Summary: {row_data['title'][:255]}
Issue Type: {row_data['type'] or 'Task'}
Description: {description}
Priority: {['Critical', 'High', 'Medium', 'Low'][min(row_data.get('priority', 2), 3)]}
Assignee: {row_data['assignee'] or row_data['owner'] or ''}
Labels: sprint-automation
Component: {JIRA_DEFAULT_COMPONENT if JIRA_DEFAULT_COMPONENT else 'No component'}
Sprint ID: 1 (Sprint 2)

Additional context:
- Client Impact: {row_data['client_impact']}
- Effort: {row_data['effort']}
- Owner: {row_data['owner']}

Create this issue in Jira and assign it to Sprint ID 1.
"""
                
                # Use Agent/Runner pattern to create ticket
                result = await Runner.run(
                    starting_agent=jira_agent,
                    input=jira_prompt,
                )
                
                if result and result.final_output:
                    created += 1
                    print(f"[AGENT 3] ✅ Created ticket successfully!")
                    print(f"[AGENT 3]    Result: {str(result.final_output)[:300]}...")
                else:
                    print(f"[AGENT 3] ❌ Failed to create ticket")
            else:
                print(f"[AGENT 3] ⚠️  Unknown event type: {event.get('type')}")
            
        except Exception as e:
            print(f"[AGENT 3] ❌ Error creating Jira ticket: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always call task_done to acknowledge the event
            create_queue.task_done()  # Not async, don't await


# ===== Main Orchestrator =====
async def main():
    """
    Orchestrates the three agents using Redis Streams for communication
    """
    print("\n" + "="*70)
    print("🚀 A2A Sprint Planning System Starting")
    print("="*70)
    
    # Create Redis client and queue manager (reduced connection pool to avoid timeouts)
    redis_client = create_redis_client(url=REDIS_URL, max_connections=10)
    queue_manager = RedisStreamsQueueManager(redis_client, prefix=QUEUE_PREFIX)
    
    # Create queues for agent communication
    enrich_queue = await queue_manager.create_or_tap(QUEUE_ENRICH)
    create_queue = await queue_manager.create_or_tap(QUEUE_CREATE)
    
    print(f"✅ Redis queues created:")
    print(f"   - Enrichment: {QUEUE_PREFIX}{QUEUE_ENRICH}")
    print(f"   - Creation: {QUEUE_PREFIX}{QUEUE_CREATE}")
    print(f"✅ Google Sheet: {SPREADSHEET_ID}")
    print(f"✅ Jira Project: {JIRA_PROJECT_KEY}")
    print()
    
    # Run agents sequentially to avoid timing issues
    # Agent 1: Read from Google Sheets and enqueue for enrichment
    await sprint_reader_agent(enrich_queue)
    
    # Agent 2: Enrich tickets and enqueue for Jira creation
    await context_enricher_agent(enrich_queue, create_queue)
    
    # Agent 3: Create Jira tickets
    await jira_creator_agent(create_queue)
    
    print("\n" + "="*70)
    print("✨ All agents completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
