#!/usr/bin/env python3
"""
Test Script: Sheet Reading + Search + Enrichment
Tests everything before Jira ticket creation
"""
import os, asyncio, textwrap, json
from dotenv import load_dotenv
from composio import Composio
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import HybridQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import OpenAITextVectorizer

load_dotenv()

# Configuration
SPREADSHEET_ID = "1nnErycyHRayc1OY_YWq06dOkaYTTPY9hTn6E3QwSXN4"
SHEET_RANGE = "Sheet1!A1:J1000"
REDIS_URL = os.environ["REDIS_URL"]
COMPOSIO_USER_ID = os.environ["COMPOSIO_USER_ID"]
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")
JIRA_DEFAULT_COMPONENT = os.environ.get("JIRA_COMPONENT_DEFAULT", "")

# Initialize clients
composio = Composio(api_key=COMPOSIO_API_KEY) if COMPOSIO_API_KEY else Composio()
openai_client = OpenAI()

# Redis search indices
tickets_idx = SearchIndex.from_existing("tickets_idx", redis_url=REDIS_URL)
code_idx = SearchIndex.from_existing("code_idx", redis_url=REDIS_URL)
embedder = OpenAITextVectorizer(model="text-embedding-3-small", api_config={"api_key": "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"})


# Sheet columns
COLS = ["Sprint", "Type", "Title", "Client Impact", "Effort", "Owner", "Status", "Assignee"]

# Priority mapping
PRIORITY_MAP = {
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Critical": 0
}

def get_priority_from_impact(client_impact: str) -> int:
    """Convert client impact to numeric priority"""
    for key in PRIORITY_MAP:
        if key.lower() in client_impact.lower():
            return PRIORITY_MAP[key]
    return 2

def find_latest_sprint(rows):
    """Find the latest sprint from sheet rows"""
    sprints = []
    for row in rows:
        if len(row) > 0 and row[0]:
            sprint_name = row[0].strip()
            if sprint_name and sprint_name not in sprints:
                sprints.append(sprint_name)
    return sprints[-1] if sprints else None


def search_relevant_context(query_text, component=None, k_tickets=5, k_code=3):
    """Search Redis indices for relevant tickets and code"""
    print(f"  🔍 Searching for: {query_text[:100]}...")
    qemb = embedder.embed(query_text)
    # ensure list, not numpy array
    if hasattr(qemb, "tolist"):
        qemb = qemb.tolist()

    # build an optional filter (assumes component is a TAG field; if TEXT, use Text("component")==component)
    comp_filter = (Tag("component") == component) if component else None

    # --- tickets: hybrid text+vector over title | body ---
    tq = HybridQuery(
        text=query_text,
        text_field_name="title|body",               # search both fields
        vector=qemb,
        vector_field_name="embedding",
        num_results=k_tickets,
        return_fields=["id","title","assignee","issuetype","component","body"],
        filter_expression=comp_filter,
        # alpha=0.7  # optional: weight vector (alpha) vs text (1-alpha)
    )
    tickets_hits = tickets_idx.query(tq)            # use .query() with RedisVL Query objects

    # --- code: hybrid over path | body ---
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
    """Generate Jira description using LLM"""
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
    
    # Format relevant code files with more detail
    refs_code = "\n".join([
        f"- {h['path']} ({h['lang']})\n  Preview: {h.get('body', '')[:200]}..."
        for h in code_hits[:3]
    ]) or "No relevant code files found"
    
    prompt = f"""
You are a senior software engineer writing a detailed Jira ticket description.

TICKET TITLE: "{title}"

JIRA FIELDS:
- Type: {itype}
- Priority: {impact}
- Story Points: {effort}
- Sprint: {sprint}
- Reporter: {reporter}
- Assignee: {assignee}

SIMILAR PAST TICKETS (for style reference):
{refs_tickets}

RELEVANT CODE FILES:
{refs_code}

YOUR TASK:
Analyze the codebase and write a comprehensive Jira ticket description that demonstrates deep understanding of the existing architecture.

REQUIRED SECTIONS:

1. **Overview** (2-3 paragraphs)
   - What needs to be done and why
   - Justify the Priority ({impact}) based on impact/urgency

2. **Current Implementation Analysis** ⭐
   - Reference SPECIFIC line numbers from the codebase
   - Describe how current code works
   - Identify relevant files/functions/classes
   - Example: "Currently at line 11 in main.py, storage uses `todos_db = {{}}`"

3. **Proposed Solution**
   - High-level approach
   - Technology choices with justification

4. **Technical Implementation**
   - Break down by file or component
   - Provide concrete code examples following existing patterns
   - Show what needs to change with line number ranges

5. **Files to Modify**
   - List each file: ✏️ modify existing, ➕ create new
   - Brief reason for each

6. **Testing Strategy**
   - Reference existing test patterns
   - Provide concrete test examples
   - Mention fixture updates needed

7. **Backward Compatibility**
   - Will existing functionality break?
   - Migration strategy if needed

8. **Acceptance Criteria** (checkbox format)
   - [ ] Specific, testable criteria
   - [ ] Cover happy path and edge cases

9. **Story Points Justification**
   - Why {effort} points?
   - Reference: files to change, testing complexity, risk level

10. **Related Work** (if applicable)
    - Reference related past tickets

CRITICAL REQUIREMENTS:
✅ MUST include specific line numbers from code
✅ MUST analyze current implementation before proposing changes
✅ MUST provide concrete code examples (not "update the function")
✅ MUST consider cross-file impacts
✅ MUST reference existing patterns in codebase
✅ MUST consider testing implications

❌ AVOID vague statements like "update the API"
❌ AVOID generic advice without code references
❌ AVOID ignoring existing test infrastructure

OUTPUT FORMAT:
```
JIRA METADATA:
Type: {itype}
Priority: {impact}
Story Points: {effort}
Sprint: {sprint}
Reporter: {reporter}
Assignee: {assignee}
Labels: [suggest 2-4 relevant labels like backend, frontend, database, api, bug-fix]

DESCRIPTION:
[Write full ticket description following structure above]
```

Write the ticket now. Be specific, reference actual line numbers, and demonstrate deep codebase understanding.
"""
    
    print(f"  🤖 Generating description with LLM...")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": textwrap.dedent(prompt)}],
    )
    
    return response.choices[0].message.content


async def test_workflow():
    """Test the full workflow except Jira creation"""
    
    print("\n" + "="*80)
    print("🧪 TEST: Sheet Reading + Search + Enrichment")
    print("="*80)
    
    # ===== STEP 1: Read Google Sheet =====
    print("\n📊 STEP 1: Reading Google Sheet...")
    print(f"   Sheet ID: {SPREADSHEET_ID}")
    print(f"   Range: {SHEET_RANGE}")
    
    tools = composio.tools.get(user_id=COMPOSIO_USER_ID, toolkits=["GOOGLESHEETS"])
    
    task_prompt = f"Get all data from spreadsheet {SPREADSHEET_ID}, range {SHEET_RANGE}"
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": task_prompt}],
        tools=tools,
    )
    
    values = None
    if completion.choices[0].message.tool_calls:
        for tool_call in completion.choices[0].message.tool_calls:
            result = composio.tools.execute(
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
        print("❌ No data found in sheet")
        return
    
    # Parse header and rows
    header = [h.strip() for h in values[0]]
    rows = values[1:]
    
    print(f"✅ Sheet read successfully!")
    print(f"   Columns: {header}")
    print(f"   Total rows: {len(rows)}")
    
    # Find latest sprint
    latest_sprint = find_latest_sprint(rows)
    print(f"   Latest sprint: {latest_sprint}")
    
    # Parse rows
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
    
    # Filter for latest sprint
    sprint_tickets = []
    for row in rows:
        parsed = parse_row(row)
        if not parsed["title"].strip():
            continue
        if latest_sprint and parsed["sprint"] != latest_sprint:
            continue
        parsed["priority"] = get_priority_from_impact(parsed["client_impact"])
        sprint_tickets.append(parsed)
    
    print(f"   Tickets in {latest_sprint}: {len(sprint_tickets)}")
    
    # ===== STEP 2: Search & Enrich Each Ticket =====
    print(f"\n🔎 STEP 2: Searching & Enriching {len(sprint_tickets)} Tickets...")
    
    enriched_tickets = []
    
    for i, ticket in enumerate(sprint_tickets, 1):
        print(f"\n{'─'*80}")
        print(f"Ticket {i}/{len(sprint_tickets)}: {ticket['title']}")
        print(f"{'─'*80}")
        print(f"  Type: {ticket['type']}")
        print(f"  Client Impact: {ticket['client_impact']} (Priority: {ticket['priority']})")
        print(f"  Effort: {ticket['effort']}")
        print(f"  Assignee: {ticket['assignee']}")
        
        # Search for relevant context
        query = f"{ticket['title']}\n{ticket['client_impact']}\n{ticket['effort']}"
        tickets_hits, code_hits = search_relevant_context(
            query,
            component=JIRA_DEFAULT_COMPONENT or None,
            k_tickets=5,
            k_code=3
        )
        
        print(f"\n  📋 Found {len(tickets_hits)} relevant past tickets:")
        for j, hit in enumerate(tickets_hits[:3], 1):
            print(f"     {j}. [{hit['id']}] {hit['title']}")
        
        print(f"\n  💻 Found {len(code_hits)} relevant code files:")
        for j, hit in enumerate(code_hits[:3], 1):
            print(f"     {j}. {hit['path']} ({hit['lang']})")
        
        # Generate enriched description
        enriched_desc = generate_enriched_description(ticket, tickets_hits, code_hits)
        
        print(f"\n  📝 Generated Description:")
        print("  " + "─"*76)
        for line in enriched_desc.split('\n')[:15]:  # Show first 15 lines
            print(f"  {line}")
        if len(enriched_desc.split('\n')) > 15:
            print(f"  ... ({len(enriched_desc.split('\n')) - 15} more lines)")
        print("  " + "─"*76)
        
        # Store enriched data
        enriched_tickets.append({
            "ticket_data": ticket,
            "description": enriched_desc,
            "context": {
                "tickets": [{"id": h["id"], "title": h["title"]} for h in tickets_hits[:3]],
                "code": [{"path": h["path"], "lang": h["lang"]} for h in code_hits[:3]]
            }
        })
    
    # ===== STEP 3: Show what would be sent to Jira =====
    print(f"\n{'='*80}")
    print("📤 STEP 3: Data That Would Be Sent to Jira Agent")
    print("="*80)
    
    for i, enriched in enumerate(enriched_tickets, 1):
        ticket = enriched["ticket_data"]
        print(f"\n🎫 Ticket {i}: {ticket['title']}")
        print(f"   Project Key: SPRINT")
        print(f"   Summary: {ticket['title'][:255]}")
        print(f"   Type: {ticket['type']}")
        print(f"   Priority: {['Critical', 'High', 'Medium', 'Low'][min(ticket.get('priority', 2), 3)]}")
        print(f"   Assignee: {ticket['assignee'] or ticket['owner']}")
        print(f"   Labels: sprint-automation, {ticket['sprint']}")
        print(f"   Component: {JIRA_DEFAULT_COMPONENT or 'None'}")
        print(f"   Client Impact: {ticket['client_impact']}")
        print(f"   Effort: {ticket['effort']}")
        print(f"\n   Description Preview:")
        desc_preview = enriched['description'][:300] + "..." if len(enriched['description']) > 300 else enriched['description']
        for line in desc_preview.split('\n'):
            print(f"      {line}")
    
    # Save to file for inspection
    output_file = "test_enriched_tickets.json"
    with open(output_file, 'w') as f:
        json.dump(enriched_tickets, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Test Complete!")
    print(f"   Processed: {len(enriched_tickets)} tickets")
    print(f"   Full data saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_workflow())

