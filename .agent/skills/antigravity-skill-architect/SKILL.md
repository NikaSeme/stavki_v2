---
name: antigravity-skill-architect
description: Use when generating, reviewing, or improving custom AI skills to ensure they perfectly match the native Antigravity structural quality, formatting, and assertive tone.
---

# Antigravity Skill Architect

## Overview
A custom skill is useless if the AI agent cannot parse it authoritatively. Native Antigravity skills (like `systematic-debugging`) do not just list rules; they define an architectural mindset. 

When you generate or improve a custom skill, you MUST apply this exact architectural blueprint. Failing to do so results in robotic, passive skills that are easily overwritten by competing context.

## The Iron Law
```
NO CUSTOM SKILL MAY BE SAVED WITHOUT RICH MARKDOWN VISUALS, CODE SNIPPETS ALIGNING "GOOD" VS "BAD", AND AN AUTHORITATIVE "WHEN TO USE" TRIGGER.
```

## 1. When to Use This Meta-Skill
- When a user asks you to "create a new skill" for a specific project.
- When an existing skill feels too vague or is failing to constrain the AI's behavior.
- When migrating basic checklists into the permanent `~/.gemini/antigravity/skills/` registry.

## 2. The Structural Blueprint (Mandatory)

Every high-quality skill must contain the following structural hierarchy:

### Phase A: The Frontmatter & Trigger
- **YAML Block:** Must start with `---`, contain a `name:`, and a `description:` that explicitly triggers the LLM (e.g., "Use when encountering...").
- **Core Principle:** Immediately following the title, define the absolute philosophical rule of the skill in bold or inside a fenced code block (The "Iron Law").

### Phase B: Trigger Conditions ("When to Use")
Do not assume the AI knows when to activate the skill. You must provide a bulleted list:
- **Use this ESPECIALLY when:** (List 3-4 exact scenarios).
- **Don't skip when:** (Preempt common AI excuses for laziness).

### Phase C: Visually Rich Execution (The Pattern)
Do not just list text steps. You must use Markdown to create visual anchors for the execution engine:
- Use `###` headers for distinct phases (e.g., `Phase 1: Diagnostics`, `Phase 2: Execution`).
- **Use Code Blocks heavily:** Provide ````python` or ````bash` snippets. 
- **Compare States:** Show explicitly what `❌ Bad` looks like versus `✅ Good`. Native skills teach by contrast.

### Phase D: Constraints & Red Flags
Integrate the user's "Strict Prohibitions".
- Create a specific section titled `## Red Flags - STOP and Follow Process`.
- List catastrophic thoughts the AI might have (e.g., "If you catch yourself thinking 'I'll just impute with zero', STOP.")

### Phase E: Quick Reference Tables
Native scripts consolidate complex logic. Always end the skill with a Markdown table summarizing the Phases, Key Activities, and Success Criteria.

## 3. Tone & Persona
- **Be Authoritative:** Use imperative verbs (`DO NOT`, `MUST`, `ALWAYS`, `NEVER`).
- **No Apologies:** The skill is a machine constraint. It should not contain conversational filler like "Please try to..." or "It would be best if...".
- **Self-Correction:** Build in explicit failure loops. (e.g., "If the test fails 3 times, STOP. You have an architectural problem.")

## 4. Evaluation Checklist
Before declaring the newly improved skill "complete", audit it against this table:

| Requirement | Evidence in Generated Skill |
|-------------|----------------------------|
| **YAML Trigger** | Does the description start with "Use when..."? |
| **Code Examples** | Are there concrete `✅ Good` vs `❌ Bad` code blocks? |
| **Markdown Richness** | Are tables, bold emphasis, and nested bullets utilized? |
| **Red Flags** | Are common AI rationalizations preemptively blocked? |
| **Testable Exit** | Is there a tangible command/script to prove the task succeeded? |
