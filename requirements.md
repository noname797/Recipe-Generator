
## 🎯 Project Overview
The **Multi-Agent Recipe Generator** is a sophisticated LangGraph-based application that leverages multiple AI agents to create personalized recipe recommendations

Agent Roles and Responsibilities

1. Ingredient Input Agent:LLM Handles the initial collection of user-provided ingredients, dietary preferences, allergies, cuisine, and flavor profiles. It also manages ambiguity resolution for ingredient inputs, ensuring downstream agents receive precise and actionable data.

2. Chef Agent: LLM Generates multiple recipe suggestions based on the normalized ingredients and user preferences. This agent is responsible for maximizing ingredient usage, aligning with dietary and flavor constraints, and providing detailed metadata for each recipe option.

3. Nutrition Agent: LLM Analyzes the nutritional profile of each recipe, suggesting common add-ons and new ingredients to enhance nutritional value. It synthesizes user preferences with recipe content to deliver actionable nutrition insights.

4. User Selection Agent: User selects which recipe to choose

5. User Modification Agent: Processes specific nutritional adjustment requests and ingredient modification requests.  Modifying recipes and updating nutritional profiles accordingly. It ensures that user-driven changes are reflected in both the recipe and the overall workflow state.

6. Final Recipe Agent: Synthesizes all prior outputs into a complete, step-by-step recipe, including ingredients, instructions, tips, and serving suggestions, while integrating nutritional data.

7. Sentiment & Escalation Agent: LLM Monitors user responses and sentiment across interactions; if frustration or urgency detected, flips escalation_required flag for human handoff.

8. Human Escalation Agent: Packages context and suggested responses for a human reviewer/operator UI to review and continue or override.

Prompt Format (Output instructions must be JSON dump of expected output class injected programatically)

[INSTRUCTIONS]

{intructions for the model}

[OUTPUT INSTRUCTIONS]

{output instructions}

Basic rules to skip preamble, sticking to output schema, avoid hallucinations etc.
