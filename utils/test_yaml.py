# This file contains all the fixed YAML definitions used across the agents.

fixed_yaml_definitions = {
    "chef_agent": """
recipes:
  - name: "Test Recipe 1"
    description: "A simple test recipe."
    ingredients:
      - "ingredient1"
      - "ingredient2"
    instructions:
      - "Step 1: Do something."
      - "Step 2: Do something else."
    prep_time: "10 min"
    cook_time: "20 min"
    servings: 2
    tags:
      - "test"
      - "vegan"
  - name: "Test Recipe 2"
    description: "Another test recipe."
    ingredients:
      - "ingredient3"
      - "ingredient4"
    instructions:
      - "Step 1: Prep."
      - "Step 2: Cook."
    prep_time: "5 min"
    cook_time: "15 min"
    servings: 1
    tags:
      - "test"
      - "gluten-free"
""",
    "final_recipe_agent": """
      name: "Test Final Recipe"
      description: "A test final recipe."
      ingredients:
        - "ingredient1"
        - "ingredient2"
      instructions:
        - "Step 1: Test."
        - "Step 2: Test more."
      tips:
        - "Test tip 1."
        - "Test tip 2."
      serving_suggestions:
        - "Serve with test."
        - "Serve chilled."
      nutrition:
        calories: 100
        protein: 5.0
        fat: 2.0
        carbs: 10.0
        fiber: 1.0
        sodium: 0.5
""",
    "human_escalation_agent": """
human_notes: "Test escalation: Please review the test context and proceed."
""",
    "ingredient_input_agent": """
ingredients:
    - test_ingredient1
    - test_ingredient2
dietary_preferences:
    - vegan
allergies:
    - peanuts
cuisine: test_cuisine
flavor_profiles:
    - spicy
ambiguous_ingredients:
    - test_ambiguous
normalized_ingredients:
    - test_ingredient1
    - test_ingredient2
""",
    "nutrition_agent": """
recipes:
  - name: "Test Recipe"
    nutrition_profile:
      calories: 123
      protein: 4.5
      fat: 2.1
      carbs: 10.0
      fiber: 1.0
      sodium: 0.5
    suggested_addons:
      - "addon1"
    nutrition_insights:
      - "insight1"
  - name: "Test Recipe 2"
    nutrition_profile:
      calories: 200
      protein: 8.0
      fat: 5.0
      carbs: 30.0
      fiber: 2.5
      sodium: 1.2
    suggested_addons:
      - "addon2"
      - "addon3"
    nutrition_insights:
      - "insight2"
      - "insight3"
""",
    "sentiment_escalation_agent": """
sentiment: "test-sentiment"
escalation_required: true
""",
}
