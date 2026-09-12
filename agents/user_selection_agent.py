from utils.state import RecipeState


class UserSelectionAgent:
    """
    Presents the generated recipes to the user and allows selection of one recipe for further modification or finalization.
    Updates the workflow state with the selected recipe.

    This agent no longer performs blocking terminal I/O itself, so it can be
    reused unchanged by both the CLI (`main.py`/`workflow.py`) and the Flask
    web app (`app.py`). Callers are responsible for obtaining the user's
    choice (e.g. via `input()` on the CLI, or from an HTTP request in the
    web app) and passing it in as `user_choice`.
    """

    def select_recipe(self, state: RecipeState, user_choice: str = None) -> RecipeState:
        """
        Validate `user_choice` against the available recipes and update state.

        Raises:
            ValueError: if `user_choice` is missing, not an integer, or out of range.
        """
        if user_choice is None:
            raise ValueError("user_choice must be provided to select_recipe")

        num_recipes = len(state.chef_output.recipes)
        try:
            choice = int(user_choice)
        except (TypeError, ValueError):
            raise ValueError("Invalid input. Please enter a valid number.")

        if not (1 <= choice <= num_recipes):
            raise ValueError(f"Please enter a number between 1 and {num_recipes}.")

        state.selected_recipe = state.chef_output.recipes[choice - 1]
        state.selected_nutrition_info = (
            state.nutrition_info.recipes[choice - 1] if state.nutrition_info else None
        )
        return state

    def prompt_and_select_cli(self, state: RecipeState) -> RecipeState:
        """CLI helper: repeatedly prompts via input() until a valid choice is made."""
        print("\nPlease select a recipe by entering the corresponding number:")
        for idx, recipe in enumerate(state.chef_output.recipes, 1):
            print(f"{idx}. {recipe.name}")
        while True:
            try:
                return self.select_recipe(state, input("Enter your choice (number): "))
            except ValueError as e:
                print(str(e))
