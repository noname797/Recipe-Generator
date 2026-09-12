from utils.state import RecipeState


class UserSelectionAgent:
    """
    Presents the generated recipes to the user and allows selection of one recipe for further modification or finalization.
    Updates the workflow state with the selected recipe.
    """

    def select_recipe(self, state: RecipeState) -> RecipeState:
        print("\nPlease select a recipe by entering the corresponding number:")
        for idx, recipe in enumerate(state.chef_output.recipes, 1):
            print(f"{idx}. {recipe.name}")
        while True:
            try:
                choice = int(input("Enter your choice (number): "))
                if 1 <= choice <= len(state.chef_output.recipes):
                    state.selected_recipe = state.chef_output.recipes[choice - 1]
                    state.selected_nutrition_info = state.nutrition_info.recipes[
                        choice - 1
                    ]
                    break
                else:
                    print(
                        f"Please enter a number between 1 and {len(state.chef_output.recipes)}."
                    )
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        return state
