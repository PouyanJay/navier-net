import { Band } from "../../components";
import { RecipeIcon } from "../../components/icons";
import { RECIPES, recipeMeta, type RecipeName } from "./recipes";

interface RecipeBandProps {
  recipe: RecipeName;
  /** Stamps the whole form with the chosen recipe's configuration. */
  onRecipe: (name: RecipeName) => void;
  /** Whether the form has departed from the stamped recipe. */
  modified: boolean;
  locked: boolean;
}

/**
 * The rail's first question, answered before any knob: what should this run be?
 *
 * Each card is a benched configuration with its evidence on the card, not an
 * adjective. Choosing one stamps the form; editing anything afterwards keeps
 * the card selected as the stated baseline, and the launch dock reports each
 * departure -- "full physics, plus nucleation root" is a reproducibility
 * statement, where a card that snapped to "custom" would say nothing.
 */
export function RecipeBand({
  recipe,
  onRecipe,
  modified,
  locked,
}: RecipeBandProps) {
  const meta = recipeMeta(recipe);
  return (
    <Band
      icon={RecipeIcon}
      label="Recipe"
      hint={`${meta.label.toLowerCase()}${modified ? " · modified" : ""}`}
      collapsible
    >
      <div className="recipe-cards" role="radiogroup" aria-label="Recipe">
        {RECIPES.map((option) => {
          const on = option.name === recipe;
          return (
            <button
              key={option.name}
              type="button"
              role="radio"
              aria-checked={on}
              // aria-disabled, not disabled: a locked band still explains
              // itself, and a disabled card would leave the tab order.
              aria-disabled={locked || undefined}
              className={on ? "recipe-card on" : "recipe-card"}
              onClick={() => {
                if (locked) return;
                onRecipe(option.name);
              }}
            >
              <span className="recipe-hd">
                <span className="recipe-nm">{option.label}</span>
                <span className="recipe-ev mono">{option.evidence}</span>
              </span>
              <span className="recipe-detail">{option.detail}</span>
            </button>
          );
        })}
      </div>
      <p className="rail-note">
        {modified
          ? "Changes below are tracked against this recipe and listed above Run. Re-select the card to return to it."
          : "A card stamps the whole form. Anything you change afterwards is kept, and tracked against the card."}
      </p>
    </Band>
  );
}
