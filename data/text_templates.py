from typing import Dict, List, Optional
import random
import pandas as pd


class TitanicTextTemplates:
    def __init__(self):
        self.templates = [
            # Basic template
            "A {age_desc} {sex} passenger traveling in {class_desc} class. "
            "{family_desc} {fare_desc} {embark_desc}",
            # Narrative style
            "{name_info} was a {age_desc} {sex} who boarded the Titanic {embark_desc}. "
            "Traveling in {class_desc} class, {pronoun} paid {fare} for the journey. {family_desc}",
            # Descriptive style
            "Passenger details: {sex}, {age} years old, {class_desc} class ticket. "
            "Embarked from {embarked_full}. {family_desc} Fare paid: ${fare}.",
            # Story style
            "On that fateful voyage, {a_an} {age_desc} {sex} {embark_desc} "
            "with a {class_desc} class ticket costing ${fare}. {family_desc}",
        ]

        self.embarkation_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}

    def get_age_description(self, age: Optional[float]) -> str:
        if age is None or pd.isna(age):
            return "passenger of unknown age"
        elif age < 1:
            return "infant"
        elif age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 30:
            return "young adult"
        elif age < 50:
            return "middle-aged"
        elif age < 65:
            return "mature"
        else:
            return "elderly"

    def get_class_description(self, pclass: int) -> str:
        class_map = {
            1: "luxurious first",
            2: "comfortable second",
            3: "economical third",
        }
        return class_map.get(pclass, f"{pclass}th")

    def get_family_description(self, sibsp: int, parch: int) -> str:
        total_family = sibsp + parch
        if total_family == 0:
            return "They traveled alone."
        elif sibsp > 0 and parch > 0:
            return f"They traveled with {sibsp} sibling(s)/spouse(s) and {parch} parent(s)/child(ren)."
        elif sibsp > 0:
            return f"They traveled with {sibsp} sibling(s) or spouse."
        else:
            return f"They traveled with {parch} parent(s) or child(ren)."

    def get_fare_description(self, fare: float) -> str:
        if pd.isna(fare):
            return "The fare is unknown."
        elif fare < 10:
            return f"They paid a modest fare of ${fare:.2f}."
        elif fare < 30:
            return f"They paid a moderate fare of ${fare:.2f}."
        elif fare < 100:
            return f"They paid a substantial fare of ${fare:.2f}."
        else:
            return f"They paid a premium fare of ${fare:.2f}."

    def get_embarkation_description(self, embarked: str) -> str:
        port = self.embarkation_map.get(embarked, "an unknown port")
        return f"embarked at {port}"

    def get_name_info(self, name: str) -> str:
        # Extract title from name
        if ", " in name and ". " in name:
            title = name.split(", ")[1].split(". ")[0]
            return f"{title}. {name.split(', ')[0]}"
        return "The passenger"

    def row_to_text(self, row: Dict, template_idx: Optional[int] = None) -> str:
        if template_idx is None:
            template_idx = random.randint(0, len(self.templates) - 1)

        template = self.templates[template_idx]

        # Prepare all descriptions
        age_desc = self.get_age_description(row.get("Age"))
        class_desc = self.get_class_description(row["Pclass"])
        family_desc = self.get_family_description(row["SibSp"], row["Parch"])
        fare_desc = self.get_fare_description(row["Fare"])
        embark_desc = self.get_embarkation_description(row.get("Embarked", "S"))
        name_info = self.get_name_info(row["Name"])

        # Gender-specific pronouns
        pronoun = "he" if row["Sex"] == "male" else "she"
        a_an = "an" if age_desc[0] in "aeiou" else "a"

        # Format the template
        text = template.format(
            age_desc=age_desc,
            sex=row["Sex"],
            class_desc=class_desc,
            family_desc=family_desc,
            fare_desc=fare_desc,
            embark_desc=embark_desc,
            embarked_full=self.embarkation_map.get(
                row.get("Embarked", "S"), "unknown port"
            ),
            age=row.get("Age", "unknown"),
            fare=row["Fare"],
            pronoun=pronoun,
            a_an=a_an,
            name_info=name_info,
        )

        return text

    def augment_text(self, text: str) -> List[str]:
        augmented = [text]

        # Add historical context
        augmented.append(f"During the Titanic's maiden voyage in 1912: {text}")

        # Add outcome context (for training data)
        augmented.append(f"On the night of April 14-15, 1912: {text}")

        return augmented
