# data/code/add_tags.py
import os, re, warnings
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

load_dotenv()
client = OpenAI()

# Default system prompt and user prompt template (can be overridden by script arguments)
SYSTEM_PROMPT_DEFAULT = "Your primary job is to provide corresponding subject tags based on the content of each book."

PROMPT_TEMPLATE_DEFAULT = """
Using **Traditional Chinese**, based on the core content or plot explored in the book, find 15 to 20 detailed and complete tags (each tag must be a noun phrase within 6 characters).
The tag granularity should be moderate, avoiding labels that are too broad (e.g., "History," "Literature") or too specific (e.g., "Qing Dynasty Qianlong period scholar correspondence style").

Below is the input and a corresponding example:
Title: "The Courage to Be Disliked"
Author: Ichiro Kishimi
Summary: Through a dialogue between a youth and a philosopher, the core ideas of Adlerian psychology are introduced in a simple and profound way...

Example Tags:
#AdlerianPsychology #ContributingToOthers #SelfGrowth #InterpersonalRelationships #TaskSeparation #DialogueFormat #BeliefTransformation #SelfAcceptance #CourageToChallenge #ActionInThePresent #DenialOfBurden #ResponsibilityChoice #SourceOfHappiness #InferiorityAndTranscendence #SocialRelations #SenseOfBelonging

Following the above format, please generate **15 to 20** tags **in Traditional Chinese** based on the following book information:
Title: {title}
Author: {author}
Summary: {summary}
""".strip()


def process_with_ai(
    text: str,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT_DEFAULT},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error occurred during AI processing: {e}"


def generate_topic_labels(
    title: str,
    author: str,
    summary: str,
    model: str = "gpt-4o",
    prompt_tpl: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> list[str]:
    template = (prompt_tpl or PROMPT_TEMPLATE_DEFAULT)
    prompt = template.format(title=title, author=author, summary=summary)

    raw = process_with_ai(prompt, model=model, system_prompt=system_prompt, temperature=temperature)
    raw = re.sub(r"\s+", " ", raw).strip()
    labels = [s.strip() for s in raw.split("#") if s.strip()]
    return labels


def batch_generate_labels(
    books_csv: str = "data/process/processed_books.csv",
    output_file: str = "data/process/十類title_標籤.xlsx",
    start: int = 0,
    end: int = -1,
    save_every: int = 500,
    model: str = "gpt-4o",
    prompt_tpl: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.3,
):
    """
    Batch generate tags from processed_books.csv and save as an Excel file.
    Settable parameters include start/end index, saving frequency, model, and custom prompt/system_prompt/temperature.
    """
    books = pd.read_csv(books_csv)
    books.drop_duplicates(subset=["title"], inplace=True)
    books.reset_index(drop=True, inplace=True)

    if end == -1:
        end = len(books)

    results = pd.DataFrame(columns=["title", "type", "tags"])

    for i in range(start, end):
        title = str(books.loc[i, "title"])
        author = str(books.loc[i, "author"])
        summary = str(books.loc[i, "summary"])
        type_ = str(books.loc[i, "type"])

        # Generate tags (using overrideable template/parameters)
        labels = generate_topic_labels(
            title=title,
            author=author,
            summary=summary,
            model=model,
            prompt_tpl=prompt_tpl,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # format
        labels = re.sub(r"\s+", " ", " ".join(labels)).strip()
        labels_list = [label.strip() for label in labels.split("#") if label.strip()]

        results.loc[i, "title"] = title
        results.loc[i, "type"] = type_
        results.loc[i, "tags"] = ", ".join(labels_list)

        # save
        if (i + 1) % save_every == 0 or (i + 1) == end:
            print(f"Processed {i + 1} book records, saving temporary results...")
            if os.path.exists(output_file):
                old = pd.read_excel(output_file)
                combined = pd.concat([old, results], ignore_index=True)
            else:
                combined = results
            combined.to_excel(output_file, index=False)
            results = pd.DataFrame(columns=["title", "type", "tags"])
            print("Successfully save")

if __name__ == "__main__":
    batch_generate_labels(start=1100)