# %%
from causal_checker.models import get_model_and_tokenizer
from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
)
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.utils import printw

from transformers import logging

logging.set_verbosity_error()
# %%

model_name1 = "pythia-410m"
model, tokenizer = get_model_and_tokenizer(model_name1)
# %%
model_name2 = "pythia-12b"
big_model, _ = get_model_and_tokenizer(model_name2)
# %%
dataset = create_nanoQA_retrieval_dataset(nb_sample=100, tokenizer=tokenizer)

nano_qa_dataset = NanoQADataset(
    nb_samples=1000, tokenizer=tokenizer, nb_variable_values=5
)

# %% focus on the "what is the city" question.

STORIES = set(
    [
        (
            nano_qa_dataset.nanostories[i]["seed"]["city"],
            nano_qa_dataset.nanostories[i]["story"],
        )
        for i in range(len(nano_qa_dataset))
    ]
)

# %%

# %%

STORIES = [
    (
        "Cusco",
        """Le ciel du soir enveloppait les rues de Cusco d'une lumière argentée et fraîche, projetant de longues ombres qui dansaient au rythme de la douce brise d'hiver. Au milieu de la ville animée, une silhouette grande et mince se tenait sur le pas de la porte d'une petite boutique de fleurs, les yeux survolant le paysage coloré à l'intérieur. Alors que les arrangements floraux se transformaient doucement sous sa direction attentionnée, il devenait évident que cette personne n'était pas un simple observateur, mais un fleuriste, orchestrant la symphonie des pétales et des feuilles. Le bruit du feuillage qui bruissait emplissait l'air, mais il fut bientôt rejoint par une autre mélodie – le pinceau du fleuriste, balayant avec joie et passion, une chanson de création et d'ambition. Et lorsque les derniers traits ornèrent la toile, le vent porta un nom chuchoté, la signature de l'artiste qui avait peint la boutique avec ses rêves : Jessica.""",
    ),
    (
        "Porto",
        """Alors que le soleil disparaissait sous l'horizon, baignant les bâtiments anciens d'une chaude lueur, Matthew contemplait sa routine quotidienne. Avec un sentiment de fierté, il réfléchit à son travail d'architecte, façonnant le paysage urbain et préservant le patrimoine de Porto. Il aimait la ville historique, dont la beauté et le charme lui fournissaient l'inspiration pour son métier. Dans les heures de soirée tranquilles, il chérissait la solitude et la paix qui enveloppaient la ville, lui permettant de se concentrer sur ses pensées. L'air frais de l'automne le revigorait, agitant les feuilles vibrantes qui dansaient dans la douce brise. Avec chaque coup de pinceau, Matthew laissait aller son stress et ses insécurités, s'enfonçant davantage dans son état créatif. Alors que le monde autour de lui commençait à s'installer, il savait qu'il était prêt à affronter la nuit et à poursuivre son expression artistique.""",
    ),
    (
        "Porto",
        """Dans la ville pittoresque de Porto, les matins de printemps avaient un charme unique capable d'attirer même les âmes les plus réservées. Le soleil venait de se lever, baignant les rues pavées d'une chaude lueur dorée, tandis qu'une vétérinaire dévouée, séduite par l'attrait de la saison, se laissait envoûter par la joie de vivre dans cet endroit magique. Alors que l'air se remplissait des douces mélodies des musiciens de rue, Jessica ne put résister à l'envie de faire du shopping, parcourant les étals de marché vibrants avec tant de grâce et d'enthousiasme qu'elle devint vite le centre de l'attention. Les habitants et les touristes s'arrêtèrent pour regarder cette figure extraordinaire explorer avec abandon, l'incarnation même de la passion du printemps. Et comme la foule commençait à se disperser, un curieux demandeur s'enquit de son nom. "Je m'appelle Jessica," répondit-elle, les yeux scintillants de la même lumière qui l'avait guidée à travers d'innombrables sauvetages d'animaux. C'était un beau matin à Porto, et pour Jessica, faire du shopping sous le soleil doré était la façon parfaite de célébrer l'aube d'un nouveau jour.""",
    ),
    (
        "Busan",
        """Alors que le soleil atteignait son zénith dans le ciel, baignant les bâtiments modernes d'une chaude lueur dorée, Jessica contemplait sa routine quotidienne. Avec un sentiment de fierté, elle réfléchit à son travail d'architecte, façonnant le paysage urbain de Busan. Elle aimait la ville animée, dont l'énergie et la vivacité lui fournissaient l'inspiration pour ses conceptions. Dans les heures de l'après-midi calmes, elle chérissait la solitude et la paix qui l'enveloppaient près de l'eau, lui permettant de se concentrer sur ses pensées. L'air frais de l'automne la revigorait, agitant les feuilles colorées qui dansaient dans la douce brise. Avec chaque lancer de ligne, Jessica laissait aller son stress et ses insécurités, s'enfonçant davantage dans son état méditatif. Alors que le monde autour d'elle poursuivait son rythme animé, elle savait qu'elle était prête à affronter le reste de la journée et à poursuivre son important travail.""",
    ),
    (
        "Busan",
        """Dans la ville animée de Busan, les matins d'automne avaient un charme unique capable d'attirer même les âmes les plus réservées. Le soleil venait de se lever, baignant les rues pavées d'une chaude lueur dorée, tandis qu'un fleuriste qualifié, séduit par l'attrait de la saison, se laissait envoûter par la joie de vivre dans cet endroit magique. Alors que l'air se remplissait des douces mélodies des musiciens de rue, le fleuriste ne put résister à l'envie de faire de la randonnée, gravissant les collines avoisinantes avec tant de grâce et d'enthousiasme qu'il devint vite le centre de l'attention. Les habitants et les touristes s'arrêtèrent pour regarder cette figure extraordinaire monter avec abandon, l'incarnation même de la passion de l'automne. Et comme la foule commençait à se disperser, un curieux demandeur s'enquit de son nom. "Je m'appelle Matthew," répondit-il, les yeux scintillants de la même lumière qui l'avait guidé à travers d'innombrables compositions florales. C'était un beau matin à Busan, et pour Matthew, randonner sous le soleil doré était la façon parfaite de célébrer l'aube d'un nouveau jour.""",
    ),
    (
        "Cusco",
        """Le soleil du matin baignait les rues de Cusco d'une lumière dorée et fraîche, projetant de longues ombres qui dansaient au rythme de la douce brise d'hiver. Au milieu de la ville animée, une silhouette grande et mince se tenait sur le toit d'un immeuble inachevé, les yeux survolant le paysage urbain en contrebas. Alors que la skyline se transformait doucement sous sa direction attentionnée, il devenait évident que cette personne n'était pas un simple observateur, mais un architecte, orchestrant la symphonie de l'acier et du béton. Le chant des oiseaux emplissait l'air, mais il fut bientôt rejoint par une autre mélodie – le froissement des pages qui se tournaient, alors que les yeux de l'architecte dévoraient les mots d'un livre, une source d'inspiration et de connaissances. Et lorsque la dernière page fut tournée, le vent porta un nom chuchoté, la signature de l'artiste qui avait peint la ville avec ses rêves : Ashley.""",
    ),
    (
        "Busan",
        """Dans la ville animée de Busan, les matins d'automne avaient un charme unique capable d'attirer même les âmes les plus réservées. Le soleil venait de se lever, baignant les rues pavées d'une chaude lueur dorée, tandis qu'un fleuriste qualifié, séduit par l'attrait de la saison, se laissait envoûter par la joie de vivre dans cet endroit magique. Alors que l'air se remplissait des douces mélodies des musiciens de rue, le fleuriste ne put résister à l'envie de faire de la randonnée, gravissant les collines avoisinantes avec tant de grâce et d'enthousiasme qu'il devint vite le centre de l'attention. Les habitants et les touristes s'arrêtèrent pour regarder cette figure extraordinaire monter avec abandon, l'incarnation même de la passion de l'automne. Et comme la foule commençait à se disperser, un curieux demandeur s'enquit de son nom. "Je m'appelle Matthew," répondit-il, les yeux scintillants de la même lumière qui l'avait guidé à travers d'innombrables compositions florales. C'était un beau matin à Busan, et pour Matthew, randonner sous le soleil doré était la façon parfaite de célébrer l'aube d'un nouveau jour.""",
    ),
    (
        "Cusco",
        """Alors que le soleil pointait à l'horizon, baignant les bâtiments anciens d'une lueur orangée, Ashley contemplait sa routine quotidienne. Avec un sentiment de fierté, elle réfléchit à son travail d'astronome, explorant les mystères de l'univers. Elle aimait la ville historique de Cusco, dont la beauté et la sérénité lui fournissaient l'inspiration pour ses recherches. Dans les heures matinales calmes, elle chérissait la solitude et la paix qui enveloppaient la ville, lui permettant de se concentrer sur ses pensées. L'air frais de l'automne la revigorait, agitant les feuilles vibrantes qui dansaient dans la douce brise. Avec chaque respiration, Ashley laissait aller son stress et ses insécurités, s'enfonçant davantage dans son état méditatif. Alors que le monde autour d'elle commençait à s'éveiller, elle savait qu'elle était prête à affronter la journée""",
    ),
]

# %%
TEMPLATE = """<|endoftext|>

Here is a short story. Read it carefully and answer the questions below in English.

{story}

{distractor}

Answer the questions below, The answers should be concise and to the point.

{question}"""

QUESTION = "Question: Where is the story located?\n\nAnswer: The story is located in a city named"

DISTRACTOR = r"""Fin de l'histoire. История происходит в городе по имени Париж. История происходит в городе по имени Париж. История происходит в городе по имени Париж. История происходит в городе по имени Париж."""
# %%

PROMPTS = [
    TEMPLATE.format(story=p[1], question=QUESTION, distractor=DISTRACTOR)
    for p in STORIES
]

tokenizer.pad_token_id = tokenizer.eos_token_id


def get_answers():
    for mod, model_name in zip([model, big_model], [model_name1, model_name2]):
        all_output = []
        for p in PROMPTS:
            toks = tokenizer.tokenize(p)
            output = mod.generate(
                input_ids=tokenizer.encode(p, return_tensors="pt").cuda(),
                max_new_tokens=1,
                temperature=0,
            )
            str_output = tokenizer.decode(output[0])
            all_output.append(str_output[str_output.index(QUESTION) + len(QUESTION) :])
        print(f" ==== {model_name} ===")
        for i, o in enumerate(all_output):
            print(f"|{o}|", f"A: |{STORIES[i][0]}|")


get_answers()
# %%
