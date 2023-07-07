# %%
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import torch
from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    Attribute,
    Entity,
    OperationDataset,
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
from causal_checker.datasets.dataset_utils import gen_dataset_family
from functools import partial
import random as rd


NEWS_ARTICLE_ENGLISH = {
    "climate_heroes": """
Title: "Climate Change: The Unsung Heroes"

In an era defined by increasing global temperatures and extreme weather events, the fight against climate change continues on many fronts. While prominent environmentalists and politicians often claim the limelight, behind the scenes, countless unsung heroes have dedicated their lives to combating climate change. This article aims to spotlight the work of these individuals.

At the forefront is M. {NAME1}, a marine biologist who has developed an innovative method for promoting coral reef growth. Given that coral reefs act as carbon sinks, absorbing and storing CO2 from the atmosphere, M. {NAME1}'s work has significant implications for climate mitigation. Despite facing numerous hurdles, M. {NAME1} has consistently pushed forward, driven by an unwavering commitment to oceanic health.

Next, we turn to M. {NAME2}, a climate economist from a small town who has successfully devised a market-based solution to curb industrial carbon emissions. By developing a novel carbon pricing model, M. {NAME2} has enabled a tangible shift toward greener industrial practices. The model has been adopted in several countries, resulting in significant reductions in CO2 emissions. Yet, despite these successes, M. {NAME2}'s work often flies under the mainstream media radar.

Another unsung hero in the climate change battle is M. {NAME3}, a young agricultural scientist pioneering a line of genetically modified crops that can thrive in drought conditions. With changing rainfall patterns threatening food security worldwide, M. {NAME3}'s work is of immense global relevance. However, due to controversy surrounding genetically modified organisms, the contributions of scientists like M. {NAME3} often go unnoticed.

Additionally, the story of M. {NAME4} is worth mentioning. An urban planner by profession, M. {NAME4} has been instrumental in designing green cities with a minimal carbon footprint. By integrating renewable energy sources, promoting public transportation, and creating more green spaces, M. {NAME4} has redefined urban living. While the aesthetics of these cities often capture public attention, the visionary behind them, M. {NAME4}, remains relatively unknown.

Lastly, we have M. {NAME5}, a grassroots activist working tirelessly to protect and restore the forests in her community. M. {NAME5} has mobilized local communities to halt deforestation and engage in extensive tree-planting initiatives. While large-scale afforestation projects often get global recognition, the efforts of community-level heroes like M. {NAME5} remain largely unsung.

The fight against climate change is not a single battle, but a war waged on multiple fronts. Every victory counts, no matter how small. So, as we continue this struggle, let's not forget to appreciate and honor the unsung heroes like M. {NAME1}, M. {NAME2}, M. {NAME3}, M. {NAME4}, and M. {NAME5} who, away from the spotlight, are making a world of difference.""",
    "new_species": """
Title: Hidden Wonders Revealed: New Species Discovered in Unexplored Amazon Rainforest

In a fascinating development that reinforces the boundless mysteries of our planet, a team of distinguished researchers has uncovered a collection of hitherto unknown species in a remote region of the Amazon Rainforest. This discovery represents not just a triumph for science, but also a potent reminder of the inexhaustible wonders of biodiversity.

The team, led by the renowned botanist Dr. {NAME1}, along with the experienced zoologist Dr. {NAME2}, embarked on their journey to this virtually unexplored area of the Amazon nearly a year ago. They were accompanied by the esteemed entomologist Dr. {NAME3}, who is renowned for her expertise in cataloging new insect species. The quartet was completed by the ecologist Dr. {NAME4}, a specialist in rainforest ecosystems, and Dr. {NAME5}, a pioneering geneticist whose revolutionary techniques have changed the landscape of species identification.

Under the deft leadership of Dr. {NAME1}, the team embarked on a thorough survey of this unexplored rainforest region. Dr. {NAME1}’s primary interest lies in uncovering new plant species, and in this mission, he was not disappointed. His trained eyes identified multiple varieties of orchids and a new genus of bromeliads that have adapted to the extreme conditions of the Amazon rainforest in unique ways.

Meanwhile, Dr. {NAME2} made some significant zoological discoveries. Among the most prominent were a new species of tree-dwelling marsupial and a brightly colored bird species that exhibits nocturnal behavior, a rarity among avian creatures. The genetic analyses performed by Dr. {NAME5} confirmed that these creatures indeed represented new additions to the tree of life.

Dr. {NAME3} spent her time meticulously cataloging the diverse insect population in the region. She uncovered several new species of beetles, butterflies, and even a new genus of ants that seem to have a complex social structure. Her findings could alter the way we understand insect social behavior.

Dr. {NAME4} played a crucial role in analyzing the ecosystem of the unexplored region. He provided vital insights into the adaptive strategies and symbiotic relationships of the newly discovered species with their environment. His work aids in developing a holistic understanding of the biodiversity and ecological balance within this region of the Amazon.

The findings by Drs. {NAME1}, {NAME2}, {NAME3}, {NAME4}, and {NAME5} have opened a new chapter in our understanding of Earth's biodiversity. These uncharted regions of the Amazon are a treasure trove of life, and the dedicated work of these scientists is vital to unveiling the hidden mysteries of the planet.

This discovery underscores the importance of protecting and preserving the Amazon Rainforest, home to untold numbers of undiscovered species. As we continue to explore, learn, and discover, we must also remember our role as custodians of this incredible biodiversity and ensure it survives for future generations to explore and appreciate.
""",
    "captains": """
Title: "From Pirates to Naval Heroes: Captains who Shaped Maritime History"

Throughout the annals of maritime history, we have seen the rise of many sea captains who have shaped the contours of the world as we know it today. These stalwarts of the sea were not merely the masters of their ships, but the arbiters of destiny, using the open ocean as a platform to create history and change the world. From swashbuckling pirates to heralded naval heroes, these are the captains who have left an indelible mark on the maritime world.

First, let's take a moment to appreciate the colorful figure of Capt. {NAME1}. Emerging from the shadows of the late 17th century, he was initially a notorious pirate who terrorized the Caribbean's crystalline waters. His name struck fear into the hearts of even the most seasoned seafarers, and he was known for his relentless pursuit of treasure and power. However, as the tides of time turned, so did the course of Capt. {NAME1}'s life. Following a mysterious incident involving a violent storm and the loss of his crew, he decided to put his piratical life behind him. Instead, he became a privateer, aiding the British Empire in its naval conflicts and skirmishes. His cunning tactics and unrivaled knowledge of the Caribbean's treacherous waters proved invaluable, and he became a highly respected figure within the naval ranks.

Our next mariner of note is Capt. {NAME2}, who is remembered as a stalwart defender of the British merchant fleet during the 18th century. His naval career began under the tutelage of Capt. {NAME1}, from whom he learned the ropes of naval warfare and the art of seafaring. Akin to his mentor, Capt. {NAME2} had a knack for turning adverse situations to his advantage. His tenacity and strategic acumen ensured that countless merchant vessels could safely pass through pirate-infested waters, contributing to the growth of British trade and prosperity.

The 19th century heralded the rise of Capt. {NAME3}, an explorer whose maritime expeditions dramatically altered our understanding of the world. Born to humble beginnings in coastal Norway, Capt. {NAME3} broke free from his provincial life, captivated by stories of the sea and its infinite horizons. Commanding a small but resilient ship, he set sail for the unknown. His daring voyages took him to uncharted territories, including the icy expanses of Antarctica and the remote islands of the Pacific. Through his courageous exploration, continents were mapped, and new trade routes were established.

As the age of exploration gave way to the era of empire, the need for fierce and strategic naval leaders surged. Among these leaders, Capt. {NAME4} stood out, commanding the imperial fleet during the height of the 19th-century colonial era. He successfully led numerous naval campaigns, securing vital sea lanes and ensuring the smooth flow of resources from colonies to the empire. His name was etched into the annals of naval history, remembered as a disciplined and inspiring leader.

Last, but certainly not least, is Capt. {NAME5}. As we move into the 20th century, the role of the sea captain evolved in line with technological advancements and the political climate of the time. Capt. {NAME5} was at the helm during this critical period, steering her crew through the tumultuous waters of the World Wars. Her leadership was marked by her brilliant tactical thinking and the respect she commanded from her crew, despite the gender biases prevalent at the time. Her pivotal role in key naval battles won her accolades, and she was instrumental in shaping the future of naval warfare.

In conclusion, each of these captains - Capt. {NAME1}, Capt. {NAME2}, Capt. {NAME3}, Capt. {NAME4}, and Capt. {NAME5} - has significantly shaped maritime history, leaving a legacy that continues to inspire and guide the generations of seafarers who have followed in their wake. From pirates to naval heroes, their stories embody the spirit of the sea: mysterious, untamed, and full of adventure.
""",
}

SENTENCES = {
    "climate_heroes": [
        (
            "un économiste du climat d'une petite ville qui a réussi à concevoir une solution basée sur le marché pour réduire les émissions de carbone industrielles. En développant un modèle innovant de tarification du carbone,",
            "NAME2",
        ),
        (
            "Étant donné que les récifs coralliens agissent comme des puits de carbone, absorbant et stockant le CO2 de l'atmosphère, le travail de",
            "NAME1",
        ),
        (
            "Cependant, en raison de la controverse entourant les organismes génétiquement modifiés, les contributions de scientifiques comme",
            "NAME3",
        ),
        (
            "En intégrant des sources d'énergie renouvelables, en favorisant les transports publics et en créant plus d'espaces verts,",
            "NAME4",
        ),
        (
            "Alors que les projets de reforestation à grande échelle obtiennent souvent une reconnaissance mondiale, les efforts des héros au niveau de la communauté comme",
            "NAME5",
        ),
    ],
    "new_species": [
        (
            """Este descubrimiento representa no solo un triunfo para la ciencia, sino también un poderoso recordatorio de las inagotables maravillas de la biodiversidad.

El equipo, liderado por el renombrado botánico""",
            "NAME1",
        ),
        (
            """Sus entrenados ojos identificaron múltiples variedades de orquídeas y un nuevo género de bromelias que se han adaptado a las condiciones extremas de la selva amazónica de formas únicas.

Mientras tanto,""",
            "NAME2",
        ),
        (
            """Entre los más prominentes se encontraban una nueva especie de marsupial que vive en los árboles y una especie de ave de colores brillantes que exhibe comportamiento nocturno, algo raro entre las criaturas aviares. Los análisis genéticos realizados por""",
            "NAME5",
        ),
        (
            """emprendió su viaje a esta área prácticamente inexplorada de la Amazonía hace casi un año. Estuvieron acompañados por la estimada entomóloga""",
            "NAME3",
        ),
        (
            """Descubrió varias nuevas especies de escarabajos, mariposas e incluso un nuevo género de hormigas que parecen tener una estructura social compleja. Sus hallazgos podrían alterar la forma en que entendemos el comportamiento social de los insectos.
            
            """,
            "NAME4",
        ),
    ],
    "captains": [
        (
            """Von draufgängerischen Piraten bis zu gefeierten Marinehelden, das sind die Kapitäne, die einen unauslöschlichen Eindruck in der maritimen Welt hinterlassen haben.

Zuerst wollen wir einen Moment innehalten, um die farbenfrohe Figur von""",
            "NAME1",
        ),
        (
            """Seine gerissenen Taktiken und sein unübertroffenes Wissen über die tückischen Gewässer der Karibik erwiesen sich als unschätzbar, und er wurde zu einer hoch angesehenen Figur innerhalb der Marine.

Unser nächster Seemann von Bedeutung ist""",
            "NAME2",
        ),
        (
            """ein Entdecker, dessen maritime Expeditionen unser Verständnis von der Welt dramatisch veränderten. Geboren in bescheidenen Verhältnissen an der norwegischen Küste, brach""",
            "NAME3",
        ),
        (
            """Mit dem Übergang vom Zeitalter der Entdeckung zur Ära des Empire stieg der Bedarf an unerschrockenen und strategischen Marineführern. Unter diesen Führern stach""",
            "NAME4",
        ),
        (
            """änderte sich die Rolle des Seefahrerkapitäns im Einklang mit technologischen Fortschritten und dem politischen Klima der Zeit.""",
            "NAME5",
        ),
    ],
}

NAMES = {
    "NAME1": ["Smith", "Johnson", "Williams", "Brown", "Jones"],
    "NAME2": ["Garcia", "Miller", "Davis", "Rodriguez", "Martinez"],
    "NAME3": ["Hernandez", "Lopez", "Gonzalez", "Perez", "Wilson"],
    "NAME4": ["Anderson", "Thomas", "Taylor", "Moore", "Jackson"],
    "NAME5": ["Martin", "Lee", "Walker", "Harris", "Thompson"],
}


TRANSLATION_TEMPLATE = {
    "climate_heroes": """<|endoftext|>
Here is a new article in English. Below, you can find a partial translation in French. Please complete the translation.

English:
{ENGLISH_ARTICLE}

French:
[...]
{FOREIGN_SENTENCE} M.""",
    "new_species": """
Here is a new article in English. Below, you can find a partial translation in Spanish. Please complete the translation.

English:
{ENGLISH_ARTICLE}

Spanish:
[...]
{FOREIGN_SENTENCE} Dr.""",
    "captains": """
Here is a new article in English. Below, you can find a partial translation in German. Please complete the translation.

English:
{ENGLISH_ARTICLE}

German:
[...]
{FOREIGN_SENTENCE} Capt.""",
}


def make_translation_prompt(tokenizer, dataset_name: str):
    names = {k: rd.choice(v) for k, v in NAMES.items()}
    sentence, querried_name = rd.choice(SENTENCES[dataset_name])
    english_article = NEWS_ARTICLE_ENGLISH[dataset_name].format(**names)
    translation_prompt = TRANSLATION_TEMPLATE[dataset_name].format(
        ENGLISH_ARTICLE=english_article, FOREIGN_SENTENCE=sentence
    )

    entities = []
    for name_idx, name in names.items():
        entities.append(
            Entity(
                name=" " + name,
                attributes=[Attribute(value=name_idx, name=str("name_order"))],
                tokenizer=tokenizer,
                only_tokenize_name=True,
            )
        )
    query = Query(
        queried_attribute="name",
        filter_by=[Attribute(value=querried_name, name=str("name_order"))],
    )
    prompt = ContextQueryPrompt(
        model_input=translation_prompt,
        query=query,
        context=entities,
        tokenizer=tokenizer,
    )
    return prompt


def create_translation_retrieval_dataset(
    nb_sample=100, tokenizer=None, dataset_names=None
) -> List[OperationDataset]:
    datasets = []
    if dataset_names is None:
        dataset_names = list(NEWS_ARTICLE_ENGLISH.keys())

    return gen_dataset_family(
        partial(make_translation_prompt, tokenizer),
        dataset_prefix_name="code_type_retrieval",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
    )


# %%

# %%

# %%
