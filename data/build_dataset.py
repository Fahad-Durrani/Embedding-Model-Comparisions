"""Deterministic synthetic-dataset generator for the embedding benchmark.

Hand-authored (no LLM / no network) so it is reproducible and reviewable.
Produces ``dataset.json`` with three parts:

* ``passages`` : the corpus of chunks to retrieve from.
* ``queries``  : test queries, each tagged with a ``category``.
* qrels are embedded per query as ``{passage_id: grade}`` where
  grade 2 = highly relevant, 1 = partially relevant, absent = irrelevant.

The corpus plants several *confusable pairs* (Python language vs. Python snake,
Java language vs. Java island, river bank vs. financial bank, cloud computing
vs. weather cloud) so the edge-case queries can probe semantic vs. lexical
discrimination.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "dataset.json")

# --------------------------------------------------------------------------- #
# Corpus  (mixed general knowledge, English)
# --------------------------------------------------------------------------- #
PASSAGES = [
    ("p01", "science", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food from carbon dioxide and water. It takes place in the chloroplasts and releases oxygen as a by-product."),
    ("p02", "science", "The water cycle describes how water evaporates from oceans and lakes, condenses into clouds, and falls back to the surface as rain or snow. This continuous movement redistributes fresh water across the planet."),
    ("p03", "science", "The human heart is a muscular organ that pumps blood through the circulatory system. Each beat pushes oxygen-rich blood to the body's tissues and returns oxygen-poor blood to the lungs."),
    ("p04", "science", "DNA, or deoxyribonucleic acid, carries the genetic instructions used in the growth and functioning of living organisms. Its double-helix structure stores hereditary information passed from parents to offspring."),
    ("p05", "science", "Gravity is the force of attraction between masses. Isaac Newton described how this force keeps planets in orbit around the Sun and makes objects fall toward the Earth."),
    ("p06", "science", "According to Albert Einstein's theory of relativity, the speed of light in a vacuum is constant at about 300,000 kilometres per second and nothing can travel faster than it."),
    ("p07", "science", "Vaccines train the immune system to recognise and fight specific pathogens. By exposing the body to a harmless piece of a virus, they prepare defenders that respond quickly to a real infection."),
    ("p08", "science", "Global warming is largely driven by greenhouse gases such as carbon dioxide and methane, which trap heat in the atmosphere. Rising average temperatures are causing shifts in weather patterns and sea levels."),
    ("p09", "space", "On 20 July 1969, the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above."),
    ("p10", "space", "Mars is the fourth planet from the Sun and is often called the Red Planet because of its iron-oxide dust. Robotic rovers have explored its surface searching for signs of past water."),
    ("p11", "space", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their lives."),
    ("p12", "space", "NASA, the National Aeronautics and Space Administration, is the United States government agency responsible for the civilian space programme and aeronautics research."),
    ("p13", "history", "World War II ended in 1945 after six years of global conflict. Victory in Europe was declared in May and the war in the Pacific concluded in September following Japan's surrender."),
    ("p14", "history", "The Roman Empire was one of the largest empires in ancient history, ruling much of Europe, North Africa, and the Middle East. It was known for its roads, aqueducts, and system of law."),
    ("p15", "history", "The Great Wall of China was built over many centuries to defend Chinese states against invasions and raids from northern nomadic groups. It stretches thousands of kilometres across northern China."),
    ("p16", "history", "The French Revolution began in 1789 and overthrew the monarchy in France. It introduced ideas of liberty, equality, and fraternity that reshaped European politics."),
    ("p17", "history", "The Industrial Revolution was a period of rapid technological and economic change beginning in the late 18th century. Steam power and mechanised factories transformed how goods were produced."),
    ("p18", "geography", "The Amazon rainforest in South America is the largest tropical rainforest on Earth. It is home to an enormous diversity of plants and animals and produces a significant share of the world's oxygen."),
    ("p19", "geography", "Mount Everest, on the border of Nepal and Tibet, is the highest mountain on Earth, rising about 8,849 metres above sea level. Climbers face thin air and extreme cold near its summit."),
    ("p20", "geography", "The Sahara is the largest hot desert in the world, covering much of North Africa. Its landscape ranges from vast sand dunes to rocky plateaus, with very little rainfall."),
    ("p21", "geography", "The Nile is a major river in northeastern Africa that flows northward into the Mediterranean Sea. For thousands of years it has supported agriculture and civilisation along its banks."),
    ("p22", "geography", "The Pacific Ocean is the largest and deepest of Earth's oceans, stretching from Asia and Australia to the Americas. It contains the Mariana Trench, the deepest known point in the sea."),
    # --- confusable pairs ---------------------------------------------------
    ("p23", "tech", "Python is a popular high-level programming language known for its readable syntax. It was named after the British comedy group Monty Python and is widely used for data science and web development."),
    ("p24", "nature", "The reticulated python is one of the world's longest snakes, a non-venomous reptile that kills prey by constriction. These snakes are found in the rainforests of Southeast Asia."),
    ("p25", "tech", "Java is a widely used object-oriented programming language designed to run on many platforms through the Java Virtual Machine. It powers everything from Android apps to large enterprise systems."),
    ("p26", "geography", "Java is a densely populated island of Indonesia and home to the capital city Jakarta. Its volcanic soil supports intensive rice farming across terraced fields."),
    ("p27", "geography", "The bank of a river is the sloping land alongside the water channel. River banks can erode over time and are often lined with vegetation that stabilises the soil."),
    ("p28", "finance", "A bank is a financial institution that accepts deposits from customers and lends money. People visit a bank to open accounts, withdraw cash, and apply for loans or mortgages."),
    ("p29", "tech", "Cloud computing delivers computing services such as storage and processing over the internet. Instead of owning hardware, organisations rent resources from providers and scale them on demand."),
    ("p30", "science", "A cloud is a visible mass of water droplets or ice crystals suspended in the atmosphere. Clouds form when moist air rises and cools, and they often signal approaching rain."),
    # --- more tech / everyday ----------------------------------------------
    ("p31", "tech", "Machine learning is a branch of artificial intelligence in which computer systems learn patterns from data rather than following explicit rules. Trained models can then make predictions on new inputs."),
    ("p32", "tech", "The World Wide Web is a system of interlinked documents and resources accessed over the internet using web browsers. It was invented by Tim Berners-Lee around 1989."),
    ("p33", "food", "Coffee is a popular brewed drink prepared from roasted coffee beans. It contains caffeine, a natural stimulant that many people rely on to feel alert in the morning."),
    ("p34", "food", "Tea is an aromatic beverage made by pouring hot water over cured leaves of the tea plant. Depending on the type, it contains varying amounts of caffeine and antioxidants."),
    ("p35", "games", "Chess is a two-player strategy board game played on an eight-by-eight grid. Each player commands sixteen pieces and aims to checkmate the opponent's king."),
    ("p36", "sports", "The modern Olympic Games are an international sporting event held every four years, revived in 1896. Athletes from around the world compete across a wide range of summer and winter disciplines."),
    ("p37", "tech", "Electric cars are powered by rechargeable batteries instead of petrol or diesel engines. They produce no tailpipe emissions and are seen as a key part of reducing transport pollution."),
    ("p38", "science", "Solar panels convert sunlight directly into electricity using photovoltaic cells. As a renewable energy source, they help households and businesses cut their reliance on fossil fuels."),
    ("p39", "environment", "Recycling is the process of collecting and reprocessing used materials such as paper, glass, and plastic into new products. It reduces waste sent to landfill and conserves raw resources."),
    ("p40", "nature", "Honeybees are insects that pollinate flowering plants as they collect nectar and pollen. This pollination is essential for many crops and for the reproduction of countless wild plants."),
    ("p41", "nature", "Sharks are a group of cartilaginous fish with a powerful sense of smell and rows of replaceable teeth. Despite their fearsome reputation, most species pose little threat to humans."),
    ("p42", "nature", "The Great Barrier Reef off the coast of Australia is the world's largest coral reef system. It hosts thousands of species of fish, coral, and other marine life but is threatened by warming seas."),
    ("p43", "art", "The Mona Lisa is a famous portrait painted by the Italian Renaissance artist Leonardo da Vinci. It is celebrated for the subject's enigmatic smile and now hangs in the Louvre in Paris."),
    ("p44", "art", "William Shakespeare was an English playwright and poet regarded as one of the greatest writers in the language. His works include Hamlet, Macbeth, and Romeo and Juliet."),
    ("p45", "math", "Pi is the ratio of a circle's circumference to its diameter, approximately equal to 3.14159. It is an irrational number whose digits continue forever without repeating."),
    ("p46", "math", "The Pythagorean theorem states that in a right-angled triangle the square of the hypotenuse equals the sum of the squares of the other two sides, written as a squared plus b squared equals c squared."),
    ("p47", "geography", "Mount Kilimanjaro in Tanzania is the highest mountain in Africa, rising about 5,895 metres. It is a dormant volcano famous for the snow near its summit despite its equatorial location."),
    ("p48", "geography", "Debate continues over the world's longest river: the Nile in Africa is traditionally listed at about 6,650 kilometres, while some surveys argue the Amazon in South America may be slightly longer."),
    ("p49", "science", "Caffeine is a natural stimulant that acts on the central nervous system, reducing tiredness and increasing alertness. Consuming too much can cause restlessness, a faster heartbeat, and difficulty sleeping."),
    ("p50", "history", "In 1903 the Wright brothers achieved the first controlled, powered, sustained flight of a heavier-than-air aircraft near Kitty Hawk, North Carolina, launching the age of aviation."),
]

# --------------------------------------------------------------------------- #
# Queries + graded relevance judgments (qrels)
# grade: 2 = highly relevant, 1 = partially relevant
# --------------------------------------------------------------------------- #
QUERIES = [
    # ---- exact_paraphrase --------------------------------------------------
    ("q01", "exact_paraphrase", "How do plants make their own food using sunlight?", {"p01": 2}),
    ("q02", "exact_paraphrase", "When did humans first walk on the Moon?", {"p09": 2}),
    ("q03", "exact_paraphrase", "What is the tallest mountain on Earth?", {"p19": 2}),
    ("q40", "exact_paraphrase", "Why was the Great Wall of China built?", {"p15": 2}),

    # ---- semantic (same meaning, different vocabulary) ---------------------
    ("q04", "semantic", "What organ pushes blood around the body?", {"p03": 2}),
    ("q05", "semantic", "Why is the planet heating up?", {"p08": 2}),
    ("q06", "semantic", "Which coding language was named after a comedy troupe?", {"p23": 2}),
    ("q07", "semantic", "How can we generate electricity from the sun?", {"p38": 2}),
    ("q39", "semantic", "Which insects help flowers reproduce?", {"p40": 2}),

    # ---- partial_related (topical, only partially answers) -----------------
    ("q08", "partial_related", "What does caffeine do to the body?", {"p49": 2, "p33": 1, "p34": 1}),
    ("q09", "partial_related", "Famous paintings by Italian artists", {"p43": 2}),
    ("q10", "partial_related", "Tall mountains in Africa", {"p47": 2}),
    ("q11", "partial_related", "What is the longest river in the world?", {"p48": 2, "p21": 1, "p18": 1}),

    # ---- no_match (nothing relevant in the corpus) -------------------------
    ("q12", "no_match", "Best recipe for chocolate chip cookies", {}),
    ("q13", "no_match", "How do I change a flat car tire?", {}),
    ("q14", "no_match", "What is the current stock price of Apple?", {}),
    ("q15", "no_match", "Explain the rules of cricket batting", {}),
    ("q16", "no_match", "How do I house-train a puppy?", {}),

    # ---- edge_negation -----------------------------------------------------
    ("q17", "edge_negation", "A programming language that is not Python", {"p25": 2}),
    ("q18", "edge_negation", "Common animals that are not mammals", {"p41": 2, "p40": 2, "p24": 2}),
    ("q19", "edge_negation", "Hot drinks that contain no caffeine", {}),

    # ---- edge_lexical_trap (keyword overlap, different meaning) -------------
    ("q20", "edge_lexical_trap", "Using the Python programming language for data science", {"p23": 2}),
    ("q21", "edge_lexical_trap", "The grassy bank at the edge of a river", {"p27": 2}),
    ("q22", "edge_lexical_trap", "Where can I deposit money and open an account?", {"p28": 2}),
    ("q23", "edge_lexical_trap", "The island of Java in Indonesia", {"p26": 2}),
    ("q24", "edge_lexical_trap", "Dark clouds gathering in the sky before rain", {"p30": 2}),

    # ---- edge_numeric (dates / numbers / entities) -------------------------
    ("q25", "edge_numeric", "What major event happened in 1969?", {"p09": 2}),
    ("q26", "edge_numeric", "What is the value of pi to five decimal places?", {"p45": 2}),
    ("q27", "edge_numeric", "Which world war ended in 1945?", {"p13": 2}),
    ("q28", "edge_numeric", "In what year was the first powered airplane flight?", {"p50": 2}),

    # ---- edge_typo ---------------------------------------------------------
    ("q29", "edge_typo", "how does fotosynthesis wrok in plants", {"p01": 2}),
    ("q30", "edge_typo", "the tallest montain in the wrld", {"p19": 2}),
    ("q31", "edge_typo", "explain the pythagorean theorm", {"p46": 2}),

    # ---- edge_acronym ------------------------------------------------------
    ("q32", "edge_acronym", "What is NASA?", {"p12": 2}),
    ("q33", "edge_acronym", "The American government space agency", {"p12": 2}),
    ("q34", "edge_acronym", "What does DNA do in living things?", {"p04": 2}),

    # ---- edge_ambiguous (several valid relevant passages) ------------------
    ("q35", "edge_ambiguous", "Tell me about oceans and coral reefs", {"p42": 2, "p22": 1, "p41": 1}),
    ("q36", "edge_ambiguous", "Popular hot caffeinated beverages", {"p33": 2, "p34": 2, "p49": 1}),
    ("q37", "edge_ambiguous", "Facts about famous mountains", {"p19": 2, "p47": 2}),
    ("q38", "edge_ambiguous", "Widely used programming languages", {"p23": 2, "p25": 2}),
]


def build() -> dict:
    passages = [{"id": pid, "topic": topic, "text": text} for pid, topic, text in PASSAGES]
    passage_ids = {p["id"] for p in passages}

    queries = []
    for qid, category, text, qrels in QUERIES:
        for pid in qrels:
            if pid not in passage_ids:
                raise ValueError(f"{qid}: qrel references unknown passage {pid}")
        queries.append({"id": qid, "category": category, "text": text, "qrels": qrels})

    return {
        "meta": {
            "language": "en",
            "domain": "mixed_general_knowledge",
            "n_passages": len(passages),
            "n_queries": len(queries),
            "grade_scale": {"2": "highly relevant", "1": "partially relevant"},
        },
        "passages": passages,
        "queries": queries,
    }


def validate(dataset: dict) -> None:
    passages, queries = dataset["passages"], dataset["queries"]
    assert 45 <= len(passages) <= 60, f"unexpected passage count {len(passages)}"
    assert 35 <= len(queries) <= 45, f"unexpected query count {len(queries)}"

    ids = [p["id"] for p in passages]
    assert len(ids) == len(set(ids)), "duplicate passage ids"
    qids = [q["id"] for q in queries]
    assert len(qids) == len(set(qids)), "duplicate query ids"

    # `no_match` must be unanswerable; `edge_negation` may be (an intentionally
    # unanswerable negation like "hot drinks with no caffeine"); every other
    # category must have at least one relevance judgment.
    for q in queries:
        if q["category"] == "no_match":
            assert not q["qrels"], f"{q['id']} is no_match but has qrels"
        elif q["category"] == "edge_negation":
            pass
        else:
            assert q["qrels"], f"{q['id']} ({q['category']}) has no qrels"

    categories = sorted({q["category"] for q in queries})
    print(f"  categories ({len(categories)}): {', '.join(categories)}")


def main() -> int:
    dataset = build()
    validate(dataset)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}")
    print(f"  passages: {dataset['meta']['n_passages']}")
    print(f"  queries : {dataset['meta']['n_queries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
