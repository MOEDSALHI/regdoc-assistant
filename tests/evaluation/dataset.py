"""
Ground truth dataset for RAGAS evaluation.
12 questions covering RGPD Articles 5, 17, 32, 35, 37 and CNIL sanctions.
Each entry maps to a specific article to verify retrieval precision.
"""

GROUND_TRUTH_QA = [
    # --- Article 5 — Principes ---
    {
        "question": "Quels sont les six principes fondamentaux du traitement des données personnelles selon le RGPD ?",
        "ground_truth": (
            "Le RGPD établit six principes : licéité, loyauté et transparence ; "
            "limitation des finalités ; minimisation des données ; exactitude ; "
            "limitation de la conservation ; intégrité et confidentialité. "
            "Le responsable du traitement est accountable du respect de ces principes."
        ),
        "source_article": "Article 5",
    },
    {
        "question": "Qu'est-ce que le principe de minimisation des données selon l'Article 5 du RGPD ?",
        "ground_truth": (
            "Les données doivent être adéquates, pertinentes et limitées à ce qui est "
            "nécessaire au regard des finalités pour lesquelles elles sont traitées. "
            "On ne collecte que ce dont on a réellement besoin."
        ),
        "source_article": "Article 5",
    },
    # --- Article 17 — Droit à l'effacement ---
    {
        "question": "Dans quels cas une personne peut-elle exercer son droit à l'effacement selon l'Article 17 ?",
        "ground_truth": (
            "Le droit à l'effacement s'applique quand les données ne sont plus nécessaires, "
            "quand la personne retire son consentement, quand elle s'oppose au traitement, "
            "quand le traitement est illicite, ou pour respecter une obligation légale. "
            "Ce droit ne s'applique pas si le traitement est nécessaire à la liberté d'expression "
            "ou à des fins archivistiques d'intérêt public."
        ),
        "source_article": "Article 17",
    },
    {
        "question": "Quelles sont les exceptions au droit à l'effacement prévues par le RGPD ?",
        "ground_truth": (
            "Les exceptions incluent : l'exercice du droit à la liberté d'expression, "
            "le respect d'une obligation légale, des raisons d'intérêt public dans le domaine "
            "de la santé publique, des fins archivistiques, de recherche scientifique ou historique, "
            "ou la constatation, l'exercice ou la défense de droits en justice."
        ),
        "source_article": "Article 17",
    },
    # --- Article 32 — Sécurité ---
    {
        "question": "Quelles mesures techniques et organisationnelles le RGPD exige-t-il pour assurer la sécurité des données ?",
        "ground_truth": (
            "L'Article 32 exige des mesures appropriées au risque : pseudonymisation et chiffrement, "
            "moyens pour garantir la confidentialité, l'intégrité, la disponibilité et la résilience "
            "des systèmes, capacité à rétablir l'accès en cas d'incident, et procédure de test "
            "et d'évaluation régulière des mesures de sécurité."
        ),
        "source_article": "Article 32",
    },
    {
        "question": "Comment évaluer le niveau de sécurité approprié selon l'Article 32 du RGPD ?",
        "ground_truth": (
            "Le niveau de sécurité doit tenir compte de l'état des connaissances, "
            "des coûts de mise en œuvre, de la nature des données, de la portée et du contexte "
            "du traitement, ainsi que des risques pour les droits et libertés des personnes. "
            "Le risque s'évalue en termes de probabilité et de gravité."
        ),
        "source_article": "Article 32",
    },
    # --- Article 35 — AIPD ---
    {
        "question": "Quand une analyse d'impact relative à la protection des données (AIPD) est-elle obligatoire ?",
        "ground_truth": (
            "L'AIPD est obligatoire quand le traitement est susceptible d'engendrer un risque élevé "
            "pour les droits et libertés des personnes, notamment : profilage systématique, "
            "traitement à grande échelle de données sensibles, surveillance systématique "
            "d'une zone accessible au public."
        ),
        "source_article": "Article 35",
    },
    {
        "question": "Que doit contenir une analyse d'impact (AIPD) selon l'Article 35 du RGPD ?",
        "ground_truth": (
            "L'AIPD doit contenir : une description systématique des traitements envisagés "
            "et leurs finalités, une évaluation de la nécessité et de la proportionnalité, "
            "une évaluation des risques pour les droits et libertés, et les mesures envisagées "
            "pour faire face aux risques. La consultation du DPO est requise."
        ),
        "source_article": "Article 35",
    },
    # --- Article 37 — DPO ---
    {
        "question": "Dans quels cas la désignation d'un délégué à la protection des données (DPO) est-elle obligatoire ?",
        "ground_truth": (
            "Le DPO est obligatoire dans trois cas : le traitement est effectué par une autorité "
            "ou un organisme public, les activités de base du responsable exigent un suivi "
            "régulier et systématique à grande échelle des personnes, ou les activités de base "
            "portent sur le traitement à grande échelle de données sensibles ou relatives "
            "à des condamnations pénales."
        ),
        "source_article": "Article 37",
    },
    {
        "question": "Quelles sont les missions du délégué à la protection des données selon le RGPD ?",
        "ground_truth": (
            "Le DPO informe et conseille le responsable du traitement, contrôle le respect "
            "du RGPD, dispense des conseils sur l'AIPD, coopère avec l'autorité de contrôle "
            "et fait office de point de contact avec celle-ci. Il doit être associé à toutes "
            "les questions relatives à la protection des données personnelles."
        ),
        "source_article": "Article 37",
    },
    # --- CNIL Sanctions ---
    {
        "question": "Quels sont les deux niveaux d'amendes administratives prévus par le RGPD ?",
        "ground_truth": (
            "Le RGPD prévoit deux niveaux : jusqu'à 10 millions d'euros ou 2% du chiffre "
            "d'affaires annuel mondial pour les infractions moins graves (obligations du responsable, "
            "certification) ; jusqu'à 20 millions d'euros ou 4% du chiffre d'affaires pour "
            "les infractions les plus graves (principes de base, droits des personnes, "
            "transferts internationaux)."
        ),
        "source_article": "Article 83",
    },
    {
        "question": "Quels critères la CNIL utilise-t-elle pour déterminer le montant d'une sanction ?",
        "ground_truth": (
            "La CNIL prend en compte : la nature, la gravité et la durée de la violation, "
            "le caractère intentionnel ou négligent, les mesures prises pour atténuer le dommage, "
            "le degré de coopération avec l'autorité, les catégories de données concernées, "
            "la manière dont la CNIL a eu connaissance de la violation, et les précédents."
        ),
        "source_article": "Article 83",
    },
]